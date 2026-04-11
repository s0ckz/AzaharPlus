// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "video_core/gpu.h"
#include "video_core/gpu_worker.h"

namespace VideoCore {

namespace {
// Set once the worker's Loop begins. Producer wrappers on GPU test this
// to decide between "push to queue" (caller is emu thread) and "run
// synchronously on the worker" (caller is the worker, re-entering via
// SignalInterruptForThread → GPU::SetBufferSwap or similar).
thread_local bool t_on_worker_thread = false;
} // namespace

bool GpuWorker::IsOnWorkerThread() {
    return t_on_worker_thread;
}

GpuWorker::GpuWorker() = default;

GpuWorker::~GpuWorker() {
    // Belt-and-suspenders: GPU::~GPU is responsible for calling Stop()
    // explicitly BEFORE the Impl unique_ptr destructor runs so the worker
    // doesn't observe a null this->impl mid-shutdown. If that was missed,
    // at least try to stop now; the member destruction order will cover
    // the rest.
    Stop();
}

void GpuWorker::Start(GPU* gpu) {
    if (running.load(std::memory_order_acquire)) {
        return;
    }
    gpu_owner = gpu;
    // Mark running BEFORE spawning the thread. The thread's Loop() only
    // reads gpu_owner after dequeueing a message, by which time Start()
    // has long returned and running=true is visible.
    running.store(true, std::memory_order_release);
    thread = std::make_unique<std::jthread>([this](std::stop_token stop) { Loop(stop); });
    LOG_INFO(HW_GPU, "GPU worker thread started");
}

void GpuWorker::Stop() {
    if (!running.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if (thread) {
        thread->request_stop();
        // Poke the condvar so the worker observes the stop_token.
        {
            std::scoped_lock lock{queue_mutex};
            queue.push(GpuCmdFlush{});
        }
        queue_cv.notify_one();
        thread.reset(); // joins
    }
    gpu_owner = nullptr;
    LOG_INFO(HW_GPU, "GPU worker thread stopped");
}

void GpuWorker::Push(GpuMessage msg) {
    {
        std::scoped_lock lock{queue_mutex};
        queue.push(std::move(msg));
    }
    queue_cv.notify_one();
}

void GpuWorker::Flush() {
    if (!running.load(std::memory_order_acquire)) {
        return;
    }
    if (IsOnWorkerThread()) {
        // Already draining from the worker itself (e.g. Flush() called
        // recursively from inside an ExecuteOnWorker branch). Nothing to
        // do — the current message is being processed.
        return;
    }
    flush_ack.Reset();
    Push(GpuCmdFlush{});
    flush_ack.Wait();
}

void GpuWorker::Loop(std::stop_token stop) {
    t_on_worker_thread = true;

    while (!stop.stop_requested()) {
        GpuMessage msg;
        {
            std::unique_lock lock{queue_mutex};
            queue_cv.wait(lock, [this, &stop] {
                return !queue.empty() || stop.stop_requested();
            });
            if (stop.stop_requested() && queue.empty()) {
                return;
            }
            msg = std::move(queue.front());
            queue.pop();
        }

        std::visit(
            [this](auto&& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                if constexpr (std::is_same_v<T, GpuCmdExecute>) {
                    gpu_owner->ExecuteOnWorker(cmd.command);
                } else if constexpr (std::is_same_v<T, GpuCmdVBlank>) {
                    gpu_owner->VBlankOnWorker(cmd.cycles_late);
                } else if constexpr (std::is_same_v<T, GpuCmdSetBufferSwap>) {
                    gpu_owner->SetBufferSwapOnWorker(cmd.screen_id, cmd.info);
                } else if constexpr (std::is_same_v<T, GpuCmdSetColorFill>) {
                    gpu_owner->SetColorFillOnWorker(cmd.raw);
                } else if constexpr (std::is_same_v<T, GpuCmdFlush>) {
                    flush_ack.Set();
                }
            },
            msg);
    }
}

} // namespace VideoCore
