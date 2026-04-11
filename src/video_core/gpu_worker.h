// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <variant>

#include "common/common_types.h"
#include "common/thread.h"
#include "core/hle/service/gsp/gsp_command.h"
#include "core/hle/service/gsp/gsp_gpu.h"

namespace VideoCore {

class GPU;

// Producer-consumer message types for the GpuWorker thread. Each message
// type corresponds to a public GPU method that used to run synchronously
// on the emu thread. The worker dequeues in FIFO order so PICA register
// writes stay in-order with draws and framebuffer swaps.
struct GpuCmdExecute {
    Service::GSP::Command command;
};

struct GpuCmdVBlank {
    s64 cycles_late;
};

struct GpuCmdSetBufferSwap {
    u32 screen_id;
    Service::GSP::FrameBufferInfo info;
};

struct GpuCmdSetColorFill {
    u32 raw;
};

struct GpuCmdFlush {};

using GpuMessage = std::variant<GpuCmdExecute, GpuCmdVBlank, GpuCmdSetBufferSwap,
                                GpuCmdSetColorFill, GpuCmdFlush>;

class GpuWorker {
public:
    GpuWorker();
    ~GpuWorker();

    GpuWorker(const GpuWorker&) = delete;
    GpuWorker& operator=(const GpuWorker&) = delete;
    GpuWorker(GpuWorker&&) = delete;
    GpuWorker& operator=(GpuWorker&&) = delete;

    // Spawns the worker thread. Called once from GPU::GPU() after the Impl
    // is constructed. Idempotent.
    void Start(GPU* gpu);

    // Signals stop_token, wakes the worker, and joins. Called from
    // GPU::~GPU BEFORE the impl unique_ptr destructor runs — unique_ptr::
    // reset nulls the pointer *before* running the deleter, so if the
    // worker is still running when impl destructs, its gpu_owner->impl
    // load reads nullptr and SIGSEGVs at the offset of any Impl member.
    void Stop();

    // Push a message onto the queue. Called from the emu thread only.
    void Push(GpuMessage msg);

    // Synchronous drain barrier: pushes GpuCmdFlush and blocks until the
    // worker signals flush_ack. Used to establish read-after-write
    // ordering for MMIO paths (ReadReg/WriteReg) and to quiesce before
    // rasterizer destruction (RecreateRenderer / ReleaseRenderer).
    void Flush();

    bool IsRunning() const {
        return running.load(std::memory_order_acquire);
    }

    // True if the caller is running on the worker thread. Producer
    // wrappers use this to avoid self-deadlock when re-entered (e.g.
    // SignalInterruptForThread → GPU::SetBufferSwap from the worker's
    // PDC interrupt path).
    static bool IsOnWorkerThread();

private:
    void Loop(std::stop_token stop);

    GPU* gpu_owner{nullptr};
    std::atomic<bool> running{false};
    std::unique_ptr<std::jthread> thread;

    // Plain mutex + queue + condvar. Not SPSCQueue because the codebase
    // has two conflicting Common::SPSCQueue templates whose headers clash
    // when both are transitively included — the per-push mutex is tens of
    // cycles, negligible next to the microseconds each worker message
    // represents.
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::queue<GpuMessage> queue;

    // Flush handshake. The worker calls flush_ack.Set() when it dequeues
    // a GpuCmdFlush; the emu thread waits on it inside Flush().
    Common::Event flush_ack;
};

} // namespace VideoCore
