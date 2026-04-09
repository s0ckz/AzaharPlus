// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include <SoundTouch.h>

#include "audio_core/audio_types.h"
#include "audio_core/time_stretch.h"
#include "common/assert.h"
#include "common/logging/log.h"

namespace AudioCore {

TimeStretcher::TimeStretcher() : sound_touch(std::make_unique<soundtouch::SoundTouch>()) {
    sound_touch->setChannels(2);
    sound_touch->setSampleRate(native_sample_rate);
    sound_touch->setPitch(1.0);
    sound_touch->setTempo(1.0);
}

TimeStretcher::~TimeStretcher() = default;

void TimeStretcher::SetOutputSampleRate(unsigned int sample_rate) {
    sound_touch->setSampleRate(sample_rate);
}

std::size_t TimeStretcher::Process(const s16* in, std::size_t num_in, s16* out,
                                   std::size_t num_out) {
    const double time_delta = static_cast<double>(num_out) / native_sample_rate; // seconds
    double current_ratio = static_cast<double>(num_in) / static_cast<double>(num_out);

    const double max_latency = 0.25; // seconds
    const double max_backlog = native_sample_rate * max_latency;
    const double backlog_fullness = sound_touch->numSamples() / max_backlog;
    if (backlog_fullness > 4.0) {
        // Too many samples in backlog: Don't push anymore on
        num_in = 0;
    }

    // We ideally want the backlog to be about 50% full.
    // This gives some headroom both ways to prevent underflow and overflow.
    // We tweak current_ratio to encourage this.
    constexpr double tweak_time_scale = 0.050; // seconds
    const double tweak_correction = (backlog_fullness - 0.5) * (time_delta / tweak_time_scale);
    current_ratio *= std::pow(1.0 + 2.0 * tweak_correction, tweak_correction < 0 ? 3.0 : 1.0);

    // This low-pass filter smoothes out variance in the calculated stretch ratio.
    // The time-scale determines how responsive this filter is.
    constexpr double lpf_time_scale = 0.712; // seconds
    const double lpf_gain = 1.0 - std::exp(-time_delta / lpf_time_scale);
    stretch_ratio += lpf_gain * (current_ratio - stretch_ratio);

    // Place a lower limit of 5% speed. When a game boots up, there will be
    // many silence samples. These do not need to be timestretched.
    stretch_ratio = std::max(stretch_ratio, 0.05);
    sound_touch->setTempo(stretch_ratio);

    LOG_TRACE(Audio, "{:5}/{:5} ratio:{:0.6f} backlog:{:0.6f}", num_in, num_out, stretch_ratio,
              backlog_fullness);

    if constexpr (std::is_floating_point<soundtouch::SAMPLETYPE>()) {
        // SoundTouch on most systems expects float samples. Use persistent
        // member-owned scratch buffers rather than per-call std::vector
        // construction — on weak handheld SoCs (Cortex-A55) those heap calls
        // showed up as a real chunk of the audio thread's budget.
        //
        // Note: this whole branch is compiled out on Android, where Azahar's
        // bundled SoundTouch is built with `short` SAMPLETYPE and the
        // `else if` below is taken instead.
        if (float_scratch_in.size() < 2 * num_in) {
            float_scratch_in.resize(2 * num_in);
        }
        if (float_scratch_out.size() < 2 * num_out) {
            float_scratch_out.resize(2 * num_out);
        }

        // Precompute the s16 → float scale factor once so the inner loop is
        // a hot multiply-only sequence rather than a divide-per-sample; A55
        // FDIV latency is ~14 cycles whereas FMUL is ~3.
        constexpr float kS16ToFloat = 1.0f / 32768.0f;
        constexpr float kFloatToS16 = 32767.0f;

        for (std::size_t i = 0; i < (2 * num_in); ++i) {
            float_scratch_in[i] = static_cast<float>(in[i]) * kS16ToFloat;
        }

        sound_touch->putSamples(reinterpret_cast<soundtouch::SAMPLETYPE*>(float_scratch_in.data()),
                                static_cast<u32>(num_in));

        const std::size_t samples_received = sound_touch->receiveSamples(
            reinterpret_cast<soundtouch::SAMPLETYPE*>(float_scratch_out.data()),
            static_cast<u32>(num_out));

        // Convert back to s16 with saturation so loud peaks don't wrap.
        const std::size_t out_samples = 2 * samples_received;
        for (std::size_t i = 0; i < out_samples; ++i) {
            float v = float_scratch_out[i] * kFloatToS16;
            if (v > 32767.0f) v = 32767.0f;
            else if (v < -32768.0f) v = -32768.0f;
            out[i] = static_cast<s16>(v);
        }

        return samples_received;
    } else if (std::is_same<soundtouch::SAMPLETYPE, s16>()) {
        // Use reinterpret_cast to workaround compile error when SAMPLETYPE is float.
        sound_touch->putSamples(reinterpret_cast<const soundtouch::SAMPLETYPE*>(in),
                                static_cast<u32>(num_in));
        return sound_touch->receiveSamples(reinterpret_cast<soundtouch::SAMPLETYPE*>(out),
                                           static_cast<u32>(num_out));
    } else {
        static_assert(std::is_floating_point<soundtouch::SAMPLETYPE>() ||
                      std::is_same<soundtouch::SAMPLETYPE, s16>());
        UNREACHABLE_MSG("Invalid SAMPLETYPE {}", typeid(soundtouch::SAMPLETYPE).name());
        return 0;
    }
}

void TimeStretcher::Clear() {
    sound_touch->clear();
}

void TimeStretcher::Flush() {
    sound_touch->flush();
}

} // namespace AudioCore
