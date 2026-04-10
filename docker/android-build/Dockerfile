# syntax=docker/dockerfile:1.6
#
# Local Android build image for the 30fps optimization sprint.
#
# Pinned to the same toolchain versions the upstream CI uses so the local
# build matches what GitHub Actions produces:
#
#   - JDK 17 (eclipse-temurin)
#   - Android compileSdk 35  (src/android/app/build.gradle.kts)
#   - Android build-tools 35.0.0
#   - NDK 27.1.12297006      (build.gradle.kts ndkVersion)
#   - CMake 3.30.3           (.github/workflows/build.yml installs this)
#   - ccache                 (NDK_CCACHE wraps clang invocations)
#
# Caches: ccache and gradle home are NOT baked into this image; they're
# expected to be mounted as named docker volumes by build-android.sh, so the
# image stays small-ish and the caches survive image rebuilds.

FROM eclipse-temurin:17-jdk-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    ANDROID_SDK_ROOT=/opt/android-sdk \
    ANDROID_HOME=/opt/android-sdk \
    NDK_CCACHE=/usr/bin/ccache \
    CCACHE_DIR=/ccache \
    CCACHE_BASEDIR=/workspace \
    CCACHE_COMPRESS=1 \
    CCACHE_COMPILERCHECK=content \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime \
    GRADLE_USER_HOME=/root/.gradle \
    PATH="/opt/android-sdk/cmdline-tools/latest/bin:/opt/android-sdk/platform-tools:${PATH}"

# Host packages: ccache for the NDK build, ninja for cmake, python3 because
# some Android Gradle tasks invoke it, file/unzip/wget for sdkmanager bootstrap,
# git for getGitVersion() in build.gradle.kts, libc6-i386/libstdc++6 for any
# 32-bit binaries the NDK might still ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ccache \
        ca-certificates \
        file \
        git \
        libstdc++6 \
        ninja-build \
        python3 \
        unzip \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap the Android command-line tools. The version number is the Google
# build number; bump if Google retires this URL. After unzip the directory
# layout has to be cmdline-tools/latest/{bin,lib} for sdkmanager to find its
# own resources.
ARG CMDLINE_TOOLS_VERSION=11076708
RUN mkdir -p ${ANDROID_SDK_ROOT}/cmdline-tools && \
    wget -q "https://dl.google.com/android/repository/commandlinetools-linux-${CMDLINE_TOOLS_VERSION}_latest.zip" \
        -O /tmp/cmdline-tools.zip && \
    unzip -q /tmp/cmdline-tools.zip -d ${ANDROID_SDK_ROOT}/cmdline-tools && \
    mv ${ANDROID_SDK_ROOT}/cmdline-tools/cmdline-tools ${ANDROID_SDK_ROOT}/cmdline-tools/latest && \
    rm /tmp/cmdline-tools.zip

# Accept all SDK licenses non-interactively. yes(1) pipes through 'y' until
# sdkmanager closes the pipe; the || true catches the SIGPIPE exit.
RUN yes 2>/dev/null | sdkmanager --licenses >/dev/null || true

# Install the SDK components needed by the android gradle build. Pinned versions
# match build.gradle.kts and the CI workflow exactly. NDK is ~1.8 GB so this is
# the slow layer of the image build (~3-5 minutes on a fast connection).
RUN sdkmanager --install \
        "platform-tools" \
        "platforms;android-35" \
        "build-tools;35.0.0" \
        "ndk;27.1.12297006" \
        "cmake;3.30.3" \
    >/dev/null

# Create the cache mount points so the volumes have the right ownership when
# mounted by build-android.sh. (Docker mounts inherit UID/GID from the volume,
# but pre-creating the dirs avoids permission surprises on first run.)
RUN mkdir -p /ccache /root/.gradle /workspace

WORKDIR /workspace/src/android

# Default command — point users at the wrapper script. build-android.sh
# overrides this with the actual gradle invocation.
CMD ["bash", "-c", "echo 'azahar-android-build toolchain image. Use docker/android-build/build-android.sh from the repo root.'"]
