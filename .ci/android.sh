#!/bin/bash -ex

export NDK_CCACHE=$(which ccache)

if [ ! -z "${ANDROID_KEYSTORE_B64}" ]; then
    export ANDROID_KEYSTORE_FILE="${GITHUB_WORKSPACE}/ks.jks"
    base64 --decode <<< "${ANDROID_KEYSTORE_B64}" > "${ANDROID_KEYSTORE_FILE}"
fi

cd src/android
chmod +x ./gradlew
# --stacktrace: last CI run failed at packageGooglePlayRelease inside
# IncrementalSplitterRunnable with no diagnostic info, and Gradle's own output
# literally said "Run with --stacktrace option to get the stack trace". Add it
# unconditionally so any future failure gives us the actual exception instead of
# a generic wrapper class name.
./gradlew assembleRelease --stacktrace
./gradlew bundleRelease --stacktrace

ccache -s -v

if [ ! -z "${ANDROID_KEYSTORE_B64}" ]; then
    rm "${ANDROID_KEYSTORE_FILE}"
fi
