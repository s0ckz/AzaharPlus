#!/bin/bash -ex

export NDK_CCACHE=$(which ccache)

if [ ! -z "${ANDROID_KEYSTORE_B64}" ]; then
    export ANDROID_KEYSTORE_FILE="${GITHUB_WORKSPACE}/ks.jks"
    base64 --decode <<< "${ANDROID_KEYSTORE_B64}" > "${ANDROID_KEYSTORE_FILE}"
fi

cd src/android
chmod +x ./gradlew

# On feature branches (not master/tags) only build the vanilla flavor to cut
# iteration time in half. bundleVanillaRelease is still needed because the
# copyBundleVanillaRelease task — which is what populates build/bundle/ that
# pack.sh reads — is wired up via finalizedBy(bundle<Variant>). --stacktrace
# is always on so failures surface a usable stack instead of the opaque
# IncrementalSplitterRunnable wrapper error.
if [[ "${GITHUB_REF}" == "refs/heads/master" || "${GITHUB_REF_TYPE}" == "tag" ]]; then
    ./gradlew --stacktrace assembleRelease
    ./gradlew --stacktrace bundleRelease
else
    ./gradlew --stacktrace assembleVanillaRelease bundleVanillaRelease
fi

ccache -s -v

if [ ! -z "${ANDROID_KEYSTORE_B64}" ]; then
    rm "${ANDROID_KEYSTORE_FILE}"
fi
