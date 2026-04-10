# Building Azahar locally

This branch ships a Dockerized Android build so you can produce a signed
sideload-installable APK without installing the Android SDK/NDK on the host.

## Prerequisites

- Docker Desktop (running, with the drive containing this repo enabled under
  Settings → Resources → File sharing)
- `adb` in PATH (only needed for installing the APK)

## Build

```bash
bash docker/android-build/build-android.sh
```

That's it. Default behavior:

- Builds the toolchain image on first run (slow, ~2.5 GB, cached after that)
- Runs `./gradlew assembleVanillaRelease --stacktrace` inside the container
- Builds **arm64-v8a only** (skips x86_64 to halve build time)
- Uses named Docker volumes `azahar-android-ccache` and `azahar-android-gradle`
  so subsequent builds are incremental
- Signs with the persistent keystore at `.local-keystore/keystore.jks` if
  present; otherwise falls back to the Gradle debug keystore

Output:

```
src/android/app/build/outputs/apk/vanilla/release/app-vanilla-release.apk
```

## Install on device

```bash
adb install -r src/android/app/build/outputs/apk/vanilla/release/app-vanilla-release.apk
```

## Variations

**Build both ABIs (arm64 + x86_64, e.g. for the Android emulator):**
```bash
AZAHAR_ABI_FILTER=arm64-v8a,x86_64 bash docker/android-build/build-android.sh
```

**Pass arbitrary Gradle tasks** — anything after the script name is forwarded
to `./gradlew`:
```bash
bash docker/android-build/build-android.sh clean assembleVanillaRelease
bash docker/android-build/build-android.sh assembleRelease   # both flavors
```

**Drop into an interactive shell inside the build container** (workspace and
caches mounted, useful for poking at gradle state):
```bash
bash docker/android-build/build-android.sh shell
```

**Force a clean rebuild of the toolchain image** (only needed if you edit
the Dockerfile):
```bash
bash docker/android-build/build-android.sh --rebuild-image
```

## Persistent local keystore (optional but recommended)

Without a persistent keystore, Gradle signs with a debug keystore that AGP
regenerates per Gradle home. This causes `INSTALL_FAILED_UPDATE_INCOMPATIBLE`
("signature mismatch") on the second and subsequent sideloads, forcing an
uninstall+reinstall and losing app state.

To set one up once:

```bash
mkdir -p .local-keystore

docker run --rm -v "$(pwd)/.local-keystore:/keystore" \
    azahar-android-build:latest \
    keytool -genkeypair -keystore /keystore/keystore.jks \
        -storepass CHANGEME -alias azahar-local -keypass CHANGEME \
        -keyalg RSA -keysize 2048 -validity 36500 \
        -dname "CN=Azahar Local Build, OU=Dev, O=Local, L=Local, ST=Local, C=US"

cat > .local-keystore/keystore.env <<'EOF'
ANDROID_KEYSTORE_PASS=CHANGEME
ANDROID_KEY_ALIAS=azahar-local
EOF
```

Both `keystore.jks` and `keystore.env` are gitignored. Subsequent builds pick
them up automatically.

## Wiping caches

```bash
docker volume rm azahar-android-ccache   # NDK clang object cache
docker volume rm azahar-android-gradle   # Gradle wrapper, deps, build cache
```

## Windows / Git Bash notes

- The script sets `MSYS_NO_PATHCONV=1` automatically to keep Git Bash from
  mangling Docker volume mount paths.
- First build is slow because of host→container file sync; subsequent builds
  touch much less data.
