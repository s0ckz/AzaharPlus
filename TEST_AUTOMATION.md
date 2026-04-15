# MK7 Test Automation

## Device
- Anbernic RG DS (RK3568, Mali G52)
- ADB over WiFi: `adb connect 192.168.1.51:<PORT>`
- Port changes on every reconnect

## ROM Path
```
/storage/267a9ea9-618a-440c-9edc-aa8a3f6935f4/games-internal/roms/3ds/Mario Kart 7 (Europe) (En,Fr,De,Es,It,Nl,Pt,Ru) (Rev 2).3ds
```

## Display
- Physical: 640x480
- Landscape orientation

## Full Test Procedure

### 1. Force stop + clear logs + launch
```bash
adb -s $DEVICE shell "am force-stop io.github.lime3ds.android"
sleep 2
adb -s $DEVICE shell "logcat -c"
adb -s $DEVICE shell "am start -n io.github.lime3ds.android/org.citra.citra_emu.activities.EmulationActivity \
  -a android.intent.action.VIEW \
  -d 'file:///storage/267a9ea9-618a-440c-9edc-aa8a3f6935f4/games-internal/roms/3ds/Mario%20Kart%207%20(Europe)%20(En,Fr,De,Es,It,Nl,Pt,Ru)%20(Rev%202).3ds'"
```

### 2. Wait for game to load (~20s)
```bash
sleep 20
adb -s $DEVICE shell "logcat -d | grep 'speed=' | grep -v 'speed=0.0' | tail -3"
```
Expect: `speed=100%` or similar. If `speed=0.0` only → game didn't load or hung.

### 3. Dismiss Mii dialog
On first run, a "No Mii characters" dialog appears. Tap OK:
```bash
adb -s $DEVICE shell "input tap 320 350"
```
Wait ~15s for the game to load through the title screen to main menu.

### 4. Verify main menu reached
```bash
sleep 15
# Screenshot (use MSYS_NO_PATHCONV=1 on Windows/Git Bash):
MSYS_NO_PATHCONV=1 adb -s $DEVICE shell "screencap -d 0 -p /data/local/tmp/screen.png"
MSYS_NO_PATHCONV=1 adb -s $DEVICE pull /data/local/tmp/screen.png screen_check.png
```
Expect: Mario Kart 7 main menu (Single Player, Local Multiplayer, etc.)

### 5. Monitor for crash (loop)
The main menu loads random course previews in the background, which triggers surface creation/destruction — the crash trigger.

```bash
for i in $(seq 1 20); do
  sleep 30
  CRASHED=$(adb -s $DEVICE shell "logcat -d | grep 'signal 11' | grep -c 'VulkanWorker'" 2>/dev/null)
  SPEED=$(adb -s $DEVICE shell "logcat -d | grep 'speed=' | tail -1" 2>/dev/null | sed 's/.*speed=/speed=/')
  echo "T+$((i*30))s: crashes=$CRASHED $SPEED"
  if [ "$CRASHED" -gt 0 ] 2>/dev/null; then
    echo "CRASHED!"
    break
  fi
done
```

### 6. After crash — collect logs
```bash
# Crash info
adb -s $DEVICE shell "logcat -d | grep -iE 'VkCrashDump|signal 11' | tail -5"
# Last speed before crash
adb -s $DEVICE shell "logcat -d | grep 'speed=' | grep -v 'speed=0.0' | tail -3"
# NON-WORKER descriptor flushes
adb -s $DEVICE shell "logcat -d | grep 'NON-WORKER' | wc -l"
# Latest tombstone
adb -s $DEVICE shell "head -35 $(adb -s $DEVICE shell 'ls -t /data/tombstones/tombstone_* | head -1')"
```

## One-liner: Full test cycle
```bash
DEVICE=192.168.1.51:40591 && \
adb -s $DEVICE shell "am force-stop io.github.lime3ds.android" && sleep 2 && \
adb -s $DEVICE shell "logcat -c" && \
adb -s $DEVICE shell "am start -n io.github.lime3ds.android/org.citra.citra_emu.activities.EmulationActivity -a android.intent.action.VIEW -d 'file:///storage/267a9ea9-618a-440c-9edc-aa8a3f6935f4/games-internal/roms/3ds/Mario%20Kart%207%20(Europe)%20(En,Fr,De,Es,It,Nl,Pt,Ru)%20(Rev%202).3ds'" && \
sleep 20 && \
adb -s $DEVICE shell "input tap 320 350" && \
sleep 15 && \
echo "=== Monitoring ===" && \
for i in $(seq 1 20); do sleep 30; C=$(adb -s $DEVICE shell "logcat -d | grep 'signal 11' | grep -c 'VulkanWorker'" 2>/dev/null); S=$(adb -s $DEVICE shell "logcat -d | grep 'speed=' | tail -1" 2>/dev/null | sed 's/.*speed=/speed=/'); echo "T+$((i*30))s: crashes=$C $S"; if [ "$C" -gt 0 ] 2>/dev/null; then echo "CRASHED at T+$((i*30))s"; break; fi; done
```

## FPS Baseline (current build: priority FlushRegion + VBlank routing + lightweight Event proxy)

| Metric | Value |
|---|---|
| Speed | ~100% |
| sysFPS | 59-60 |
| gameFPS | 59-60 |
| Frame time | 7-8ms |
| gpu | 0.08-0.15ms |
| rest | -3 to -5ms (headroom) |
| Crash time | ~2.5-6 min from main menu |

Loading phases show 17-47% speed as expected (shader compilation, texture loading).

**222 samples over ~3.8 minutes (detailed):**
- 84% of samples at 90%+ speed
- Frame time: median=13ms, avg=18ms, p95=38ms, max=515ms (loading spike)
- GPU time: median=0.12ms, avg=0.23ms, p95=1.13ms, max=3.13ms
- Drops below 60% are from course preview transitions (loading textures)

Any fix that drops steady-state FPS below 50 is unacceptable.

## Success Criteria

**A fix is only considered successful if the game runs for 1 HOUR at the main menu without crashing.** The main menu cycles through random course previews which triggers surface creation/destruction — the crash trigger. Any crash within 1 hour = not fixed.

## Crash patterns observed
- **`fault addr 0x60`**: Mali driver internal corruption from concurrent Vulkan API calls (GpuWorker `vkUpdateDescriptorSets` + VulkanWorker `vkCmd*`)
- **`fault addr 0x765ff80014`**: Use-after-free of specific VkImage during init (Fill surface sentencing underflow)
- **Deadlock (speed=0.0%)**: Priority FlushRegion channel + Handle::Create WaitWorker blocking GpuWorker
- Typical crash time: 2-8 minutes from main menu with course preview cycling
