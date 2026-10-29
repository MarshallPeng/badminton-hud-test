# badminton-hud

Adds a game-style HUD to a single-rally badminton clip (doubles, filmed from behind
the baseline, hand-held is fine).

```
pip install -e .
hud IMG_5013.mov
```

Writes next to the clip:

| file | what |
|---|---|
| `IMG_5013_HUD.mp4` | the clip with the HUD and the original audio |
| `IMG_5013_HUD_report.html` (+ `IMG_5013_HUD_report/`) | error check: court lines on 8 frames, every shot at the hit and 0.2 s later, long stretches without a shot, warnings |
| `IMG_5013_HUD_fixes.toml` | fixes and player names; edit and run `hud` again |
| `IMG_5013_HUD_work/` | cached steps (delete it, or use `--redo <step>`, to recompute) |

Options: `--out-dir DIR`, `--redo {prepare,people,camera,court,identity,shuttle}`,
`--model yolo11m.pt`. A GPU is used when available. On a 4-core CPU, an 12 s clip takes
about 15 minutes, mostly YOLO.

## On screen

- Foot rings lying flat on the court (a 0.55 m circle projected through the court homography).
- Name tags above heads.
- LOCK ON reticle on the shuttle, with the hitter's name and the shot's speed.
- Top bar: RALLY TIME (from the serve), RALLY COUNT (shots so far), SHOT SPEED (last shot).
- Bottom left: stamina and distance run for all 4 players.
- Bottom right: COURT MAP with the players, the current shot and the shuttle.

## How it works

1. **Prepare.** ffmpeg to 1080x1920 (portrait input) or 1920x1080 (landscape input). Extract the audio and find racket sounds: onsets in the 2–8 kHz band.
2. **Camera motion.** Frame-to-frame homography from features on the ceiling and walls (everything that is not green floor and not a person).
3. **People.** YOLO11m + ByteTrack on every frame, imgsz 1280.
4. **Court in the first frame.**
   - Find white floor lines with Hough.
   - Find two vanishing points: lines running away from the camera, and lines parallel to the net. Lettering is rejected because it is wider than a painted line.
   - Take the two nearest net-parallel lines as the baseline (0 m) and the doubles long service line (0.76 m).
   - Match the running-away lines to x = 0, 0.46, 3.05, 5.64, 6.10 m with a cross-ratio fit.
   - Build the homography and refine it on every matched line. The best fit is the one covering the most white paint, with the 4 players standing on it.
5. **Court in every frame.** Predict from the camera motion, snap onto the white lines (near half only), and every 2nd frame re-read the lines from the picture. Keep the fit that covers more paint.
6. **Who is who.** Torso and leg colour, side of the net, last position, and box size. A player whose feet are hidden behind someone nearer the camera is matched on head position. Player 1/2 are on the camera's side (1 = darker torso), 3/4 on the far side.
7. **Shuttle.**
   - Candidates: small bright blobs brighter than the frames ±3 and ±10. Each neighbour frame is aligned twice, once by the background and once by the floor plane. Blobs on people's torsos and legs are ignored.
   - Tracklets: candidates are linked frame to frame and cut at sharp turns.
   - Path: one path through the rally that may only change direction at a racket of one of the 4 players. A flight that was never seen between two players of opposite teams can be bridged. This drops shuttles from the next courts.
8. **Shots.**
   - Hits come from that path. Hit times snap to a racket sound within 0.17 s.
   - Racket sounds fill in shots where sides would not alternate, or where shots are more than 2.4 s apart.
   - The hitter is the team member closer to the shuttle, measured relative to body height.
   - The rally ends when the shuttle is seen lying or sliding on the floor.
   - Launch speed = (e^(kD) − 1) / (kT), k = 9.81 / 6.8², D = hitter → next hitter (m), T = flight time. For the last shot, D runs to where the shuttle landed.
9. **HUD.** Drawn with Pillow (Rajdhani font, OFL). The original audio is added back.
10. **Report.** Warnings for speeds above 250 or below 25 km/h, for shots less than 0.33 s or more than 2.4 s apart, and for shots that were not seen on the shuttle path.

## Fixes

Edit `<clip>_HUD_fixes.toml` and run `hud <clip>` again:

```toml
names = ["Anna", "Ben", "Chris", "Dee"]
remove_shot = [6.93]        # shot times (s) to delete, ±0.12 s

[rally]
end_time = 9.60             # when the rally ends (s)

[[add_shot]]
time = 8.30
player = 4                  # optional, 1-4
```

Stamina is a game value, not a measurement. It drops with distance run and with sprints above 3 m/s, and recovers when a player stands still.
