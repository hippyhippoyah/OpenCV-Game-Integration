# Tracking recordings

Drop files saved with the K key here. `src/intent/recordings.test.ts` replays them through the
current detector, so real-camera behaviour is checked on every test run. Fixtures are trimmed to
the tracking frames (no raw landmarks) plus what fired at the time:

- `punches-close.json` — alternating jabs from about 0.83 m (sensitivity ×1.4).
- `punches-2.json` — jabs and combos, some thrown mid-lean.
- `punches-3.json` — alternating jabs about every half second; recorded in open-hand mode by
  mistake (P), where only 7 fired. The fist detector finds 19.
- `leaning.json` — leaning and ducking at about 1.3 m; fast leans used to fire punches.
- `swaying.json` — only swaying side to side; it fired 9 punches at the sway turnarounds.
- `raised-hands-1.json`, `raised-hands-2.json` — no punches: fists held up by and over the head
  (charging, trying the slam) and lowered again. Face overlap made the depth jump; 22 and 13 punches
  fired. Now at most 2.
- `charge-high.json` — about 11 right punches, most thrown from a fist cocked high by or over the
  head (charging). Only the 4 near head height fired; now 8, 5 of them charged.
