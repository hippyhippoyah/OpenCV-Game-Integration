# Tracking recordings

Drop files saved with the K key here. `src/intent/recordings.test.ts` replays them through the
current detector, so real-camera behaviour is checked on every test run. Fixtures are trimmed to
the tracking frames (no raw landmarks) plus what fired at the time:

- `punches-close.json` — alternating jabs from about 0.83 m (sensitivity ×1.4).
- `punches-2.json` — jabs and combos, some thrown mid-lean.
- `leaning.json` — leaning and ducking at about 1.3 m; fast leans used to fire punches.
- `swaying.json` — only swaying side to side; it fired 9 punches at the sway turnarounds.
