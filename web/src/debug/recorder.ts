import type { Landmark } from '../input/landmarks';
import type { TrackingFrame } from '../input/types';
import type { Calibration } from '../intent/calibration';
import { TUNING, type Intent } from '../intent/interpret';

/** The raw MediaPipe output a frame was built from, so detection can be re-run offline. */
export interface RawLandmarks {
  hands: Landmark[][];
  handsWorld: Landmark[][];
  pose: Landmark[];
  poseWorld: Landmark[];
}

export interface Recording {
  version: 1;
  recordedAt: string;
  calibration: Calibration;
  tuning: typeof TUNING;
  samples: { frame: TrackingFrame; raw: RawLandmarks | null; intent: Intent }[];
}

/** Records a few seconds of tracking (numbers only, no video) for offline debugging. */
export class Recorder {
  private rec: Recording | null = null;
  private until = 0;
  private startT: number | null = null;
  private seconds = 0;

  constructor(private onDone: (r: Recording) => void) {}

  get active(): boolean { return this.rec !== null; }

  start(seconds: number, calibration: Calibration): void {
    this.seconds = seconds;
    this.startT = null;
    this.rec = { version: 1, recordedAt: new Date().toISOString(), calibration, tuning: { ...TUNING }, samples: [] };
  }

  push(frame: TrackingFrame, raw: RawLandmarks | null, intent: Intent): void {
    if (!this.rec) return;
    if (this.startT === null) {
      this.startT = frame.t;
      this.until = frame.t + this.seconds;
    }
    this.rec.samples.push(structuredClone({ frame, raw, intent }));
    if (frame.t >= this.until) {
      const done = this.rec;
      this.rec = null;
      this.onDone(done);
    }
  }
}

/** Save a recording as a JSON download. */
export function downloadRecording(r: Recording): void {
  const blob = new Blob([JSON.stringify(r)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `firebending-recording-${r.recordedAt.replace(/[:.]/g, '-')}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
