import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { toFrame } from './landmarks';
import type { Tracker, TrackingFrame } from './types';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

/** A failure we can explain to the player. */
export class CameraError extends Error {}

/** Webcam + MediaPipe hand and pose models, polled once per new video frame. */
export class CameraTracker implements Tracker {
  private lastVideoTime = -1;

  private constructor(
    readonly video: HTMLVideoElement,
    private hands: HandLandmarker,
    private pose: PoseLandmarker,
    private stream: MediaStream,
  ) {}

  static async create(onStatus: (message: string) => void): Promise<CameraTracker> {
    onStatus('Asking for camera access…');
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
    } catch (e) {
      const blocked = e instanceof DOMException && e.name === 'NotAllowedError';
      throw new CameraError(blocked ? 'Camera access was blocked.' : 'No camera was found.');
    }
    const video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    onStatus('Loading hand and pose tracking…');
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      const [hands, pose] = await Promise.all([
        HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        }),
        PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numPoses: 1,
        }),
      ]);
      return new CameraTracker(video, hands, pose, stream);
    } catch {
      stream.getTracks().forEach(t => t.stop());
      throw new CameraError('Could not load the tracking models. Check your internet connection.');
    }
  }

  poll(now: number): TrackingFrame | null {
    if (this.video.readyState < 2 || this.video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = this.video.currentTime;
    const hands = this.hands.detectForVideo(this.video, now);
    const pose = this.pose.detectForVideo(this.video, now);
    return toFrame(now / 1000, hands.landmarks, pose.landmarks[0]);
  }

  dispose(): void {
    this.stream.getTracks().forEach(t => t.stop());
    this.hands.close();
    this.pose.close();
  }
}
