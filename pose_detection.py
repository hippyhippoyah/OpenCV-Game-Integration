import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables for Pygame integration
jump = False
lean = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coordinates(landmarks, landmark_name):
    landmark = landmarks[landmark_name.value]
    return (landmark.x, landmark.y)

def is_resting(landmarks, image):
    l_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    l_elbow = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    l_wrist = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    r_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    r_elbow = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
    r_wrist = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
    l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    height, width, _ = image.shape
    cv2.putText(image, str(l_angle), tuple(np.multiply(l_elbow, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(r_angle), tuple(np.multiply(r_elbow, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return l_angle < 50 and r_angle < 50

def lean_direction(landmarks, image):
    l_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    height, width, _ = image.shape
    cv2.putText(image, str(l_shoulder[1]), tuple(np.multiply(l_shoulder, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(r_shoulder[1]), tuple(np.multiply(r_shoulder, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    if l_shoulder[1] > r_shoulder[1] + 0.08:
        return "left"
    elif r_shoulder[1] > l_shoulder[1] + 0.08:
        return "right"
    else:
        return "center"

def is_jump(landmarks, previous_frames):
    l_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    for frame in previous_frames:
        l_shoulder_prev = get_landmark_coordinates(frame, mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_shoulder_prev = get_landmark_coordinates(frame, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        if l_shoulder[1] > l_shoulder_prev[1] + 0.15 and r_shoulder[1] > r_shoulder_prev[1] + 0.15:
            return True
    return False

# Create a deque with a maximum size of 10
previous_frames = deque(maxlen=10)

def run_pose_detection(queue):
    global jump, lean
    jump = False
    lean = "Center"
    resting = False

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # print("Frame read")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                previous_frames.append(landmarks)
                resting = is_resting(landmarks, image)
                lean = lean_direction(landmarks, image)
                jump = is_jump(landmarks, previous_frames)
                data = {
                    "jump": jump,         # Replace with actual condition
                    "lean": lean,       # Replace with actual condition
                    "attack": False         # Replace with actual condition
                }
                queue.put(data)
            except Exception as e:
                print(e)
                pass

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'Resting', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(resting), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Lean Direction', (65, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(lean), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Jump', (165, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(jump), (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# run_pose_detection()
# start_pose_detection()

