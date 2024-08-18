import cv2
import mediapipe as mp
import numpy as np
import socket
import json
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
    if landmark.visibility < 0.5:
        return None
    return (landmark.x, landmark.y)

def get_landmark_coordinates_depth(landmarks, landmark_name):
    landmark = landmarks[landmark_name.value]
    return (landmark.x, landmark.y, landmark.z)
# This is so bad so im not using it. 

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
        if l_shoulder[1] + 0.15 < l_shoulder_prev[1] and r_shoulder[1]  + 0.15< r_shoulder_prev[1]:
            return True
    return False


def is_attack(landmarks, image, previous_frames):
    l_elbow = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    l_wrist = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    r_elbow = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
    r_wrist = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
    if l_elbow is None or l_wrist is None or r_elbow is None or r_wrist is None:
        return {"type":"No attack"}
    # l_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    # r_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    # try:
    #     if calculate_angle(l_shoulder, l_elbow, l_wrist) < 60 or calculate_angle(r_shoulder, r_elbow, r_wrist) < 60:
    #         return {"type":"resting"}
    # except:
    #     pass
    height, width, _ = image.shape

    for frame in list(previous_frames)[-5:]:
        l_wrist_prev = get_landmark_coordinates(frame, mp_pose.PoseLandmark.LEFT_WRIST)
        r_wrist_prev = get_landmark_coordinates(frame, mp_pose.PoseLandmark.RIGHT_WRIST)
        if l_wrist_prev is None or r_wrist_prev is None:
            continue

        # Calculate the vectors from elbow to wrist
        l_arm_direction = np.array(l_wrist) - np.array(l_elbow)
        r_arm_direction = np.array(r_wrist) - np.array(r_elbow)

        # Calculate the movement vectors
        l_movement_direction = np.array(l_wrist) - np.array(l_wrist_prev)
        r_movement_direction = np.array(r_wrist) - np.array(r_wrist_prev)

        # Normalize the vectors
        l_arm_direction_norm = l_arm_direction / np.linalg.norm(l_arm_direction)
        r_arm_direction_norm = r_arm_direction / np.linalg.norm(r_arm_direction)
        l_movement_direction_norm = l_movement_direction / np.linalg.norm(l_movement_direction)
        r_movement_direction_norm = r_movement_direction / np.linalg.norm(r_movement_direction)

        # Calculate the dot product to check alignment between movement and arm direction
        l_dot_product = np.dot(l_arm_direction_norm, l_movement_direction_norm)
        r_dot_product = np.dot(r_arm_direction_norm, r_movement_direction_norm)

        # Calculate the angle between the vectors
        l_angle = np.arccos(l_dot_product)
        r_angle = np.arccos(r_dot_product)

        left = np.linalg.norm(l_movement_direction) > np.linalg.norm(l_arm_direction) and l_angle < np.pi/3.3
        right = np.linalg.norm(r_movement_direction) > np.linalg.norm(r_arm_direction) and r_angle < np.pi/3.3

        if left and right:
            if l_wrist[1]+0.3<l_wrist_prev[1] and r_wrist[1]+0.3<r_wrist_prev[1]:
                cv2.arrowedLine(image, tuple(np.multiply(l_wrist_prev, [width, height]).astype(int)), tuple(np.multiply(l_wrist, [width, height]).astype(int)), (0, 0, 255), 2, tipLength=0.5)
                cv2.arrowedLine(image, tuple(np.multiply(r_wrist_prev, [width, height]).astype(int)), tuple(np.multiply(r_wrist, [width, height]).astype(int)), (0, 0, 255), 2, tipLength=0.5)
                return {"type":"uppercut", "direction":"both"}

            cv2.arrowedLine(image, tuple(np.multiply(l_wrist_prev, [width, height]).astype(int)), tuple(np.multiply(l_wrist, [width, height]).astype(int)), (0, 0, 255), 2, tipLength=0.5)
            cv2.arrowedLine(image, tuple(np.multiply(r_wrist_prev, [width, height]).astype(int)), tuple(np.multiply(r_wrist, [width, height]).astype(int)), (0, 0, 255), 2, tipLength=0.5)
            return {"type":"burst", "direction":"both"}
        if left:  # Threshold for direction match
            cv2.arrowedLine(image, tuple(np.multiply(l_wrist_prev, [width, height]).astype(int)), tuple(np.multiply(l_wrist, [width, height]).astype(int)), (0, 0, 255), 2, tipLength=0.5)
            return {"type":"strike", "direction":"left", "prev_cord":l_wrist_prev, "curr_cord":l_wrist}
        
        if right:  # Threshold for direction match
            cv2.arrowedLine(image, tuple(np.multiply(r_wrist_prev, [width, height]).astype(int)), tuple(np.multiply(r_wrist, [width, height]).astype(int)), (0, 0, 255), 2, tipLength=0.5)
            return {"type":"strike", "direction":"right", "prev_cord":r_wrist_prev, "curr_cord":r_wrist}

    return {"type":"None"}


        
    
    
    


def run_pose_detection(sockets, serverAddressPort):
    global jump, lean
    jump = False
    lean = "center"
    attack = {"type":"resting"}
    # Create a deque with a maximum size of 10
    previous_frames = deque(maxlen=10)  

    cap = cv2.VideoCapture(1)
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                attack = is_attack(landmarks, image, previous_frames)
                previous_frames.append(landmarks)
                lean = lean_direction(landmarks, image)
                jump = is_jump(landmarks, previous_frames)
                data = {
                    "jump": jump,
                    "lean": lean,
                    "attack": attack
                }
                jsonData = json.dumps(data)
                sockets.sendto(str.encode(jsonData), serverAddressPort)
            except Exception as e:
                print(e)
                pass

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'Attack', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(attack["type"]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Lean Direction', (65, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(lean), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'Jump', (165, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, str(jump), (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sockets = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5052)
    run_pose_detection(sockets, serverAddressPort)
