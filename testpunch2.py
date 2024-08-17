import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
punch_speed_threshold = 1  # Adjust based on experiment
punch_distance_threshold = 0.25  # Adjust based on experiment
buffer_size = 10  # Number of frames to consider
cooldown_frames = 30  # Number of frames for cooldown
arrow_display_frames = 10  # Number of frames to display the arrow

# Variables
cooldown_counter = 0
arrow_display_counter = 0
arrow_start_point = (0, 0)
arrow_end_point = (0, 0)

# Function to calculate the speed of the hand
def calculate_speed(points, dt):
    dist = np.linalg.norm(points[1] - points[0])
    speed = dist / dt
    return speed

# Function to detect if hand is a fist
def is_fist(landmarks):
    finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    finger_bases = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]

    for tip, base in zip(finger_tips, finger_bases):
        if tip.y < base.y:
            return False
    return True

# Function to calculate the direction vector
def calculate_direction_vector(landmarks_buffer):
    if len(landmarks_buffer) >= 2:
        return landmarks_buffer[-1] - landmarks_buffer[0]
    return np.array([0, 0, 0])

# Capture video from webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    landmarks_buffer = []
    time_buffer = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect hands
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to numpy array
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

                # Detect if the hand is a fist
                if is_fist(hand_landmarks.landmark):

                    # Add the current landmarks and timestamp to the buffer
                    landmarks_buffer.append(landmarks[0])
                    time_buffer.append(time.time())

                    # Ensure the buffer does not exceed the specified size
                    if len(landmarks_buffer) > buffer_size:
                        landmarks_buffer.pop(0)
                        time_buffer.pop(0)

                    # Calculate the direction vector
                    direction_vector = calculate_direction_vector(landmarks_buffer)

                    # Normalize the direction vector to get the direction
                    fist_direction = direction_vector / np.linalg.norm(direction_vector) if np.linalg.norm(direction_vector) > 0 else np.array([0, 0, 0])

                    # Cooldown logic
                    if cooldown_counter == 0:
                        # Calculate the average speed and movement over the buffer
                        if len(landmarks_buffer) >= 2:
                            for i in range(1, min(len(landmarks_buffer), 9)):
                                dt = time_buffer[-1] - time_buffer[-(i + 1)]
                                speed = calculate_speed([landmarks_buffer[-1], landmarks_buffer[-(i + 1)]], dt)
                                distance = np.linalg.norm(landmarks_buffer[-1] - landmarks_buffer[-(i + 1)])

                                # Check if the punch direction matches the fist direction
                                if speed > punch_speed_threshold and distance > punch_distance_threshold:
                                    print(f"Punch detected: Speed: {speed:.2f} m/s, Distance: {distance:.2f} m, Direction: {fist_direction}")
                                    cooldown_counter = cooldown_frames  # Start cooldown

                                    # Prepare arrow for display
                                    arrow_start_point = tuple((landmarks[0][:2] * [frame.shape[1], frame.shape[0]]).astype(int))
                                    arrow_end_point = tuple((arrow_start_point + (fist_direction[:2] * 100)).astype(int))
                                    arrow_display_counter = arrow_display_frames  # Set arrow display counter
                                    break
                    else:
                        cooldown_counter -= 1  # Decrement cooldown counter

        # Display the arrow if within display frames
        if arrow_display_counter > 0:
            cv2.arrowedLine(image, arrow_start_point, arrow_end_point, (0, 255, 0), 5)
            arrow_display_counter -= 1

        # Display the frame
        cv2.imshow('Punch Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
