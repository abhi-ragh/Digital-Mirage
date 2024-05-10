import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

def draw_stickman(image, pose_landmarks):
    # Create a white background image
    white_image = np.zeros(image.shape, dtype=np.uint8)
    white_image.fill(255)

    if pose_landmarks is None:
        return white_image

    # Define the body parts and their connections
    connections = [
        (0, 11), (0, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # hips
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28)   # right leg
    ]

    # Draw the stickman lines
    for connection in connections:
        start_index, end_index = connection
        start_landmark = pose_landmarks.landmark[start_index]
        end_landmark = pose_landmarks.landmark[end_index]
        start_x, start_y = int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0])
        end_x, end_y = int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0])
        cv2.line(white_image, (start_x, start_y), (end_x, end_y), (0, 0, 0), 2)

    # Draw the head
    nose_landmark = pose_landmarks.landmark[0]
    nose_x, nose_y = int(nose_landmark.x * image.shape[1]), int(nose_landmark.y * image.shape[0])
    head_radius = 15
    cv2.circle(white_image, (nose_x, nose_y - head_radius), head_radius, (0, 0, 0), -1)

    return white_image

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with mp_face.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results_hands = hands.process(image_rgb)
                results_face = face_detection.process(image_rgb)
                results_pose = pose.process(image_rgb)

                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if results_face.detections:
                    for detection in results_face.detections:
                        mp_drawing.draw_detection(frame, detection)

                stickman_image = draw_stickman(frame, results_pose.pose_landmarks)
                cv2.imshow('Frame', stickman_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()