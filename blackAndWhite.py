import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_hands = hands.process(image_rgb)
                results_face = face_mesh.process(image_rgb)
                results_pose = pose.process(image_rgb)

                blank_image = np.zeros_like(frame)

                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            blank_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

                if results_pose.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        blank_image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

                cv2.imshow('Frame', blank_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()