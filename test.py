import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)  # 0 for webcam

def draw_landmarks(image, results):
  # Iterate over the drawing styles
  mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                           mp_drawing.DrawingSpec(color=(80, 95, 247), thickness=2, circle_radius=1),
                           mp_drawing.DrawingSpec(color=(80, 247, 95), thickness=2, circle_radius=1)
                           )
  # Similarly draw hand landmarks and pose landmarks based on results.hand_landmarks and results.pose_landmarks
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()

    # Process the image with MediaPipe Holistic
    results = holistic.process(image)

    # Draw landmarks on the image (if desired)
    if results.pose_landmarks:
      draw_landmarks(image, results)

    # Access and analyze pose/hand/face landmarks from results object (refer to MediaPipe documentation for details on structure)
    # Example: Head pose estimation using Euler angles (pitch, yaw, roll) can be extracted from results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].visibility

    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to quit
      break

cap.release()
