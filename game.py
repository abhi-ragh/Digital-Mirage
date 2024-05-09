import cv2
import mediapipe as mp
import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the display
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D Character Tracking")

# Load character images
head_img = pygame.transform.rotate(pygame.image.load("models/head.png").convert_alpha(), 180)
body_img = pygame.image.load("models/body.png").convert_alpha()
left_hand_img = pygame.image.load("models/right_hand.png").convert_alpha()
right_hand_img = pygame.image.load("models/left_hand.png").convert_alpha()
left_forearm_img = pygame.image.load("models/right_forearm.png").convert_alpha()
right_forearm_img = pygame.image.load("models/left_forearm.png").convert_alpha()

# Combine the character images
character_height = head_img.get_height() + body_img.get_height()
character_img = pygame.Surface((body_img.get_width(), character_height), pygame.SRCALPHA)
character_img.blit(body_img, (0, 0))
character_img.blit(head_img, (body_img.get_width() // 2 - head_img.get_width() // 2, body_img.get_height()))

# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open the camera
cap = cv2.VideoCapture(0)

def get_head_rotation(results_pose):
    if results_pose.pose_landmarks:
        left_eye = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
        return math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
    return 0

def draw_character(display, head_rotation):
    # Define fixed position for the body image
    body_pos = (WINDOW_WIDTH // 2, WINDOW_HEIGHT + 400)

    # Draw the body image without rotation
    character_rect = character_img.get_rect()
    character_rect.midbottom = body_pos
    display.blit(character_img, character_rect)

    # Draw the head image on top of the body with rotation
    head_offset = (0, -head_img.get_height() // 2)  # Offset to position head on top of body
    rotated_head_img = pygame.transform.rotate(head_img, math.degrees(head_rotation))
    head_rect = rotated_head_img.get_rect(center=(400, 465))
    display.blit(rotated_head_img, head_rect)

    return character_rect

def get_arm_rotations(results_pose):
    arm_rotations = [0, 0, 0, 0]  # Initialize with default values [left_upper_arm, right_upper_arm, left_forearm, right_forearm]
    if results_pose.pose_landmarks:
        # Left arm
        left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        if left_shoulder and left_elbow:
            arm_rotations[0] = math.atan2(left_elbow.y - left_shoulder.y, left_elbow.x - left_shoulder.x)
        if left_elbow and left_wrist:
            arm_rotations[2] = math.atan2(left_wrist.y - left_elbow.y, left_wrist.x - left_elbow.x)

        # Right arm
        right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        if right_shoulder and right_elbow:
            arm_rotations[1] = math.atan2(right_elbow.y - right_shoulder.y, right_elbow.x - right_shoulder.x)
        if right_elbow and right_wrist:
            arm_rotations[3] = math.atan2(right_wrist.y - right_elbow.y, right_wrist.x - right_elbow.x)

    # Adjust the rotation angles as per your requirements
    arm_rotations[0] = arm_rotations[0] - 1.52
    arm_rotations[1] = arm_rotations[1] - 1.52
    arm_rotations[2] = arm_rotations[2] - 1.52
    arm_rotations[3] = arm_rotations[3] - 1.52

    return arm_rotations

def draw_arms(display, arm_rotations, character_rect):
    # Define fixed positions for the arms
    left_arm_pos = (355, 495)
    left_forearm_pos = (330, 540)
    
    right_arm_pos = (445, 480)
    right_forearm_pos = (480, 520)

    # Draw the arms at fixed positions with rotations
    rotated_left_hand_img = pygame.transform.rotate(left_hand_img, math.degrees(arm_rotations[0]))
    rotated_right_hand_img = pygame.transform.rotate(right_hand_img, math.degrees(arm_rotations[1]))

    rotated_left_forearm_img = pygame.transform.rotate(left_forearm_img, math.degrees(arm_rotations[2]))
    rotated_right_forearm_img = pygame.transform.rotate(right_forearm_img, math.degrees(arm_rotations[3]))

    left_hand_rect = rotated_left_hand_img.get_rect(center=left_arm_pos)
    right_hand_rect = rotated_right_hand_img.get_rect(center=right_arm_pos)

    left_forearm_rect = rotated_left_forearm_img.get_rect(center=left_forearm_pos)
    #left_forearm_rect.midtop = left_hand_rect.midbottom

    right_forearm_rect = rotated_right_forearm_img.get_rect(center=right_forearm_pos)
    #right_forearm_rect.midtop = right_hand_rect.midbottom

    display.blit(rotated_left_hand_img, left_hand_rect)
    display.blit(rotated_right_hand_img, right_hand_rect)
    display.blit(rotated_left_forearm_img, left_forearm_rect)
    display.blit(rotated_right_forearm_img, right_forearm_rect)

def main():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(image_rgb)

            # Clear the display
            display.fill((255, 255, 255))

            # Get the head rotation
            head_rotation = get_head_rotation(results_pose)

            # Overlay the character image based on the tracked positions and rotations
            character_rect = draw_character(display, head_rotation)

            # Get the user's arm rotations
            arm_rotations = get_arm_rotations(results_pose)

            # Overlay the arm images based on the tracked positions and rotations
            draw_arms(display, arm_rotations, character_rect)

            # Update the display
            pygame.display.update()

            # Check for the 'q' key to quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit()

if __name__ == "__main__":
    main()
