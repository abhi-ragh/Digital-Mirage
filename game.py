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
left_hand_img = pygame.image.load("models/left_hand.png").convert_alpha()
right_hand_img = pygame.image.load("models/right_hand.png").convert_alpha()

# Combine the character images
character_height = head_img.get_height() + body_img.get_height()
character_img = pygame.Surface((body_img.get_width(), character_height), pygame.SRCALPHA)
character_img.blit(body_img, (0, 0))
character_img.blit(head_img, (body_img.get_width() // 2 - head_img.get_width() // 2, body_img.get_height()))

# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Open the camera
cap = cv2.VideoCapture(0)

def get_body_position(results_pose):
    if results_pose.pose_landmarks:
        return (int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * WINDOW_WIDTH),
                int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * WINDOW_HEIGHT))

def get_arm_positions_and_rotations(results_pose):
    arm_positions = []
    arm_rotations = []
    if results_pose.pose_landmarks:
        # Left arm
        left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_arm_position = (int(left_elbow.x * WINDOW_WIDTH), int(left_elbow.y * WINDOW_HEIGHT))
        arm_positions.append(left_arm_position)

        left_arm_rotation = math.atan2(left_elbow.y - left_shoulder.y, left_elbow.x - left_shoulder.x)
        arm_rotations.append(left_arm_rotation)

        # Right arm
        right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_arm_position = (int(right_elbow.x * WINDOW_WIDTH), int(right_elbow.y * WINDOW_HEIGHT))
        arm_positions.append(right_arm_position)

        right_arm_rotation = math.atan2(right_elbow.y - right_shoulder.y, right_elbow.x - right_shoulder.x)
        arm_rotations.append(right_arm_rotation)

    return arm_positions, arm_rotations

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

    # Define fixed positions for the hands
    left_hand_pos = (character_rect.centerx - 50, character_rect.centery)
    right_hand_pos = (character_rect.centerx + 50, character_rect.centery)

    # Draw the hands at fixed positions
    display.blit(left_hand_img, left_hand_pos)
    display.blit(right_hand_img, right_hand_pos)


def draw_arms(display, arm_positions, arm_rotations):
    # Adjust this value to bring the arms closer to the body
    arm_offset_x = 100

    for i, arm_pos in enumerate(arm_positions):
        arm_img_to_use = right_hand_img if i == 1 else left_hand_img
        rotated_arm_img = pygame.transform.rotate(arm_img_to_use, math.degrees(arm_rotations[i]))
        arm_rect = rotated_arm_img.get_rect()

        # Adjust the arm position based on the offset
        if i == 0:  # Left arm
            arm_rect.center = (arm_pos[0] - arm_offset_x, arm_pos[1])
        else:  # Right arm
            arm_rect.center = (arm_pos[0] + arm_offset_x, arm_pos[1])

        display.blit(rotated_arm_img, arm_rect)

def main():
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_hands = hands.process(image_rgb)  # This is still needed for head rotation
                results_pose = pose.process(image_rgb)

                # Clear the display
                display.fill((255, 255, 255))

                # Get the user's body position
                body_pos = get_body_position(results_pose)

                # Get the user's arm positions and orientations
                arm_positions, arm_rotations = get_arm_positions_and_rotations(results_pose)

                # Get the head rotation
                head_rotation = get_head_rotation(results_pose)

                # Overlay the character image based on the tracked positions and rotations
                draw_character(display, head_rotation)

                # Overlay the arm images based on the tracked positions and rotations
                draw_arms(display, arm_positions, arm_rotations)

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
