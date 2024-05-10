import cv2
import mediapipe as mp
import pygame
import math
import random

pygame.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D Character Tracking")

air_index = 100
temp = 25 

head_img = pygame.transform.rotate(pygame.image.load("models/head.png").convert_alpha(), 180)
body_img = pygame.image.load("models/body.png").convert_alpha()
body_cold_img = pygame.image.load("models/body_cold.png").convert_alpha() 
left_hand_img = pygame.image.load("models/right_hand.png").convert_alpha()
right_hand_img = pygame.image.load("models/left_hand.png").convert_alpha()
left_forearm_img = pygame.image.load("models/right_forearm.png").convert_alpha()
right_forearm_img = pygame.image.load("models/left_forearm.png").convert_alpha()

head_mask_img = pygame.transform.rotate(pygame.image.load("models/head_mask.png").convert_alpha(), 180)

character_height = head_img.get_height() + body_img.get_height()
character_img = pygame.Surface((body_img.get_width(), character_height), pygame.SRCALPHA)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

def reset_character_img(character_img):
    character_img.blit(body_img, (0, 0))
    character_img.blit(head_img, (body_img.get_width() // 2 - head_img.get_width() // 2, body_img.get_height()))

reset_character_img(character_img)

def handle_air_index_keys(event):
    global air_index
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
            air_index = min(air_index + 10, 100)
        elif event.key == pygame.K_DOWN:
            air_index = max(air_index - 10, 0)

def handle_temp_keys(event):
    global temp
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
            temp += 1
            if temp >= 15:
                reset_character_img(character_img)
        elif event.key == pygame.K_s:
            temp -= 1

def draw_air_index(display, air_index):
   font = pygame.font.Font(None, 36)
   text = font.render(f"Air Index: {air_index}", True, (0, 0, 0))
   display.blit(text, (WINDOW_WIDTH - text.get_width() - 10, 10))

def draw_temp(display, temp):
   font = pygame.font.Font(None, 36)
   text = font.render(f"Temp: {temp}°C", True, (0, 0, 0))
   display.blit(text, (10, 10))

def draw_air_quality_message(display, message):
   font = pygame.font.Font(None, 36)
   text = font.render(message, True, (0, 0, 0))
   display.blit(text, (100, 100))

def get_head_rotation(results_pose):
   if results_pose.pose_landmarks:
       left_eye = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
       right_eye = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
       return math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
   return 0

def add_shivering_effect(img, intensity=5):
   shivering_img = img.copy()
   for x in range(shivering_img.get_width()):
       for y in range(shivering_img.get_height()):
           if random.randint(0, 100) < intensity:
               shivering_img.set_at((x, y), (0, 0, 0, 0))
   return shivering_img

def draw_character(display, head_rotation, head_img, temp):
    body_pos = (WINDOW_WIDTH // 2, WINDOW_HEIGHT + 400)

    if temp < 15: 
        shivering_body_img = add_shivering_effect(body_cold_img)
        shivering_head_img = add_shivering_effect(head_img)
        character_img.blit(shivering_body_img, (0, 0))
        rotated_head_img = pygame.transform.rotate(shivering_head_img, math.degrees(head_rotation))
    else:
        rotated_head_img = pygame.transform.rotate(head_img, math.degrees(head_rotation))

    character_rect = character_img.get_rect()
    character_rect.midbottom = body_pos
    display.blit(character_img, character_rect)

    head_rect = rotated_head_img.get_rect(center=(400, 465))
    display.blit(rotated_head_img, head_rect)

    return character_rect

def get_arm_rotations(results_pose):
   arm_rotations = [0, 0, 0, 0]
   if results_pose.pose_landmarks:
       left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
       left_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
       left_wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
       if left_shoulder and left_elbow:
           arm_rotations[0] = math.atan2(left_elbow.y - left_shoulder.y, left_elbow.x - left_shoulder.x)

       if left_elbow and left_wrist:
           arm_rotations[2] = math.atan2(left_wrist.y - left_elbow.y, left_wrist.x - left_elbow.x)

       right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
       right_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
       right_wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
       if right_shoulder and right_elbow:
           arm_rotations[1] = math.atan2(right_elbow.y - right_shoulder.y, right_elbow.x - right_shoulder.x)
       if right_elbow and right_wrist:
           arm_rotations[3] = math.atan2(right_wrist.y - right_elbow.y, right_wrist.x - right_elbow.x)

   arm_rotations[0] = arm_rotations[0] - 1.52
   arm_rotations[0] = max(arm_rotations[0], -1.0947271096735096)
   arm_rotations[1] = arm_rotations[1] - 1.52
   arm_rotations[1] = min(arm_rotations[1], 1.2097018048643866)
   arm_rotations[1] = max(arm_rotations[1], 0.2)
   arm_rotations[2] = arm_rotations[2] - 1.52
   arm_rotations[3] = arm_rotations[3] - 1.52

   return arm_rotations

def draw_arms(display, arm_rotations, character_rect, temp):
   left_arm_pos = (355, 495)
   right_arm_pos = (445, 480)

   if temp >= 15: 
       rotated_left_hand_img = pygame.transform.rotate(left_hand_img, math.degrees(arm_rotations[0]))
       rotated_right_hand_img = pygame.transform.rotate(right_hand_img, math.degrees(arm_rotations[1]))

       left_hand_rect = rotated_left_hand_img.get_rect(center=left_arm_pos)
       right_hand_rect = rotated_right_hand_img.get_rect(center=right_arm_pos)

       display.blit(rotated_left_hand_img, left_hand_rect)
       display.blit(rotated_right_hand_img, right_hand_rect)

       left_elbow_pos = ((left_hand_rect.centerx + math.cos(arm_rotations[0]) * left_forearm_img.get_width() // 2) - 60,
                         (left_hand_rect.centery + math.sin(arm_rotations[0]) * left_forearm_img.get_width() // 2) + 60)

       right_elbow_pos = ((right_hand_rect.centerx - math.cos(arm_rotations[1]) * right_forearm_img.get_width() // 2) + 60,
                          (right_hand_rect.centery - math.sin(arm_rotations[1]) * right_forearm_img.get_width() // 2) + 60)

       rotated_left_forearm_img = pygame.transform.rotate(left_forearm_img, math.degrees(arm_rotations[2]))
       rotated_right_forearm_img = pygame.transform.rotate(right_forearm_img, math.degrees(arm_rotations[3]))

       left_forearm_rect = rotated_left_forearm_img.get_rect(center=left_elbow_pos)
       right_forearm_rect = rotated_right_forearm_img.get_rect(center=right_elbow_pos)

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

            display.fill((255, 255, 255))

            head_rotation = get_head_rotation(results_pose)

            if air_index < 40:
                character_rect = draw_character(display, head_rotation, head_mask_img, temp)
                draw_air_quality_message(display, "Feeling the air's not right?Remember to Wear Your Mask!")
            else:
                character_rect = draw_character(display, head_rotation, head_img, temp)

            arm_rotations = get_arm_rotations(results_pose)
            draw_arms(display, arm_rotations, character_rect, temp)

            draw_air_index(display, air_index)
            draw_temp(display, temp)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit()
                handle_air_index_keys(event)
                handle_temp_keys(event)

if __name__ == "__main__":
    main()