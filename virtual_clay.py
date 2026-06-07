import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os

def draw_3d_wireframe(img, X, Y, Z, elev, azim):
    h, w = img.shape[:2]
    rad_a = np.radians(azim)
    rad_e = np.radians(elev)
    
    X_rot1 = X * np.cos(rad_a) - Y * np.sin(rad_a)
    Y_rot1 = X * np.sin(rad_a) + Y * np.cos(rad_a)
    Z_rot1 = Z
    
    y_screen = Z_rot1 * np.cos(rad_e) - Y_rot1 * np.sin(rad_e)
    
    scale = min(w, h) / 6.0
    sx = (X_rot1 * scale + w / 2).astype(np.int32)
    sy = (-y_screen * scale + h / 2).astype(np.int32)
    
    rows, cols = X.shape
    color = (180, 255, 120) 
    
    for i in range(rows):
        for j in range(cols - 1):
            cv2.line(img, (sx[i, j], sy[i, j]), (sx[i, j+1], sy[i, j+1]), color, 1)
    for i in range(rows - 1):
        for j in range(cols):
            cv2.line(img, (sx[i, j], sy[i, j]), (sx[i+1, j], sy[i+1, j]), color, 1)

def is_fist(hand_landmarks):
    fist_score = 0
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        dist_tip = (hand_landmarks[tip].x - hand_landmarks[0].x)**2 + (hand_landmarks[tip].y - hand_landmarks[0].y)**2
        dist_pip = (hand_landmarks[pip].x - hand_landmarks[0].x)**2 + (hand_landmarks[pip].y - hand_landmarks[0].y)**2
        if dist_tip < dist_pip:
            fist_score += 1
    return fist_score == 4 

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)

RADIUS = 2.0 
X_orig = RADIUS * np.outer(np.cos(u), np.sin(v))
Y_orig = RADIUS * np.outer(np.sin(u), np.sin(v))
Z_orig = RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

X_clay = X_orig.copy()
Y_clay = Y_orig.copy()
Z_clay = Z_orig.copy()

elev_angle = 20.0
azim_angle = 0.0
global_scale_z = 1.0   
global_scale_xy = 1.0  

target_hands = 0
stable_timer = 0
active_mode_hands = 0

is_running = True
prev_hand_x = None
prev_hand_y = None

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", model_path)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2 
)
landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

while cap.isOpened() and is_running:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
    deform_mode = None
    status_text = "System Ready"
    status_color = (200, 200, 200)
    
    current_detected_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
    if current_detected_hands > 0:
        for hand_idx in range(current_detected_hands):
            for landmark in results.hand_landmarks[hand_idx]:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 122, 255) if hand_idx==0 else (255, 122, 0), -1)
    
    if current_detected_hands != target_hands:
        target_hands = current_detected_hands
        stable_timer = 0
    else:
        stable_timer += 1

    if stable_timer > 30: 
        active_mode_hands = target_hands

    fist_detected = False
    if current_detected_hands > 0:
        for landmarks in results.hand_landmarks:
            if is_fist(landmarks):
                fist_detected = True
                break

    if fist_detected:
        status_text = "PAUSED (Rock Gesture Detected)"
        status_color = (0, 0, 255)
        prev_hand_x = None

    elif active_mode_hands == 1 and current_detected_hands >= 1:
        hand_0 = results.hand_landmarks[0]
        hand_x, hand_y = hand_0[9].x, hand_0[9].y
        
        if prev_hand_x is not None:
            azim_angle += (hand_x - prev_hand_x) * -400.0  
            elev_angle += (hand_y - prev_hand_y) * 400.0
            status_text, status_color = "MODE: Rotation", (0, 255, 255)
        
        prev_hand_x, prev_hand_y = hand_x, hand_y
        
    elif active_mode_hands == 2 and current_detected_hands >= 2:
        prev_hand_x = None 
        hand_0 = results.hand_landmarks[0]
        hand_1 = results.hand_landmarks[1]
        
        dist_pinch_0 = np.sqrt((hand_0[4].x - hand_0[8].x)**2 + (hand_0[4].y - hand_0[8].y)**2)
        dist_pinch_1 = np.sqrt((hand_1[4].x - hand_1[8].x)**2 + (hand_1[4].y - hand_1[8].y)**2)
        
        if dist_pinch_0 < 0.08 or dist_pinch_1 < 0.08: 
            deform_mode = "DONUT_HOLE"
            status_text, status_color = "MODE: Deep Sculpting", (255, 0, 255)
            
            if dist_pinch_0 < 0.08:
                thumb_x, thumb_y = int(hand_0[4].x * w), int(hand_0[4].y * h)
                index_x, index_y = int(hand_0[8].x * w), int(hand_0[8].y * h)
            else:
                thumb_x, thumb_y = int(hand_1[4].x * w), int(hand_1[4].y * h)
                index_x, index_y = int(hand_1[8].x * w), int(hand_1[8].y * h)
                
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 5)
            
        else:
            dist_x = np.abs(hand_0[0].x - hand_1[0].x)
            dist_y = np.abs(hand_0[0].y - hand_1[0].y)
            
            # 微調乘數，讓手不用拉到最開也能變大
            target_scale_xy = np.clip(dist_x * 3.5, 0.4, 3.0)
            target_scale_z = np.clip(dist_y * 3.5, 0.4, 3.0)
            
            # 阻尼系統：改成 0.05，大幅減緩跟隨速度
            global_scale_xy += (target_scale_xy - global_scale_xy) * 0.05
            global_scale_z += (target_scale_z - global_scale_z) * 0.05
            
            status_text, status_color = "MODE: Smooth Stretch", (0, 255, 122)
            center_x1, center_y1 = int(hand_0[0].x * w), int(hand_0[0].y * h)
            center_x2, center_y2 = int(hand_1[0].x * w), int(hand_1[0].y * h)
            cv2.line(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 255, 122), 2)
    else:
        status_text = "Stabilizing..."
        prev_hand_x = None

    if deform_mode == "DONUT_HOLE":
        for i in range(X_clay.shape[0]):
            for j in range(X_clay.shape[1]):
                r_xy = np.sqrt(X_clay[i, j]**2 + Y_clay[i, j]**2)
                
                if r_xy < 1e-3: 
                    X_clay[i, j] = np.cos(u[i]) * 1e-3
                    Y_clay[i, j] = np.sin(u[i]) * 1e-3
                    r_xy = 1e-3
                
                weight_z = np.exp(-(r_xy**2) / (2 * 1.5**2))
                
                Z_clay[i, j] *= (1.0 - 0.08 * weight_z) 
                
                push_force = np.exp(-(r_xy**2) / (2 * 1.5**2)) * 0.03
                X_clay[i, j] += (X_clay[i, j] / r_xy) * push_force
                Y_clay[i, j] += (Y_clay[i, j] / r_xy) * push_force

    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, "Fist (Rock) to pause / Pinch to dig hole", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 120), 1)
    
    cv2.putText(frame, "Press 'R' to Reset Clay", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    X_display = X_clay * global_scale_xy
    Y_display = Y_clay * global_scale_xy
    Z_display = Z_clay * global_scale_z
    
    clay_canvas = np.zeros((h, h, 3), dtype=np.uint8)
    draw_3d_wireframe(clay_canvas, X_display, Y_display, Z_display, elev_angle, azim_angle)
    
    combined_view = np.hstack((frame, clay_canvas))
    cv2.imshow('Ultimate Clay Master', combined_view)
    
    # ----------------------------------------------------
    # 修復的按鍵區塊：不再閃退
    # ----------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q'), 27]:
        break
    elif key in [ord('r'), ord('R')]: 
        # 徹底清空模型網格
        X_clay, Y_clay, Z_clay = X_orig.copy(), Y_orig.copy(), Z_orig.copy()
        # 縮放與角度
        global_scale_z, global_scale_xy = 1.0, 1.0
        azim_angle = 0.0      
        elev_angle = 20.0     
        # 清空所有狀態機的殘留判定
        target_hands = 0
        stable_timer = 0
        active_mode_hands = 0
        prev_hand_x = None
        prev_hand_y = None

cap.release()
cv2.destroyAllWindows()