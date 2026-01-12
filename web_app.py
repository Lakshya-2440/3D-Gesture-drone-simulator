"""
Flask Web App for Gesture-Driven Drone Simulator
Access the simulator through your web browser at http://localhost:5000
"""

from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import os
import urllib.request
import ssl
from collections import deque
import threading

app = Flask(__name__)

# ------------------ CONFIG ------------------

CAM_W, CAM_H = 640, 480
SIM_W, SIM_H = 900, 700
TOTAL_W, TOTAL_H = CAM_W + SIM_W, max(CAM_H, SIM_H)

DRONE_SPEED = 0.4
WORLD_RADIUS = 8.0
GESTURE_STABLE_FRAMES = 3

# ------------------ MEDIAPIPE SETUP ------------------

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    min_hand_presence_confidence=0.4
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    print("Pose model downloaded!")

pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ------------------ DRONE STATE ------------------

drone_x = 0.0
drone_z = 0.0
drone_state = "LANDED"
current_gesture = "NONE"
stable_gesture = "NONE"
gesture_history = deque(maxlen=GESTURE_STABLE_FRAMES)

# Thread lock for shared state
state_lock = threading.Lock()

# ------------------ UTIL FUNCTIONS ------------------

def fingers_up(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    out = []
    for tip, pip in zip(tips, pips):
        out.append(landmarks[tip].y < landmarks[pip].y)
    return out


def thumb_up(landmarks):
    # Thumb is up if tip (4) is to the left of IP joint (3) for right hand
    # Using x-coordinate comparison for thumb
    return landmarks[4].x < landmarks[3].x


def classify_gesture(landmarks):
    index, middle, ring, pinky = fingers_up(landmarks)
    thumb = thumb_up(landmarks)
    up_count = sum([index, middle, ring, pinky])

    # Rock (fist) = LAND
    if up_count == 0:
        return "LAND"
    
    # Palm (all 4 fingers + thumb up) = HOVER
    if index and middle and ring and pinky and thumb:
        return "HOVER"
    
    # 4 fingers (all 4 fingers but no thumb) = LEFT
    if index and middle and ring and pinky and not thumb:
        return "MOVE_LEFT"
    
    # 1 finger (index only) = FORWARD
    if index and not middle and not ring and not pinky:
        return "MOVE_FORWARD"
    
    # 2 fingers (index + middle) = BACK
    if index and middle and not ring and not pinky:
        return "MOVE_BACKWARD"
    
    # 3 fingers (index + middle + ring) = RIGHT
    if index and middle and ring and not pinky:
        return "MOVE_RIGHT"
    
    return "NONE"


def update_stable_gesture(new_gesture):
    global stable_gesture
    gesture_history.append(new_gesture)
    if len(gesture_history) == GESTURE_STABLE_FRAMES:
        unique = set(gesture_history)
        if len(unique) == 1 and list(unique)[0] != "NONE":
            stable_gesture = list(unique)[0]


def update_drone():
    global drone_x, drone_z, drone_state

    if stable_gesture == "LAND":
        drone_state = "LANDED"
        drone_x *= 0.9
        drone_z *= 0.9
        return

    if stable_gesture == "HOVER":
        drone_state = "HOVERING"
        return

    drone_state = "FLYING"

    if stable_gesture == "MOVE_FORWARD":
        drone_z -= DRONE_SPEED
    elif stable_gesture == "MOVE_BACKWARD":
        drone_z += DRONE_SPEED
    elif stable_gesture == "MOVE_LEFT":
        drone_x -= DRONE_SPEED
    elif stable_gesture == "MOVE_RIGHT":
        drone_x += DRONE_SPEED

    r = math.sqrt(drone_x**2 + drone_z**2)
    if r > WORLD_RADIUS:
        scale = WORLD_RADIUS / r
        drone_x *= scale
        drone_z *= scale


def draw_landmarks_on_image(image, hand_landmarks_list, image_width, image_height):
    t = time.time()
    overlay = image.copy()
    glow_layer = np.zeros_like(image)
    
    for hand_landmarks in hand_landmarks_list:
        points = []
        for landmark in hand_landmarks:
            point = (int(landmark.x * image_width), int(landmark.y * image_height))
            points.append(point)
        
        palm_indices = [0, 5, 9, 13, 17]
        palm_x = int(np.mean([points[i][0] for i in palm_indices]))
        palm_y = int(np.mean([points[i][1] for i in palm_indices]))
        palm_center = (palm_x, palm_y)
        
        pulse = abs(math.sin(t * 4)) * 0.5 + 0.5
        pulse2 = abs(math.sin(t * 6 + 1)) * 0.5 + 0.5
        
        primary_color = (255, 200, 0)
        secondary_color = (255, 100, 0)
        accent_color = (255, 255, 100)
        glow_color = (255, 150, 0)
        
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = points[start_idx]
            end_point = points[end_idx]
            cv2.line(glow_layer, start_point, end_point, glow_color, 8)
            cv2.line(glow_layer, start_point, end_point, primary_color, 4)
            cv2.line(overlay, start_point, end_point, accent_color, 2)
        
        for i, point in enumerate(points):
            radius = int(12 + pulse * 4)
            cv2.circle(glow_layer, point, radius, glow_color, 2)
            cv2.circle(overlay, point, 8, primary_color, 2)
            cv2.circle(overlay, point, 4, accent_color, -1)
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(overlay, point, int(6 + pulse2 * 3), secondary_color, 2)
                cv2.circle(glow_layer, point, 18, glow_color, 1)
        
        for i in range(3):
            radius = int(40 + i * 20 + pulse * 8)
            thickness = 2 if i == 0 else 1
            alpha = 1.0 - (i * 0.3)
            color = tuple(int(c * alpha) for c in primary_color)
            cv2.circle(overlay, palm_center, radius, color, thickness)
        
        arc_radius = 70
        for i in range(6):
            angle_offset = (t * 2 + i * 60) % 360
            start_angle = int(angle_offset)
            end_angle = int(angle_offset + 30)
            cv2.ellipse(overlay, palm_center, (arc_radius, arc_radius), 
                       0, start_angle, end_angle, primary_color, 2)
        
        arc_radius2 = 50
        for i in range(4):
            angle_offset = (-t * 3 + i * 90) % 360
            start_angle = int(angle_offset)
            end_angle = int(angle_offset + 45)
            cv2.ellipse(overlay, palm_center, (arc_radius2, arc_radius2), 
                       0, start_angle, end_angle, secondary_color, 1)
        
        fingertips = [4, 8, 12, 16, 20]
        for tip_idx in fingertips:
            tip = points[tip_idx]
            dist = np.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
            if dist > 0:
                num_dashes = int(dist / 10)
                for j in range(num_dashes):
                    if j % 2 == 0:
                        t1 = j / num_dashes
                        t2 = (j + 1) / num_dashes
                        p1 = (int(palm_center[0] + t1 * (tip[0] - palm_center[0])),
                              int(palm_center[1] + t1 * (tip[1] - palm_center[1])))
                        p2 = (int(palm_center[0] + t2 * (tip[0] - palm_center[0])),
                              int(palm_center[1] + t2 * (tip[1] - palm_center[1])))
                        cv2.line(overlay, p1, p2, (200, 100, 0), 1)
        
        cv2.putText(overlay, f"HAND LOCK", 
                   (palm_x - 40, palm_y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, accent_color, 1)
        cv2.rectangle(overlay, (palm_x - 45, palm_y - 90), (palm_x + 45, palm_y - 70), primary_color, 1)
        cv2.putText(overlay, f"X:{palm_x:03d} Y:{palm_y:03d}", 
                   (palm_x - 35, palm_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, primary_color, 1)
    
    glow_layer = cv2.GaussianBlur(glow_layer, (21, 21), 0)
    result = cv2.addWeighted(image, 0.7, glow_layer, 0.5, 0)
    result = cv2.addWeighted(result, 1.0, overlay, 0.8, 0)
    
    return result


def draw_sim(sim):
    sim[:] = 0
    h, w, _ = sim.shape
    cx, ground_y = w // 2, int(h * 0.75)
    horizon_y = int(h * 0.25)
    t = time.time()

    # Draw perspective grid with cyan accent
    num_lines = 20
    for i in range(-num_lines, num_lines + 1):
        x_bottom = cx + i * 30
        x_top = cx + int(i * 10)
        cv2.line(sim, (x_bottom, ground_y), (x_top, horizon_y), (30, 50, 50), 1)

    for j in range(1, 10):
        tj = j / 10.0
        y = int(horizon_y + tj * (ground_y - horizon_y))
        x_span = int((1 - tj) * (w * 0.45))
        cv2.line(sim, (cx - x_span, y), (cx + x_span, y), (25, 45, 45), 1)

    span_x = w * 0.4
    sx = int(cx + (drone_x / WORLD_RADIUS) * span_x)
    z_norm = (drone_z + WORLD_RADIUS) / (2 * WORLD_RADIUS)
    sy = int(horizon_y + z_norm * (ground_y - horizon_y))

    # Create glow layer for drone
    glow = np.zeros_like(sim)
    
    # Pulsing effect
    pulse = abs(math.sin(t * 4)) * 0.5 + 0.5
    
    # Tech drone colors
    body_color = (80, 80, 80)  # Dark gray body
    accent_color = (255, 200, 0)  # Cyan accent
    led_color = (255, 150, 0)  # Bright cyan LED
    highlight = (255, 255, 200)  # White-cyan highlight
    
    # Drone body - sleek hexagonal shape
    body_w, body_h = 50, 22
    
    # Main body with gradient effect
    pts = np.array([
        [sx - body_w//2, sy],
        [sx - body_w//3, sy - body_h//2],
        [sx + body_w//3, sy - body_h//2],
        [sx + body_w//2, sy],
        [sx + body_w//3, sy + body_h//2],
        [sx - body_w//3, sy + body_h//2],
    ], np.int32)
    cv2.fillPoly(sim, [pts], body_color)
    cv2.polylines(sim, [pts], True, accent_color, 2)
    
    # Central LED core (pulsing)
    core_radius = int(8 + pulse * 3)
    cv2.circle(glow, (sx, sy), core_radius + 8, led_color, -1)
    cv2.circle(sim, (sx, sy), core_radius, highlight, -1)
    cv2.circle(sim, (sx, sy), core_radius - 3, accent_color, -1)
    
    # Arms with LED strips
    arm_len = 35
    arm_positions = [
        (sx - body_w//2 - 5, sy - 8, -1, -0.5),   # Front-left
        (sx + body_w//2 + 5, sy - 8, 1, -0.5),    # Front-right
        (sx - body_w//2 - 5, sy + 8, -1, 0.5),    # Back-left
        (sx + body_w//2 + 5, sy + 8, 1, 0.5),     # Back-right
    ]
    
    rotors = []
    for i, (ax, ay, dx, dy) in enumerate(arm_positions):
        # Calculate arm end point
        end_x = int(ax + dx * arm_len)
        end_y = int(ay + dy * arm_len)
        
        # Draw arm with gradient/LED effect
        cv2.line(sim, (int(ax), int(ay)), (end_x, end_y), body_color, 4)
        cv2.line(sim, (int(ax), int(ay)), (end_x, end_y), accent_color, 2)
        
        # LED dots along arm
        for j in range(3):
            led_t = (j + 1) / 4
            led_x = int(ax + dx * arm_len * led_t)
            led_y = int(ay + dy * arm_len * led_t)
            led_pulse = abs(math.sin(t * 6 + i + j)) * 0.5 + 0.5
            led_size = int(2 + led_pulse)
            cv2.circle(sim, (led_x, led_y), led_size, highlight, -1)
        
        rotors.append((end_x, end_y, i))
    
    # Draw rotors with spinning effect
    rotor_r = 12
    for rx, ry, idx in rotors:
        # Rotor glow
        cv2.circle(glow, (rx, ry), rotor_r + 5, led_color, -1)
        
        # Rotor ring
        cv2.circle(sim, (rx, ry), rotor_r, accent_color, 2)
        cv2.circle(sim, (rx, ry), rotor_r - 3, body_color, -1)
        
        # Spinning blades (if not landed)
        if drone_state != "LANDED":
            blade_angle = (t * 20 + idx * 90) % 360
            for b in range(3):
                angle_rad = math.radians(blade_angle + b * 120)
                bx = int(rx + (rotor_r - 2) * math.cos(angle_rad))
                by = int(ry + (rotor_r - 2) * math.sin(angle_rad))
                cv2.line(sim, (rx, ry), (bx, by), highlight, 1)
        
        # Center hub
        cv2.circle(sim, (rx, ry), 3, highlight, -1)
    
    # Apply glow
    glow = cv2.GaussianBlur(glow, (21, 21), 0)
    sim[:] = cv2.addWeighted(sim, 1.0, glow, 0.4, 0)
    
    # Status indicator lights on body
    for i in range(3):
        light_x = sx - 10 + i * 10
        light_pulse = abs(math.sin(t * 3 + i * 0.5)) * 0.8 + 0.2
        if drone_state == "LANDED":
            light_color = (0, 0, int(255 * light_pulse))  # Red when landed
        elif drone_state == "HOVERING":
            light_color = (0, int(255 * light_pulse), int(255 * light_pulse))  # Yellow when hovering
        else:
            light_color = (0, int(255 * light_pulse), 0)  # Green when flying
        cv2.circle(sim, (light_x, sy - body_h//2 - 3), 2, light_color, -1)
    
    # HUD text
    cv2.putText(sim, "3D Gesture Drone Simulator",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(sim, f"Gesture: {stable_gesture}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(sim, f"Drone: {drone_state}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(sim, f"Pos: x={drone_x:.2f}  z={drone_z:.2f}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(sim, "Controls:",
                (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(sim, "Rock = LAND, Palm = HOVER",
                (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(sim, "1 finger = FWD, 2 = BACK, 3 = RIGHT, 4 = LEFT",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


def draw_hud_frame(frame, current_gesture, stable_gesture):
    h, w = frame.shape[:2]
    t = time.time()
    
    cyan = (255, 200, 0)
    bright_cyan = (255, 255, 100)
    dark_cyan = (180, 100, 0)
    orange = (0, 140, 255)
    
    pulse = abs(math.sin(t * 3)) * 0.4 + 0.6
    
    bracket_len = 60
    bracket_thickness = 2
    margin = 15
    
    cv2.line(frame, (margin, margin), (margin + bracket_len, margin), cyan, bracket_thickness)
    cv2.line(frame, (margin, margin), (margin, margin + bracket_len), cyan, bracket_thickness)
    cv2.line(frame, (w - margin, margin), (w - margin - bracket_len, margin), cyan, bracket_thickness)
    cv2.line(frame, (w - margin, margin), (w - margin, margin + bracket_len), cyan, bracket_thickness)
    cv2.line(frame, (margin, h - margin), (margin + bracket_len, h - margin), cyan, bracket_thickness)
    cv2.line(frame, (margin, h - margin), (margin, h - margin - bracket_len), cyan, bracket_thickness)
    cv2.line(frame, (w - margin, h - margin), (w - margin - bracket_len, h - margin), cyan, bracket_thickness)
    cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - bracket_len), cyan, bracket_thickness)
    
    scan_y = int((t * 100) % h)
    cv2.line(frame, (margin + 5, scan_y), (w - margin - 5, scan_y), dark_cyan, 1)
    
    cv2.rectangle(frame, (margin + bracket_len + 10, margin - 5), 
                  (w - margin - bracket_len - 10, margin + 25), (0, 0, 0), -1)
    cv2.rectangle(frame, (margin + bracket_len + 10, margin - 5), 
                  (w - margin - bracket_len - 10, margin + 25), cyan, 1)
    cv2.putText(frame, "GESTURE INTERFACE v2.0", 
               (margin + bracket_len + 20, margin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bright_cyan, 1)
    
    panel_y = h // 2 - 60
    cv2.rectangle(frame, (margin, panel_y), (margin + 120, panel_y + 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (margin, panel_y), (margin + 120, panel_y + 120), cyan, 1)
    
    cv2.putText(frame, "STATUS", (margin + 25, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cyan, 1)
    cv2.line(frame, (margin + 5, panel_y + 28), (margin + 115, panel_y + 28), dark_cyan, 1)
    
    cv2.putText(frame, f"DETECT:", (margin + 8, panel_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dark_cyan, 1)
    gesture_color = orange if current_gesture != "NONE" else dark_cyan
    cv2.putText(frame, current_gesture[:8], (margin + 8, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, gesture_color, 1)
    
    cv2.putText(frame, f"LOCKED:", (margin + 8, panel_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dark_cyan, 1)
    stable_color = bright_cyan if stable_gesture != "NONE" else dark_cyan
    cv2.putText(frame, stable_gesture[:8], (margin + 8, panel_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, stable_color, 1)
    
    indicator_x = w - margin - 30
    indicator_y = h // 2
    indicator_r = int(15 + pulse * 5)
    cv2.circle(frame, (indicator_x, indicator_y), indicator_r, cyan, 2)
    cv2.circle(frame, (indicator_x, indicator_y), 8, bright_cyan, -1)
    
    cv2.rectangle(frame, (margin + bracket_len + 10, h - margin - 30), 
                  (w - margin - bracket_len - 10, h - margin + 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (margin + bracket_len + 10, h - margin - 30), 
                  (w - margin - bracket_len - 10, h - margin + 5), cyan, 1)
    
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"TIME: {timestamp}  |  FPS: 30  |  LINK: ACTIVE  |  SYSTEM: ONLINE", 
               (margin + bracket_len + 20, h - margin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, cyan, 1)
    
    for i in range(5):
        bar_x = w - margin - 25
        bar_y = panel_y + 10 + i * 18
        bar_w = int(20 * (0.3 + 0.7 * abs(math.sin(t * 2 + i))))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), cyan, -1)
    
    return frame


def draw_arc_reactor(frame, pose_landmarks, image_width, image_height):
    if not pose_landmarks:
        return frame
    
    t = time.time()
    
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    
    chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * image_width)
    chest_y = int((left_shoulder.y + right_shoulder.y) / 2 * image_height + 40)
    
    shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image_width
    reactor_radius = int(shoulder_width * 0.12)
    reactor_radius = max(18, min(reactor_radius, 40))
    
    glow = np.zeros_like(frame)
    
    white_core = (255, 255, 255)
    cyan_glow = (255, 220, 150)
    light_cyan = (255, 200, 120)
    metallic_gray = (140, 140, 140)
    dark_metal = (80, 80, 85)
    segment_glow = (255, 180, 100)
    
    pulse = abs(math.sin(t * 3)) * 0.15 + 0.85
    
    outer_r = reactor_radius + 12
    cv2.circle(frame, (chest_x, chest_y), outer_r + 3, dark_metal, 4)
    cv2.circle(frame, (chest_x, chest_y), outer_r, metallic_gray, 3)
    
    num_segments = 10
    segment_inner_r = reactor_radius + 2
    segment_outer_r = reactor_radius + 10
    segment_angle = 360 / num_segments
    gap_angle = 8
    
    for i in range(num_segments):
        start_angle = i * segment_angle + gap_angle / 2
        end_angle = (i + 1) * segment_angle - gap_angle / 2
        
        cv2.ellipse(glow, (chest_x, chest_y), (segment_outer_r, segment_outer_r),
                   0, start_angle, end_angle, cyan_glow, 6)
        cv2.ellipse(frame, (chest_x, chest_y), (segment_outer_r, segment_outer_r),
                   0, start_angle, end_angle, light_cyan, 3)
        cv2.ellipse(frame, (chest_x, chest_y), (segment_inner_r, segment_inner_r),
                   0, start_angle, end_angle, light_cyan, 2)
        
        mid_angle = (start_angle + end_angle) / 2
        for angle in [start_angle + 2, mid_angle, end_angle - 2]:
            rad = math.radians(angle)
            x1 = int(chest_x + segment_inner_r * math.cos(rad))
            y1 = int(chest_y + segment_inner_r * math.sin(rad))
            x2 = int(chest_x + segment_outer_r * math.cos(rad))
            y2 = int(chest_y + segment_outer_r * math.sin(rad))
            cv2.line(glow, (x1, y1), (x2, y2), segment_glow, 4)
            cv2.line(frame, (x1, y1), (x2, y2), light_cyan, 1)
    
    inner_metal_r = reactor_radius - 2
    cv2.circle(frame, (chest_x, chest_y), inner_metal_r + 2, metallic_gray, 3)
    cv2.circle(frame, (chest_x, chest_y), inner_metal_r, dark_metal, 2)
    
    mid_ring_r = int(reactor_radius * 0.65)
    cv2.circle(glow, (chest_x, chest_y), mid_ring_r + 2, cyan_glow, 3)
    cv2.circle(frame, (chest_x, chest_y), mid_ring_r, light_cyan, 2)
    cv2.circle(frame, (chest_x, chest_y), mid_ring_r - 3, metallic_gray, 2)
    
    core_r = int(reactor_radius * 0.4 * pulse)
    cv2.circle(glow, (chest_x, chest_y), core_r + 15, cyan_glow, -1)
    cv2.circle(glow, (chest_x, chest_y), core_r + 8, light_cyan, -1)
    cv2.circle(glow, (chest_x, chest_y), core_r + 3, white_core, -1)
    cv2.circle(frame, (chest_x, chest_y), core_r, white_core, -1)
    cv2.circle(frame, (chest_x, chest_y), int(core_r * 0.6), (240, 245, 250), 1)
    
    glow = cv2.GaussianBlur(glow, (25, 25), 0)
    frame = cv2.addWeighted(frame, 1.0, glow, 0.6, 0)
    
    return frame


# ------------------ VIDEO STREAMING ------------------

def generate_frames():
    global current_gesture
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_W, CAM_H))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = hand_landmarker.detect(mp_image)
        pose_results = pose_landmarker.detect(mp_image)

        current_gesture = "NONE"
        
        if pose_results.pose_landmarks:
            frame = draw_arc_reactor(frame, pose_results.pose_landmarks[0], CAM_W, CAM_H)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            frame = draw_landmarks_on_image(frame, results.hand_landmarks, CAM_W, CAM_H)
            current_gesture = classify_gesture(hand_landmarks)
            update_stable_gesture(current_gesture)

        update_drone()

        frame = draw_hud_frame(frame, current_gesture, stable_gesture)

        sim = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)
        draw_sim(sim)

        combined = np.zeros((TOTAL_H, TOTAL_W, 3), dtype=np.uint8)
        combined[:SIM_H, :SIM_W] = sim
        cam_y_offset = (TOTAL_H - CAM_H) // 2
        combined[cam_y_offset:cam_y_offset + CAM_H, SIM_W:SIM_W + CAM_W] = frame

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# ------------------ HTML TEMPLATE ------------------

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Gesture Drone Simulator</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: #000;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Rajdhani', sans-serif;
            color: #00d4ff;
            overflow: hidden;
        }
        
        .container {
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
        }
        
        h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #00ff88, #00d4ff);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 3s linear infinite;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 4px;
        }
        
        @keyframes shimmer {
            0% { background-position: 0% center; }
            100% { background-position: 200% center; }
        }
        
        .subtitle {
            font-size: 1rem;
            color: #888;
            margin-bottom: 25px;
            letter-spacing: 2px;
        }
        
        .video-container {
            position: relative;
            border: 2px solid #00d4ff;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 
                0 0 20px rgba(0, 212, 255, 0.3),
                0 0 40px rgba(0, 212, 255, 0.1),
                inset 0 0 20px rgba(0, 212, 255, 0.05);
            background: #000;
            max-width: 98vw;
            max-height: 96vh;
        }
        
        .video-container::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #00d4ff, #00ff88, #00d4ff, #00ff88);
            background-size: 400% 400%;
            z-index: -1;
            border-radius: 12px;
            animation: gradient-border 4s ease infinite;
            opacity: 0.5;
        }
        
        @keyframes gradient-border {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .video-feed {
            display: block;
            width:100vw;
            height: auto;
            max-width: 98vw;
            max-height: 96vh;
            object-fit: contain;
        }
        
        .controls-info {
            margin-top: 25px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            max-width: 800px;
        }
        
        .control-item {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .control-item:hover {
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
            transform: translateY(-2px);
        }
        
        .gesture-icon {
            font-size: 2rem;
            margin-bottom: 8px;
        }
        
        .gesture-name {
            font-weight: 600;
            color: #00ff88;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .gesture-action {
            font-size: 0.9rem;
            color: #aaa;
            margin-top: 5px;
        }
        
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.8);
            border-top: 1px solid #00d4ff;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        @media (max-width: 768px) {
            h1 {
                font-size: 1.5rem;
            }
            
            .controls-info {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-container">
            <img class="video-feed" src="{{ url_for('video_feed') }}" alt="Live Video Feed">
        </div>
    </div>
</body>
</html>
'''


# ------------------ ROUTES ------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------ MAIN ------------------

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  J.A.R.V.I.S. GESTURE DRONE SIMULATOR - WEB INTERFACE")
    print("="*60)
    print("\n  Open your browser and go to:")
    print("  ➜  http://localhost:5001")
    print("  ➜  http://127.0.0.1:5001")
    print("\n  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
