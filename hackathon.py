import cv2
import numpy as np
import pyttsx3
import threading
import time
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter
import math # Needed for atan2
import textwrap # For wrapping long tip text

# --- Load the MoveNet Model ---
print("Loading MoveNet model...")
try:
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading MoveNet model: {e}")
    print("Please check your internet connection and TensorFlow Hub setup.")
    exit()

# --- Keypoint Dictionary ---
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

# --- Text-to-Speech Setup ---
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: pyttsx3 initialization failed: {e}. Audio feedback will be disabled.")
    engine = None

# --- Feedback State ---
last_feedback_message = ""

# --- Session State ---
current_exercise = "squats"
exercise_active = False
session_stats = {
    "squats": {"reps": 0, "feedback": []}, "curls": {"reps": 0, "feedback": []},
    "jacks": {"reps": 0, "feedback": []}, "pushups": {"reps": 0, "feedback": []},
    "lunges": {"reps": 0, "feedback": []}, "press": {"reps": 0, "feedback": []},
    "raises": {"reps": 0, "feedback": []}, "goodmornings": {"reps": 0, "feedback": []},
    # "highknees": {"reps": 0, "feedback": []}, # Removed High Knees
    "crunches": {"reps": 0, "feedback": []},
}

# Exercise Stage Variables (Global Scope)
exercise_stage = "start" # Generic stage variable for all exercises

# --- Configuration ---
VISIBILITY_THRESHOLD = 0.2 # Minimum confidence score for a keypoint to be considered reliable

# --- Geometry and Landmark Functions ---

def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three points a, b, c (b is the vertex)."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    # Calculate vectors from the vertex
    ba = a - b
    bc = c - b
    # Calculate angle using atan2 for better stability
    angle_rad = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle_deg = np.degrees(angle_rad)
    # Normalize to range [0, 360) then get the smaller angle (<180)
    angle_final = angle_deg % 360
    if angle_final > 180:
        angle_final = 360 - angle_final
    return angle_final if not np.isnan(angle_final) else 0.0

def get_landmark_px(landmarks, keypoint_name, frame_shape):
    """Gets pixel coordinates [x, y] and confidence of a keypoint."""
    y_max, x_max, _ = frame_shape
    idx = KEYPOINT_DICT.get(keypoint_name)
    if idx is None or landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape != (17, 3) or idx >= 17:
        return [0, 0], 0.0
    # Landmarks from MoveNet are [y, x, conf]
    y, x, conf = landmarks[idx]
    return [int(x * x_max), int(y * y_max)], float(conf)

# --- RE-ADDED THIS FUNCTION ---
def get_landmark(landmarks, keypoint_name):
    """Gets the normalized (0.0-1.0) coordinates AND confidence of a landmark."""
    idx = KEYPOINT_DICT.get(keypoint_name)
    # Check if keypoint name is valid and landmarks array is usable
    if idx is None or landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape != (17, 3) or idx >= 17:
        #print(f"Warning: Invalid keypoint name '{keypoint_name}' or landmarks array.")
        return np.array([0.0, 0.0, 0.0]) # Default if invalid

    keypoint = landmarks[idx]
    # Check for valid keypoint data before returning
    if keypoint.shape == (3,) and len(keypoint) == 3:
        return keypoint # Returns [y, x, confidence]
    else:
        #print(f"Warning: Invalid landmark data format for {keypoint_name}. Shape: {keypoint.shape if isinstance(keypoint, np.ndarray) else 'Not ndarray'}")
        return np.array([0.0, 0.0, 0.0]) # Default if invalid
# --- END OF RE-ADDED FUNCTION ---

# --- Feedback ---

def say_feedback(text, exercise_key=None, force_say=False):
    """Handles TTS output and logging feedback, preventing repeats unless forced."""
    global last_feedback_message
    if not engine: return
    is_rep_count = text.isdigit()

    if is_rep_count or force_say or text != last_feedback_message:
        if exercise_key and text and not is_rep_count and exercise_key in session_stats:
            session_stats[exercise_key]["feedback"].append(text)
        if not is_rep_count:
            last_feedback_message = text # Only update last message for non-rep counts

        # Run TTS in a separate thread to avoid blocking
        def run_tts():
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception: pass # Ignore TTS errors
        # Start thread only if no other TTS thread is running
        if not any(t.name == 'tts_thread' and t.is_alive() for t in threading.enumerate()):
            tts_thread = threading.Thread(target=run_tts, name='tts_thread', daemon=True)
            tts_thread.start()

def reset_feedback_state():
    """Resets the last feedback message to allow tips again."""
    global last_feedback_message
    last_feedback_message = ""

# --- Drawing ---

def draw_keypoints_and_skeleton(frame, keypoints_with_scores, confidence_threshold):
    """Draws detected keypoints and skeleton lines."""
    y_max, x_max, _ = frame.shape
    try:
        # Keypoints are already in the correct format [y, x, score] from TF Hub
        # No need to squeeze if input is already correct shape
        if keypoints_with_scores is None or keypoints_with_scores.shape != (1, 1, 17, 3):
             return # Invalid input

        keypoints = np.squeeze(keypoints_with_scores) # Now shape (17, 3)
        if keypoints.shape != (17, 3): return # Check shape after squeeze

        # Draw Keypoints
        for i in range(keypoints.shape[0]):
            y, x, score = keypoints[i]
            if score > confidence_threshold:
                cv2.circle(frame, (int(x * x_max), int(y * y_max)), 4, (0, 255, 0), -1) # Green dots

        # Draw Skeleton Edges
        for p1_idx, p2_idx in KEYPOINT_EDGES:
             # Check indices are within bounds before accessing
            if 0 <= p1_idx < keypoints.shape[0] and 0 <= p2_idx < keypoints.shape[0]:
                y1, x1, score1 = keypoints[p1_idx]
                y2, x2, score2 = keypoints[p2_idx]
                if score1 > confidence_threshold and score2 > confidence_threshold:
                    cv2.line(frame, (int(x1 * x_max), int(y1 * y_max)),
                             (int(x2 * x_max), int(y2 * y_max)), (255, 0, 0), 2) # Blue lines
            # else: print(f"Warning: Invalid indices in KEYPOINT_EDGES: {p1_idx}, {p2_idx}") # Optional debug

    except Exception as e:
        print(f"Error in draw_keypoints_and_skeleton: {e}")


def draw_ui(frame, rep_count, stage, exercise_name):
    """Draws the UI elements onto the frame."""
    h, w, _ = frame.shape
    alpha = 0.6
    try:
        # Top Bar
        ui_overlay = frame.copy(); cv2.rectangle(ui_overlay, (0, 0), (w, 100), (20, 20, 20), -1)
        frame[:] = cv2.addWeighted(ui_overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, 'REPS', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(rep_count), (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        stage_str = str(stage).upper() if stage is not None else "N/A"
        cv2.putText(frame, stage_str, (w - 190, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Bottom Bar
        bar_h = 130; ui_overlay_bottom = frame.copy()
        cv2.rectangle(ui_overlay_bottom, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        frame[:] = cv2.addWeighted(ui_overlay_bottom, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, f"EXERCISE: {exercise_name.upper()}", (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        font_scale, line_h, start_y = 0.5, 20, h - 75
        # Removed High Knees from controls
        controls = ["(s) Squats | (c) Curls | (j) Jacks | (p) Pushups | (l) Lunges",
                    "(o) O.Press | (r) L.Raises | (g) G.Mornings",
                    "(n) Crunches | (q) Quit Report"] # Removed (h) High Knees
        for i, line in enumerate(controls):
            cv2.putText(frame, line, (20, start_y + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # No User Overlay
        if not exercise_active:
            overlay = frame.copy(); cv2.rectangle(overlay, (0, 0), (w, h), (0,0,0), -1)
            frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            text = "NO USER DETECTED"; (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    except Exception as e:
        print(f"Error drawing UI: {e}")


# --- NEW Exercise Processing Logic ---

def process_exercise(landmarks, frame_shape, exercise_key):
    """
    Main dispatcher for exercise processing. Uses simpler state logic.
    Returns (rep_count, stage)
    """
    global exercise_stage # Use the generic stage variable

    rep_count = session_stats[exercise_key]['reps']
    current_stage = exercise_stage # Start with the current global stage

    try:
        # Get coordinates and confidences, check visibility early
        coords = {}
        confs = {}
        required_joints = []

        # Define required joints (same as before, excluding highknees)
        if exercise_key == "squats": required_joints = ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle']
        elif exercise_key == "curls": required_joints = ['left_shoulder', 'left_elbow', 'left_wrist']
        elif exercise_key == "jacks": required_joints = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'left_ankle', 'right_ankle', 'left_hip', 'right_hip']
        elif exercise_key == "pushups": required_joints = ['left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee']
        elif exercise_key == "lunges": required_joints = ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle']
        elif exercise_key == "press": required_joints = ['left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']
        elif exercise_key == "raises": required_joints = ['left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hip']
        elif exercise_key == "goodmornings": required_joints = ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle']
        # Removed highknees case
        elif exercise_key == "crunches": required_joints = ['left_shoulder', 'left_hip', 'left_knee']

        all_visible = True
        for joint in required_joints:
            coords[joint], confs[joint] = get_landmark_px(landmarks, joint, frame_shape)
            if confs[joint] < VISIBILITY_THRESHOLD:
                all_visible = False
                break

        if not all_visible:
            return rep_count, "N/A" # Return N/A immediately if not visible

        # --- Exercise Specific Logic with Hysteresis ---
        # State: "start" -> "down" -> "up" -> "down" (counts rep on up->down or down->up depending on exercise)

        # Bicep Curl (Left Arm)
        if exercise_key == "curls":
            angle = calculate_angle(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
            UP_THRESH = 70    # Angle clearly below this is 'up'
            DOWN_THRESH = 140 # Angle clearly above this is 'down'

            if angle < UP_THRESH and exercise_stage != "up":
                current_stage = "up"
            elif angle > DOWN_THRESH and exercise_stage == "up": # Count rep on transition from UP to DOWN
                current_stage = "down"
                rep_count += 1
                session_stats[exercise_key]['reps'] = rep_count
                print(f"Curl Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
            # else: keep current_stage = exercise_stage (hysteresis)

            # Feedback
            if current_stage == 'up' and angle > UP_THRESH + 15: say_feedback("Curl higher", exercise_key)
            elif current_stage == 'down' and angle < DOWN_THRESH - 15: say_feedback("Extend arm", exercise_key)

        # Squat
        elif exercise_key == "squats":
            knee_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
            hip_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
            DOWN_KNEE_THRESH, DOWN_HIP_THRESH = 110, 100
            UP_KNEE_THRESH, UP_HIP_THRESH = 150, 160

            if knee_angle < DOWN_KNEE_THRESH and hip_angle < DOWN_HIP_THRESH and exercise_stage != "down":
                current_stage = "down"
            elif knee_angle > UP_KNEE_THRESH and hip_angle > UP_HIP_THRESH and exercise_stage == "down": # Count on DOWN to UP
                current_stage = "up"
                rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                print(f"Squat Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
            # else: keep current_stage = exercise_stage

            # Feedback
            if current_stage == 'down':
                if knee_angle > DOWN_KNEE_THRESH + 10 or hip_angle > DOWN_HIP_THRESH + 10: say_feedback("Go lower", exercise_key)
                elif hip_angle < 70: say_feedback("Keep chest up", exercise_key)

        # Lateral Raises (Shoulder Abduction Angle)
        elif exercise_key == "raises":
            l_sh_angle = calculate_angle(coords['left_hip'], coords['left_shoulder'], coords['left_elbow'])
            r_sh_angle = calculate_angle(coords['right_hip'], coords['right_shoulder'], coords['right_elbow'])
            l_el_angle = calculate_angle(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
            r_el_angle = calculate_angle(coords['right_shoulder'], coords['right_elbow'], coords['right_wrist'])
            UP_SHOULDER_THRESH, DOWN_SHOULDER_THRESH = 70, 30
            STRAIGHT_ELBOW_THRESH = 130

            arms_straight = l_el_angle > STRAIGHT_ELBOW_THRESH and r_el_angle > STRAIGHT_ELBOW_THRESH
            shoulders_up = l_sh_angle > UP_SHOULDER_THRESH and r_sh_angle > UP_SHOULDER_THRESH
            shoulders_down = l_sh_angle < DOWN_SHOULDER_THRESH and r_sh_angle < DOWN_SHOULDER_THRESH

            if shoulders_up and arms_straight and exercise_stage != "up":
                current_stage = "up"
            elif shoulders_down and exercise_stage == "up": # Count on UP to DOWN
                current_stage = "down"
                rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                print(f"Raise Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
            # else: keep current_stage = exercise_stage

            # Feedback
            if current_stage == 'up':
                if not arms_straight: say_feedback("Straighter arms", exercise_key)
                nose_coord, _ = get_landmark_px(landmarks, 'nose', frame_shape)
                # Check if wrist Y is above nose Y (adjust threshold as needed)
                if coords['left_wrist'][1] < nose_coord[1] - 30 or coords['right_wrist'][1] < nose_coord[1] - 30:
                     say_feedback("Don't raise too high", exercise_key)
                elif l_sh_angle < UP_SHOULDER_THRESH - 5 or r_sh_angle < UP_SHOULDER_THRESH - 5: # If arms dropped slightly
                     say_feedback("Lift arms higher", exercise_key)

        # Pushups
        elif exercise_key == "pushups":
            elbow_angle = calculate_angle(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
            body_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
            DOWN_ELBOW_THRESH, UP_ELBOW_THRESH = 100, 150
            STRAIGHT_BODY_THRESH = 150

            if elbow_angle < DOWN_ELBOW_THRESH and exercise_stage != "down":
                current_stage = "down"
            elif elbow_angle > UP_ELBOW_THRESH and exercise_stage == "down": # Count on DOWN to UP
                 current_stage = "up"
                 rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                 print(f"Pushup Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
            # else: keep current_stage = exercise_stage

            if body_angle < STRAIGHT_BODY_THRESH: say_feedback("Keep back straight", exercise_key)
            if current_stage == 'down' and elbow_angle > DOWN_ELBOW_THRESH + 10: say_feedback("Go lower", exercise_key)

        # Lunges
        elif exercise_key == "lunges":
            front_knee_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
            back_knee_angle = calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
            DOWN_FRONT_KNEE_THRESH, DOWN_BACK_KNEE_THRESH = 120, 130
            UP_FRONT_KNEE_THRESH, UP_BACK_KNEE_THRESH = 160, 150

            if front_knee_angle < DOWN_FRONT_KNEE_THRESH and back_knee_angle < DOWN_BACK_KNEE_THRESH and exercise_stage != "down":
                current_stage = "down"
            elif front_knee_angle > UP_FRONT_KNEE_THRESH and back_knee_angle > UP_BACK_KNEE_THRESH and exercise_stage == "down": # Count on DOWN to UP
                 current_stage = "up"
                 rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                 print(f"Lunge Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
            # else: keep current_stage = exercise_stage

            if current_stage == 'down':
                 if front_knee_angle > DOWN_FRONT_KNEE_THRESH + 10 or back_knee_angle > DOWN_BACK_KNEE_THRESH + 10: say_feedback("Lower body", exercise_key)
                 if coords['left_knee'][0] > coords['left_ankle'][0] + 30: say_feedback("Knee over ankle", exercise_key)

        # Overhead Press
        elif exercise_key == "press":
            l_el_angle = calculate_angle(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
            r_el_angle = calculate_angle(coords['right_shoulder'], coords['right_elbow'], coords['right_wrist'])
            UP_ELBOW_THRESH, DOWN_ELBOW_THRESH = 150, 100
            wrists_above_shoulders = coords['left_wrist'][1] < coords['left_shoulder'][1] - 10 and coords['right_wrist'][1] < coords['right_shoulder'][1] - 10
            wrists_at_shoulders = abs(coords['left_wrist'][1] - coords['left_shoulder'][1]) < 40 and abs(coords['right_wrist'][1] - coords['right_shoulder'][1]) < 40

            if wrists_above_shoulders and l_el_angle > UP_ELBOW_THRESH and r_el_angle > UP_ELBOW_THRESH and exercise_stage != "up":
                 current_stage = "up"
            elif wrists_at_shoulders and l_el_angle < DOWN_ELBOW_THRESH and r_el_angle < DOWN_ELBOW_THRESH and exercise_stage == "up": # Count on UP to DOWN
                 current_stage = "down"
                 rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                 print(f"Press Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
            # else: keep current_stage = exercise_stage

            if current_stage == 'up' and (l_el_angle < UP_ELBOW_THRESH - 5 or r_el_angle < UP_ELBOW_THRESH - 5): say_feedback("Extend arms", exercise_key)
            elif current_stage == 'down' and not wrists_at_shoulders: say_feedback("Lower to shoulders", exercise_key)

        # Good Mornings
        elif exercise_key == "goodmornings":
             hip_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
             knee_angle = calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
             DOWN_HIP_THRESH, UP_HIP_THRESH = 130, 160
             STRAIGHT_KNEE_THRESH = 150

             if hip_angle < DOWN_HIP_THRESH and exercise_stage != "down":
                  current_stage = "down"
             elif hip_angle > UP_HIP_THRESH and exercise_stage == "down": # Count on DOWN to UP
                  current_stage = "up"
                  rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                  print(f"GM Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
             # else: keep current_stage = exercise_stage

             if knee_angle < STRAIGHT_KNEE_THRESH: say_feedback("Keep legs straighter", exercise_key)
             if current_stage == 'down' and hip_angle > DOWN_HIP_THRESH + 10: say_feedback("Hinge lower", exercise_key)

        # Removed High Knees Logic Block

        # Crunches
        elif exercise_key == "crunches":
             hip_angle = calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
             UP_HIP_THRESH, DOWN_HIP_THRESH = 95, 110

             if hip_angle < UP_HIP_THRESH and exercise_stage != "up":
                  current_stage = "up"
             elif hip_angle > DOWN_HIP_THRESH and exercise_stage == "up": # Count on UP to DOWN
                  current_stage = "down"
                  rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                  print(f"Crunch Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
             # else: keep current_stage = exercise_stage

             if current_stage == 'up' and hip_angle > UP_HIP_THRESH + 5: say_feedback("Lift higher", exercise_key)

        # Jacks (Jumping Jacks)
        elif exercise_key == "jacks":
            shoulder_width = abs(coords['left_shoulder'][0] - coords['right_shoulder'][0])
            hip_width = abs(coords['left_hip'][0] - coords['right_hip'][0])
            base_width = shoulder_width if shoulder_width > 20 else hip_width
            if base_width < 10: base_width = 50
            leg_distance = abs(coords['left_ankle'][0] - coords['right_ankle'][0])
            arms_raised = coords['left_wrist'][1] < coords['left_shoulder'][1] + 40 and coords['right_wrist'][1] < coords['right_shoulder'][1] + 40
            LEGS_WIDE_THRESH, LEGS_NARROW_THRESH = base_width * 1.3, base_width * 1.6

            if arms_raised and leg_distance > LEGS_WIDE_THRESH and exercise_stage != "up":
                current_stage = "up"
            elif not arms_raised and leg_distance < LEGS_NARROW_THRESH and exercise_stage == "up": # Count on UP to DOWN
                current_stage = "down"
                rep_count += 1; session_stats[exercise_key]['reps'] = rep_count
                print(f"Jack Count: {rep_count}"); say_feedback(str(rep_count)); reset_feedback_state()
            # else: keep current_stage = exercise_stage

            if current_stage == 'up':
                if not arms_raised: say_feedback("Arms higher", exercise_key)
                if leg_distance < LEGS_WIDE_THRESH: say_feedback("Legs wider", exercise_key)
            elif current_stage == 'down':
                 if arms_raised and (coords['left_wrist'][1] < coords['left_shoulder'][1] - 50 or coords['right_wrist'][1] < coords['right_shoulder'][1] - 50): say_feedback("Arms down", exercise_key)
                 if leg_distance > LEGS_NARROW_THRESH: say_feedback("Feet together", exercise_key)

        # Default for unhandled exercises
        else:
            current_stage = "N/A"

        # Update global stage only if it changed or was N/A initially
        if current_stage != "N/A":
             exercise_stage = current_stage
        return rep_count, current_stage

    except Exception as e:
        print(f"Error in process_exercise for {exercise_key}: {e}")
        exercise_stage = "start" # Reset on error
        return rep_count, "Error"


# --- Report Generation ---

def generate_report():
    """Generates a final session report image as a table."""
    report_img = np.full((1080, 1920, 3), (255, 255, 255), dtype=np.uint8)
    y_pos = 100
    left_margin = 100
    img_width = 1920
    col_exercise_x = left_margin
    col_reps_x = 550 # Adjusted column start
    col_tip_x = 800  # Adjusted column start
    col_tip_width = img_width - col_tip_x - left_margin # Width for tip text wrapping
    row_height = 60 # Default row height
    header_font_scale = 1.2
    body_font_scale = 1.0
    tip_font_scale = 0.9
    header_color = (0, 0, 0)
    line_color = (200, 200, 200)
    tip_color = (0, 0, 180)
    perfect_color = (0, 180, 0)

    def put_text_multiline(text, x, y, size=1.0, color=(0, 0, 0), bold=False, max_width=None):
        """Draws text, wrapping if max_width is specified. Returns total height used."""
        thickness = 3 if bold else 2
        text_str = str(text) if text is not None else ""
        lines = []
        if max_width:
             # Estimate average char width (very rough)
             avg_char_width = int(size * 20) # Adjust multiplier
             wrap_width_chars = max(10, max_width // avg_char_width) if avg_char_width > 0 else 10
             wrapped = textwrap.wrap(text_str, width=wrap_width_chars)
             lines.extend(wrapped)
        else:
            lines.append(text_str)

        line_y = y
        total_height = 0
        line_spacing = int(size * 40) # Adjust multiplier as needed
        for line in lines:
             if line.strip():
                 cv2.putText(report_img, line, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
                 line_y += line_spacing
                 total_height += line_spacing
        return total_height if total_height > 0 else line_spacing # Return at least one line height


    def draw_line(y1, y2, x): # Vertical line
        cv2.line(report_img, (x, y1), (x, y2), line_color, 2)

    def draw_hline(y, x1=left_margin, x2=img_width - left_margin): # Horizontal line
        cv2.line(report_img, (x1, y), (x2, y), line_color, 2)

    # Report Title
    put_text_multiline("AI Physiotherapist - Session Report", left_margin, y_pos, 2.0, (0, 165, 255), bold=True)
    y_pos += 60
    draw_hline(y_pos)
    y_pos += row_height // 2 # Space before header

    # Table Header
    header_y = y_pos + int(header_font_scale * 30) # Vertical center alignment
    put_text_multiline("Exercise", col_exercise_x, header_y, header_font_scale, header_color, bold=True)
    put_text_multiline("Reps", col_reps_x, header_y, header_font_scale, header_color, bold=True)
    put_text_multiline("Top Tip", col_tip_x, header_y, header_font_scale, header_color, bold=True)
    y_pos += row_height
    draw_hline(y_pos)
    table_start_y = y_pos # Remember where table body starts for vertical lines

    # Table Body
    total_reps = 0
    max_rows_on_screen = 10 # Limit rows to prevent going off screen
    rows_drawn = 0

    for exercise, data in session_stats.items():
        if data["reps"] > 0 and y_pos < (1080 - 100): # Check if there's vertical space
            rows_drawn += 1
            total_reps += data["reps"]

            # Calculate the starting Y position for this row's content (top align)
            content_start_y = y_pos + 20

            # --- Draw Row Content ---
            ex_height = put_text_multiline(exercise.upper(), col_exercise_x, content_start_y, body_font_scale, header_color)
            rep_height = put_text_multiline(str(data['reps']), col_reps_x, content_start_y, body_font_scale, header_color)

            # Top Tip (with wrapping)
            tip_text = "N/A"
            color = header_color
            if data["feedback"]:
                try:
                    counts = Counter(data["feedback"])
                    if counts:
                        tip, count = counts.most_common(1)[0]
                        tip_text = f"'{tip}' ({count} times)"
                        color = tip_color
                    else:
                        tip_text = "No specific feedback."
                        color = (100, 100, 100)
                except IndexError:
                    tip_text = "Error retrieving tip."
                    color = (100, 100, 100)
            else:
                tip_text = "Perfect form!"
                color = perfect_color

            tip_block_height = put_text_multiline(tip_text, col_tip_x, content_start_y, tip_font_scale, color, max_width=col_tip_width)

            # --- Calculate Row Height and Draw Bottom Line ---
            # Determine max height needed for this row
            actual_row_height = max(row_height, ex_height, rep_height, tip_block_height) + 20 # Add padding
            y_pos += actual_row_height # Move y_pos to the bottom of the current row
            if y_pos < (1080 - 80): # Don't draw line too close to bottom
                 draw_hline(y_pos) # Draw the line at the bottom
            else:
                 # Indicate truncation if we ran out of space
                 if sum(1 for d in session_stats.values() if d["reps"] > 0) > rows_drawn:
                      put_text_multiline("...", left_margin, y_pos - 10, 1.0, (100,100,100))
                 break # Stop drawing rows


    table_end_y = y_pos # Remember where table body ends

    # Draw Vertical Lines for the table (adjust end Y position)
    draw_line(table_start_y, table_end_y, col_reps_x - 30) # Between Ex and Reps
    draw_line(table_start_y, table_end_y, col_tip_x - 30)  # Between Reps and Tip

    if rows_drawn == 0:
        put_text_multiline("No exercises were completed.", left_margin, y_pos + row_height, 1.2)
        y_pos += row_height * 2


    put_text_multiline("Press 'q' to close this report.", left_margin, 1080 - 60, 1.0, (100, 100, 100))
    return report_img


# --- Main Application Loop ---

def main():
    global current_exercise, exercise_active, exercise_stage

    # --- CAMERA SELECTION ---
    camera_index = 0 # Default back to 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Warning: Could not open camera at index {camera_index}. Trying index 1.")
        camera_index = 1
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
             print(f"Error: Could not open webcam at index 0 or 1.")
             return

    print(f"AI Physiotherapist starting using camera index {camera_index}...")

    window_name = 'AI Physiotherapist'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Reset stage when starting
    exercise_stage = "start"
    say_feedback(f"Welcome to your AI Gym. Starting with {current_exercise}. Check controls.", None, force_say=True)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: print("Error: Camera frame empty."); time.sleep(1); break
        frame = cv2.flip(frame, 1) # Mirror effect

        # --- Resize for Inference ---
        input_size = 192 # MoveNet Lightning input size
        h, w, _ = frame.shape
        if h == 0 or w == 0: continue
        # Maintain aspect ratio by padding
        if h > w: pad = (h - w) // 2; input_frame = cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT)
        elif w > h: pad = (w - h) // 2; input_frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT)
        else: input_frame = frame
        try:
             img_resized = cv2.resize(input_frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e: print(f"Resize error: {e}"); continue
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Model expects RGB

        # --- Model Inference ---
        input_tensor = tf.cast(tf.expand_dims(img_rgb, axis=0), dtype=tf.int32)
        keypoints_with_scores = None
        try:
            results = movenet(input_tensor)
            # Output is shape (1, 1, 17, 3) [batch, instance, keypoint, (y, x, score)]
            keypoints_with_scores = results['output_0'].numpy()
        except Exception as e: print(f"Inference error: {e}"); exercise_active = False; landmarks = None

        # --- Landmark Processing & Activity Check ---
        landmarks = None # Use this for processing functions
        if keypoints_with_scores is not None and keypoints_with_scores.shape == (1, 1, 17, 3):
            landmarks = np.squeeze(keypoints_with_scores) # Shape (17, 3)
            conf_scores = landmarks[:, 2]
            visible_scores = conf_scores[conf_scores > VISIBILITY_THRESHOLD]
            avg_confidence = np.mean(visible_scores) if len(visible_scores) > 0 else 0
            # Activity check: Need at least a few points with reasonable confidence
            exercise_active = len(visible_scores) >= 5 and avg_confidence > 0.25
        else:
            exercise_active = False

        # --- Get Current State ---
        rep_count = session_stats.get(current_exercise, {}).get('reps', 0)
        # Use the global exercise_stage, default to "start" if not set
        current_stage_display = exercise_stage if exercise_stage else "start"

        # --- Process Exercise Logic ---
        if landmarks is not None and exercise_active:
            # Call the unified processing function
            new_rep_count, new_stage = process_exercise(landmarks, frame.shape, current_exercise)
            # Update display variables if processing was successful
            if new_stage != "Error": # Check for processing error
                 rep_count = new_rep_count
                 current_stage_display = new_stage if new_stage != "N/A" else "N/A" # Show N/A if visibility failed
            else:
                 current_stage_display = "Error" # Show error on UI if processing failed
        elif not exercise_active:
             current_stage_display = "---" # Show inactive state

        # --- Draw Visuals ---
        if landmarks is not None and exercise_active:
            # Pass the original keypoints_with_scores for drawing
            draw_keypoints_and_skeleton(frame, keypoints_with_scores, VISIBILITY_THRESHOLD)

            # --- VISUAL DEBUGGING (Example for Lateral Raises - Shoulder Angle) ---
            if current_exercise == "raises":
                try:
                    ls_coord, ls_conf = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
                    le_coord, le_conf = get_landmark_px(landmarks, 'left_elbow', frame_shape)
                    lh_coord, lh_conf = get_landmark_px(landmarks, 'left_hip', frame_shape)
                    rs_coord, rs_conf = get_landmark_px(landmarks, 'right_shoulder', frame_shape)
                    re_coord, re_conf = get_landmark_px(landmarks, 'right_elbow', frame_shape)
                    rh_coord, rh_conf = get_landmark_px(landmarks, 'right_hip', frame_shape)

                    # Draw Left Shoulder Angle
                    if min(ls_conf, le_conf, lh_conf) > 0.1:
                        l_sh_angle = calculate_angle(lh_coord, ls_coord, le_coord)
                        cv2.putText(frame, f"LSh:{l_sh_angle:.0f}", (int(ls_coord[0]-50), int(ls_coord[1]-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Cyan
                    # Draw Right Shoulder Angle
                    if min(rs_conf, re_conf, rh_conf) > 0.1:
                        r_sh_angle = calculate_angle(rh_coord, rs_coord, re_coord)
                        cv2.putText(frame, f"RSh:{r_sh_angle:.0f}", (int(rs_coord[0]+10), int(rs_coord[1]-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Cyan
                except Exception: pass # Ignore drawing errors


        draw_ui(frame, rep_count, current_stage_display, current_exercise)
        cv2.imshow(window_name, frame)

        # --- Key Controls ---
        key = cv2.waitKey(10) & 0xFF
        def switch_exercise(ex_key, ex_speech):
            global current_exercise, exercise_stage, last_feedback_message
            if current_exercise != ex_key:
                print(f"Switching to {ex_key}")
                current_exercise = ex_key
                exercise_stage = "start" # Reset stage on switch
                reset_feedback_state()
                say_feedback(ex_speech, None, force_say=True)

        if key == ord('q'): say_feedback("Workout complete. Generating report.", None, force_say=True); time.sleep(2); break
        elif key == ord('s'): switch_exercise("squats", "Squats")
        elif key == ord('c'): switch_exercise("curls", "Bicep Curls. Left arm.")
        elif key == ord('j'): switch_exercise("jacks", "Jumping Jacks")
        elif key == ord('p'): switch_exercise("pushups", "Pushups. Side view recommended.")
        elif key == ord('l'): switch_exercise("lunges", "Lunges. Left leg forward.")
        elif key == ord('o'): switch_exercise("press", "Overhead Press")
        elif key == ord('r'): switch_exercise("raises", "Lateral Raises")
        elif key == ord('g'): switch_exercise("goodmornings", "Good Mornings. Side view recommended.")
        # Removed High Knees key bind
        elif key == ord('n'): switch_exercise("crunches", "Crunches. Side view recommended.")

        frame_count += 1

    # --- Cleanup and Report ---
    if cap.isOpened(): cap.release()
    cv2.destroyAllWindows()

    # --- Display Final Report ---
    print("\n--- SESSION REPORT ---")
    total_reps_all = sum(data["reps"] for data in session_stats.values())
    for ex, data in session_stats.items():
        if data["reps"] > 0:
            print(f"\n{ex.upper()}: Reps: {data['reps']}")
            if data["feedback"]:
                try: counts = Counter(data["feedback"]); tip, count = counts.most_common(1)[0]; print(f"  Top Tip: '{tip}' ({count} times)")
                except IndexError: print("  Top Tip: No specific feedback recorded.")
            else: print("  Top Tip: Perfect form!")
    if total_reps_all == 0: print("No exercises completed.")
    print("\nClose report window to exit.")

    try:
        report_image = generate_report()
        report_window = 'Session Report'; cv2.namedWindow(report_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(report_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(report_window, report_image)
        while True:
            if cv2.getWindowProperty(report_window, cv2.WND_PROP_VISIBLE) < 1: break
            if cv2.waitKey(100) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
    except Exception as e: print(f"Error displaying final report: {e}"); cv2.destroyAllWindows()

if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(f"\n--- An unexpected error occurred in main ---"); print(f"Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        # TTS Cleanup
        if 'engine' in globals() and engine is not None:
             try:
                 is_busy = getattr(engine, 'isBusy', lambda: False)()
                 if is_busy: engine.stop()
                 if getattr(engine, '_inLoop', False): engine.endLoop()
             except Exception: pass # Ignore cleanup errors
        cv2.destroyAllWindows(); print("\nApplication finished.")

