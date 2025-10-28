import cv2
import numpy as np
import pyttsx3
import threading
import time
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter

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

# --- Real-time feedback state ---
last_feedback_message = ""

# --- Session State and Report Data ---
current_exercise = "squats"  # Default exercise
exercise_active = False # To check if user is in frame
session_stats = {
    "squats": {"reps": 0, "feedback": []},
    "curls": {"reps": 0, "feedback": []},
    "jacks": {"reps": 0, "feedback": []},
    "pushups": {"reps": 0, "feedback": []},
    "lunges": {"reps": 0, "feedback": []},
    "press": {"reps": 0, "feedback": []},
    "raises": {"reps": 0, "feedback": []},
    "goodmornings": {"reps": 0, "feedback": []},
    "highknees": {"reps": 0, "feedback": []},
    "crunches": {"reps": 0, "feedback": []},
}

# Exercise-specific states
squat_stage = "up"
curl_stage = "down"
jack_stage = "down"
pushup_stage = "up"
lunge_stage = "up"
press_stage = "down"
raise_stage = "down"
good_morning_stage = "up"
high_knee_stage = "down"
crunch_stage = "down"

# --- Geometry and Landmark Functions ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def get_landmark_px(landmarks, keypoint_name, frame_shape):
    """Gets the pixel coordinates of a specific landmark."""
    y, x, _ = frame_shape
    keypoint = landmarks[KEYPOINT_DICT[keypoint_name]]
    # Check for valid keypoint data before calculation
    if keypoint.shape == (3,) and len(keypoint) == 3:
         return [keypoint[1] * x, keypoint[0] * y]
    else:
        # Return a default or raise an error if keypoint data is invalid
        #print(f"Warning: Invalid landmark data for {keypoint_name}. Landmark: {keypoint}")
        return [0,0] # Default to origin if invalid data

def get_landmark(landmarks, keypoint_name):
    """Gets the normalized (0.0-1.0) coordinates of a landmark."""
    # Check for valid keypoint data before returning
    keypoint = landmarks[KEYPOINT_DICT[keypoint_name]]
    if keypoint.shape == (3,) and len(keypoint) == 3:
        return keypoint
    else:
        #print(f"Warning: Invalid landmark data for {keypoint_name}. Landmark: {keypoint}")
        return np.array([0.0, 0.0, 0.0]) # Default if invalid

# --- Feedback Function ---

def say_feedback(text, exercise_name=None, force_say=False): # Added force_say
    """Says feedback and logs it, prevents spamming the same tip unless forced."""
    global last_feedback_message
    if not engine: return
    is_rep_count = text.isdigit()

    # Allow rep counts or new messages, or if forced (for suggestions)
    if is_rep_count or (text != last_feedback_message) or force_say:
        if exercise_name and text and not is_rep_count:
            session_stats[exercise_name]["feedback"].append(text)
        if not is_rep_count: # Don't overwrite last message with rep counts
            last_feedback_message = text
        def run():
            try: engine.say(text); engine.runAndWait()
            except Exception: pass
        threading.Thread(target=run).start()

def reset_feedback_state():
    """Resets the last feedback message."""
    global last_feedback_message
    last_feedback_message = ""

# --- NEW: Initial Pose Analysis ---

def analyze_initial_pose(landmarks):
    """Analyzes initial landmarks to guess if standing or sitting/lying."""
    try:
        # Get normalized Y coordinates (0.0 = top, 1.0 = bottom)
        left_hip_y = get_landmark(landmarks, 'left_hip')[0]
        left_knee_y = get_landmark(landmarks, 'left_knee')[0]
        left_ankle_y = get_landmark(landmarks, 'left_ankle')[0]
        left_shoulder_y = get_landmark(landmarks, 'left_shoulder')[0]

        # Basic Standing Check: Hip above knee, knee above ankle, significant distance
        is_standing = (left_hip_y < left_knee_y - 0.1 and
                       left_knee_y < left_ankle_y - 0.1)

        # Basic Sitting/Lying Check: Hip is near shoulder height or lower than knee
        is_sitting_lying = (abs(left_hip_y - left_shoulder_y) < 0.15 or
                            left_hip_y > left_knee_y)

        if is_standing:
            return "standing"
        elif is_sitting_lying:
            return "sitting_lying"
        else:
            return "unknown" # Could be transitioning or bad detection
    except Exception as e:
        print(f"Error analyzing initial pose: {e}")
        return "unknown"

def suggest_exercises(initial_pose_state):
    """Suggests exercises based on the initial pose."""
    suggestions = []
    if initial_pose_state == "standing":
        suggestions = ["Squats", "Lunges", "Jumping Jacks", "Overhead Press", "High Knees"]
        suggestion_text = "You seem to be standing. Try Squats, Lunges, or Jumping Jacks."
    elif initial_pose_state == "sitting_lying":
        suggestions = ["Pushups", "Crunches", "Bicep Curls"]
        suggestion_text = "You seem to be sitting or lying down. How about Pushups, Crunches, or Bicep Curls?"
    else:
        suggestion_text = "Could not determine starting pose clearly. Starting with Squats."

    print(f"Initial pose detected: {initial_pose_state}")
    print(f"Suggested exercises: {', '.join(suggestions) if suggestions else 'None'}")
    say_feedback(suggestion_text, force_say=True) # Force TTS to say suggestion


# --- Drawing Functions ---

def draw_keypoints_and_skeleton(frame, keypoints, confidence_threshold=0.4):
    """Draws keypoints and skeleton on the frame."""
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    if shaped.ndim == 1 or shaped.shape[0] != 17: # Handle case where squeeze might over-reduce or landmarks invalid
        #print("Warning: Invalid keypoints shape for drawing.")
        return # Skip drawing if data is bad

    for kp in shaped:
        # Ensure kp is iterable and has 3 elements
        if hasattr(kp, '__iter__') and len(kp) == 3:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold: cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
        #else: print(f"Warning: Invalid keypoint format: {kp}")


    for edge in KEYPOINT_EDGES:
         # Ensure indices are within bounds
        if shaped.shape[0] > max(edge):
            p1_idx, p2_idx = edge
            # Ensure shape results are valid before accessing
            if (hasattr(shaped[p1_idx], '__iter__') and len(shaped[p1_idx]) == 3 and
                hasattr(shaped[p2_idx], '__iter__') and len(shaped[p2_idx]) == 3):
                y1, x1, c1 = shaped[p1_idx]; y2, x2, c2 = shaped[p2_idx]
                if c1 > confidence_threshold and c2 > confidence_threshold:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            #else: print(f"Warning: Invalid shape data for edge {edge}")
        #else: print(f"Warning: Keypoint index out of bounds for edge {edge}")


def draw_ui(frame, rep_count, stage, exercise_name): # Removed demo_image argument
    """Draws the main UI elements on the frame."""
    h, w, _ = frame.shape
    alpha = 0.6  # Transparency

    # --- Top Bar (Reps and Stage) ---
    ui_overlay = frame.copy()
    cv2.rectangle(ui_overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(ui_overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, 'REPS', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(rep_count), (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(frame, 'STAGE', (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, stage.upper(), (w - 190, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    # --- Top-Right Corner Demo Image ---
    # Removed demo image drawing logic

    # --- Bottom Bar (Controls) ---
    bottom_bar_height = 130
    ui_overlay_bottom = frame.copy()
    cv2.rectangle(ui_overlay_bottom, (0, h - bottom_bar_height), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(ui_overlay_bottom, alpha, frame, 1 - alpha, 0, frame)

    # Line 1: Current Exercise
    cv2.putText(frame, f"EXERCISE: {exercise_name.upper()}", (20, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA) # Orange color

    # Split controls into multiple lines
    font_scale = 0.5
    line_height = 20
    start_y = h - 75

    controls_line_1 = "(s) Squats | (c) Curls | (j) Jacks | (p) Pushups | (l) Lunges"
    controls_line_2 = "(o) Overhead Press | (r) Lateral Raises | (g) Good Mornings"
    controls_line_3 = "(h) High Knees | (n) Crunches | (q) Quit Report"

    cv2.putText(frame, controls_line_1, (20, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, controls_line_2, (20, start_y + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, controls_line_3, (20, start_y + (line_height * 2)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    if not exercise_active:
        # Faded overlay if no user is detected
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0,0,0), -1) # Use -1 for fill
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Center text
        text = "NO USER DETECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


# --- Exercise Processing Functions ---
# (All process_... functions remain unchanged from previous version)

def process_squats(landmarks, frame_shape):
    """Analyzes squat form and updates state."""
    global squat_stage
    rep_count = session_stats["squats"]["reps"]

    try:
        shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        hip = get_landmark_px(landmarks, 'left_hip', frame_shape)
        knee = get_landmark_px(landmarks, 'left_knee', frame_shape)
        ankle = get_landmark_px(landmarks, 'left_ankle', frame_shape)

        # Basic visibility check
        if any(coord == 0 for coord in shoulder + hip + knee + ankle): # Check if any landmark defaulted to [0,0]
             return rep_count, squat_stage # Skip processing if landmarks are missing

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        if knee_angle < 100 and hip_angle < 100:
            squat_stage = "down"
        if knee_angle > 160 and hip_angle > 170 and squat_stage == 'down':
            squat_stage = "up"
            rep_count += 1
            session_stats["squats"]["reps"] = rep_count
            print(f"Squat Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if squat_stage == 'down':
            if knee_angle > 100:
                say_feedback("Go lower for a full squat", "squats")
            elif hip_angle < 80:
                say_feedback("Keep your chest up", "squats")

        return rep_count, squat_stage

    except Exception:
        # print(f"Error processing squats: {e}") # Optional debug
        return rep_count, squat_stage

def process_curls(landmarks, frame_shape):
    """Analyzes bicep curl form (left arm) and updates state."""
    global curl_stage
    rep_count = session_stats["curls"]["reps"]

    try:
        shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        elbow = get_landmark_px(landmarks, 'left_elbow', frame_shape)
        wrist = get_landmark_px(landmarks, 'left_wrist', frame_shape)

        if any(coord == 0 for coord in shoulder + elbow + wrist):
             return rep_count, curl_stage

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        if elbow_angle < 50:
            curl_stage = "up"
        if elbow_angle > 160 and curl_stage == 'up':
            curl_stage = "down"
            rep_count += 1
            session_stats["curls"]["reps"] = rep_count
            print(f"Curl Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if curl_stage == 'up' and elbow_angle > 50:
            say_feedback("Bring your arm all the way up", "curls")
        elif curl_stage == 'down' and elbow_angle < 160:
            say_feedback("Straighten your arm fully", "curls")

        return rep_count, curl_stage

    except Exception:
        # print(f"Error processing curls: {e}")
        return rep_count, curl_stage

def process_jacks(landmarks, frame_shape):
    """Analyzes jumping jack form and updates state."""
    global jack_stage
    rep_count = session_stats["jacks"]["reps"]

    try:
        l_shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        r_shoulder = get_landmark_px(landmarks, 'right_shoulder', frame_shape)
        l_wrist = get_landmark_px(landmarks, 'left_wrist', frame_shape)
        r_wrist = get_landmark_px(landmarks, 'right_wrist', frame_shape)
        l_ankle = get_landmark_px(landmarks, 'left_ankle', frame_shape)
        r_ankle = get_landmark_px(landmarks, 'right_ankle', frame_shape)

        if any(coord == 0 for coord in l_shoulder + r_shoulder + l_wrist + r_wrist + l_ankle + r_ankle):
            return rep_count, jack_stage

        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        leg_distance = abs(l_ankle[0] - r_ankle[0])
        # Check if wrists are physically above shoulders (Y coordinate is smaller)
        arms_up = (l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1])

        # State definitions
        is_down = leg_distance < shoulder_width * 1.2 and not arms_up
        is_up = leg_distance > shoulder_width * 1.5 and arms_up

        if is_up:
            jack_stage = "up"
        if is_down and jack_stage == 'up':
            jack_stage = "down"
            rep_count += 1
            session_stats["jacks"]["reps"] = rep_count
            print(f"Jack Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        # Feedback conditions
        if jack_stage == 'up':
            if not arms_up:
                say_feedback("Bring your arms all the way up", "jacks")
            if leg_distance < shoulder_width * 1.5:
                say_feedback("Spread your legs wider", "jacks")

        return rep_count, jack_stage

    except Exception:
        # print(f"Error processing jacks: {e}")
        return rep_count, jack_stage

def process_pushups(landmarks, frame_shape):
    """Analyzes push-up form (side view) and updates state."""
    global pushup_stage
    rep_count = session_stats["pushups"]["reps"]

    try:
        # Assuming a side view, tracks left side
        shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        elbow = get_landmark_px(landmarks, 'left_elbow', frame_shape)
        wrist = get_landmark_px(landmarks, 'left_wrist', frame_shape)
        hip = get_landmark_px(landmarks, 'left_hip', frame_shape)
        knee = get_landmark_px(landmarks, 'left_knee', frame_shape)

        if any(coord == 0 for coord in shoulder + elbow + wrist + hip + knee):
             return rep_count, pushup_stage

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        body_angle = calculate_angle(shoulder, hip, knee) # Angle to check for straight back

        if elbow_angle < 90:
            pushup_stage = "down"
        if elbow_angle > 160 and pushup_stage == 'down':
            pushup_stage = "up"
            rep_count += 1
            session_stats["pushups"]["reps"] = rep_count
            print(f"Pushup Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if body_angle < 160:
            say_feedback("Keep your back straight", "pushups")
        if pushup_stage == 'down' and elbow_angle > 90:
            say_feedback("Go lower for a full push-up", "pushups")

        return rep_count, pushup_stage

    except Exception:
        # print(f"Error processing pushups: {e}")
        return rep_count, pushup_stage

def process_lunges(landmarks, frame_shape):
    """Analyzes lunge form (left leg forward) and updates state."""
    global lunge_stage
    rep_count = session_stats["lunges"]["reps"]

    try:
        # Tracks left leg as the forward leg
        l_hip = get_landmark_px(landmarks, 'left_hip', frame_shape)
        l_knee = get_landmark_px(landmarks, 'left_knee', frame_shape)
        l_ankle = get_landmark_px(landmarks, 'left_ankle', frame_shape)

        r_hip = get_landmark_px(landmarks, 'right_hip', frame_shape)
        r_knee = get_landmark_px(landmarks, 'right_knee', frame_shape)
        # Use a more stable point like hip for the back angle reference if ankle is unstable
        r_ankle = get_landmark_px(landmarks, 'right_ankle', frame_shape)

        if any(coord == 0 for coord in l_hip + l_knee + l_ankle + r_hip + r_knee + r_ankle):
             return rep_count, lunge_stage


        front_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        # Angle of back leg thigh relative to torso (approx)
        back_thigh_angle = calculate_angle(r_knee, r_hip, l_hip) # Angle at right hip

        # Check vertical alignment for back knee (approximate)
        #back_knee_low_enough = r_knee[1] > (frame_shape[0] * 0.7) # Adjust threshold as needed

        is_down = front_knee_angle < 110 and back_thigh_angle > 80 # Back thigh more vertical
        is_up = front_knee_angle > 160 and back_thigh_angle < 70 # Back thigh more horizontal

        if is_down:
            lunge_stage = "down"
        if is_up and lunge_stage == 'down':
            lunge_stage = "up"
            rep_count += 1
            session_stats["lunges"]["reps"] = rep_count
            print(f"Lunge Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if lunge_stage == 'down':
            if front_knee_angle > 110:
                say_feedback("Lower your body", "lunges")
            # Check if front knee's X is significantly ahead of ankle's X
            if (l_knee[0] - l_ankle[0]) > 50: # Adjust threshold pixel value as needed
                say_feedback("Keep front knee behind toe", "lunges")

        return rep_count, lunge_stage

    except Exception:
        # print(f"Error processing lunges: {e}")
        return rep_count, lunge_stage

def process_overhead_press(landmarks, frame_shape):
    """Analyzes overhead press form and updates state."""
    global press_stage
    rep_count = session_stats["press"]["reps"]

    try:
        # Tracks both arms
        l_shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        l_elbow = get_landmark_px(landmarks, 'left_elbow', frame_shape)
        l_wrist = get_landmark_px(landmarks, 'left_wrist', frame_shape)

        r_shoulder = get_landmark_px(landmarks, 'right_shoulder', frame_shape)
        r_elbow = get_landmark_px(landmarks, 'right_elbow', frame_shape)
        r_wrist = get_landmark_px(landmarks, 'right_wrist', frame_shape)

        if any(coord == 0 for coord in l_shoulder + l_elbow + l_wrist + r_shoulder + r_elbow + r_wrist):
             return rep_count, press_stage


        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Arms are up if wrists are above shoulders and elbows are straight
        arms_up = (l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1] and
                   l_elbow_angle > 160 and r_elbow_angle > 160)
        # Arms are down if wrists are near shoulders and elbows are bent
        arms_down = (l_wrist[1] > l_shoulder[1] - 30 and r_wrist[1] > r_shoulder[1] - 30 and # Allow slightly above
                     l_elbow_angle < 100 and r_elbow_angle < 100)

        if arms_up:
            press_stage = "up"
        if arms_down and press_stage == 'up':
            press_stage = "down"
            rep_count += 1
            session_stats["press"]["reps"] = rep_count
            print(f"Press Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if press_stage == 'up' and (l_elbow_angle < 160 or r_elbow_angle < 160):
            say_feedback("Extend your arms fully", "press")
        elif press_stage == 'down' and (l_elbow_angle > 100 or r_elbow_angle > 100):
             # Only give feedback if arms aren't fully down yet during the down phase
             if l_wrist[1] > l_shoulder[1] or r_wrist[1] > r_shoulder[1]:
                say_feedback("Lower arms to shoulders", "press")


        return rep_count, press_stage

    except Exception:
        # print(f"Error processing press: {e}")
        return rep_count, press_stage

def process_lateral_raises(landmarks, frame_shape):
    """Analyzes lateral raise form and updates state."""
    global raise_stage
    rep_count = session_stats["raises"]["reps"]

    try:
        l_shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        l_elbow = get_landmark_px(landmarks, 'left_elbow', frame_shape)
        l_wrist = get_landmark_px(landmarks, 'left_wrist', frame_shape)
        l_hip = get_landmark_px(landmarks, 'left_hip', frame_shape)

        r_shoulder = get_landmark_px(landmarks, 'right_shoulder', frame_shape)
        r_elbow = get_landmark_px(landmarks, 'right_elbow', frame_shape)
        r_wrist = get_landmark_px(landmarks, 'right_wrist', frame_shape)
        r_hip = get_landmark_px(landmarks, 'right_hip', frame_shape)

        if any(coord == 0 for coord in l_shoulder + l_elbow + l_wrist + l_hip +
                                        r_shoulder + r_elbow + r_wrist + r_hip):
             return rep_count, raise_stage

        # Angle at the shoulder (armpit) - Use hip as stable anchor
        l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        # Angle at the elbow (to check for straight arms)
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Arms are up if raised roughly parallel to the ground (angle ~90)
        arms_up = l_shoulder_angle > 75 and r_shoulder_angle > 75
        # Arms are down if close to the body
        arms_down = l_shoulder_angle < 30 and r_shoulder_angle < 30

        if arms_up:
            raise_stage = "up"
        if arms_down and raise_stage == 'up':
            raise_stage = "down"
            rep_count += 1
            session_stats["raises"]["reps"] = rep_count
            print(f"Raise Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if raise_stage == 'up':
            if l_elbow_angle < 150 or r_elbow_angle < 150:
                say_feedback("Keep your arms straighter", "raises")
            # Check if wrists go significantly above shoulders
            if l_wrist[1] < l_shoulder[1] - 30 or r_wrist[1] < r_shoulder[1] - 30: # 30px tolerance
                say_feedback("Don't raise above shoulder height", "raises")
            if l_shoulder_angle < 75 or r_shoulder_angle < 75:
                 say_feedback("Raise arms parallel to floor", "raises")


        return rep_count, raise_stage

    except Exception:
        # print(f"Error processing raises: {e}")
        return rep_count, raise_stage

def process_good_mornings(landmarks, frame_shape):
    """Analyzes 'Good Morning' (hip hinge) form and updates state."""
    global good_morning_stage
    rep_count = session_stats["goodmornings"]["reps"]

    try:
        # Tracks left side
        shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        hip = get_landmark_px(landmarks, 'left_hip', frame_shape)
        knee = get_landmark_px(landmarks, 'left_knee', frame_shape)
        ankle = get_landmark_px(landmarks, 'left_ankle', frame_shape)

        if any(coord == 0 for coord in shoulder + hip + knee + ankle):
             return rep_count, good_morning_stage

        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)

        is_down = hip_angle < 110 # Allow slightly more bend than before
        is_up = hip_angle > 165 # Slightly less than fully straight

        if is_down:
            good_morning_stage = "down"
        if is_up and good_morning_stage == 'down':
            good_morning_stage = "up"
            rep_count += 1
            session_stats["goodmornings"]["reps"] = rep_count
            print(f"Good Morning Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if knee_angle < 150:
            say_feedback("Keep legs straighter, bend at hips", "goodmornings")
        if good_morning_stage == 'down' and hip_angle > 110:
            say_feedback("Hinge lower from your hips", "goodmornings")

        return rep_count, good_morning_stage

    except Exception:
        # print(f"Error processing good mornings: {e}")
        return rep_count, good_morning_stage

def process_high_knees(landmarks, frame_shape):
    """Analyzes high knees form and updates state."""
    global high_knee_stage
    rep_count = session_stats["highknees"]["reps"]

    try:
        # Get normalized Y coordinates (0.0 at top, 1.0 at bottom)
        l_knee_y = get_landmark(landmarks, 'left_knee')[0]
        l_hip_y = get_landmark(landmarks, 'left_hip')[0]
        r_knee_y = get_landmark(landmarks, 'right_knee')[0]
        r_hip_y = get_landmark(landmarks, 'right_hip')[0]

        # Check confidence scores
        l_knee_conf = get_landmark(landmarks, 'left_knee')[2]
        l_hip_conf = get_landmark(landmarks, 'left_hip')[2]
        r_knee_conf = get_landmark(landmarks, 'right_knee')[2]
        r_hip_conf = get_landmark(landmarks, 'right_hip')[2]

        # Only process if key landmarks are visible
        if min(l_knee_conf, l_hip_conf, r_knee_conf, r_hip_conf) < 0.3:
            return rep_count, high_knee_stage

        # Knee is 'up' if its Y coord is less than (higher than) the hip's Y coord
        # Add a threshold to avoid minor movements counting
        knee_threshold = 0.05
        left_knee_up = l_knee_y < (l_hip_y - knee_threshold)
        right_knee_up = r_knee_y < (r_hip_y - knee_threshold)

        if high_knee_stage == 'down' and (left_knee_up or right_knee_up):
            high_knee_stage = 'up' # One knee is up
            if left_knee_up:
                if l_knee_y > (l_hip_y - 0.1): # Check if knee is *barely* above hip
                     say_feedback("Lift left knee higher", "highknees")
            elif right_knee_up:
                if r_knee_y > (r_hip_y - 0.1):
                     say_feedback("Lift right knee higher", "highknees")

        elif high_knee_stage == 'up' and not left_knee_up and not right_knee_up:
            high_knee_stage = 'down' # Both knees are down
            rep_count += 1 # Count one rep after one cycle (e.g., left up, then down)
            session_stats["highknees"]["reps"] = rep_count
            print(f"High Knee Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        return rep_count, high_knee_stage

    except Exception:
        # print(f"Error processing high knees: {e}")
        return rep_count, high_knee_stage

def process_crunches(landmarks, frame_shape):
    """Analyzes crunch form (side view) and updates state."""
    global crunch_stage
    rep_count = session_stats["crunches"]["reps"]

    try:
        # Assuming side view, tracks left side
        shoulder = get_landmark_px(landmarks, 'left_shoulder', frame_shape)
        hip = get_landmark_px(landmarks, 'left_hip', frame_shape)
        knee = get_landmark_px(landmarks, 'left_knee', frame_shape)

        if any(coord == 0 for coord in shoulder + hip + knee):
             return rep_count, crunch_stage

        # Angle at the hip. Crunched = smaller angle.
        hip_angle = calculate_angle(shoulder, hip, knee)

        is_down = hip_angle > 115 # Lying relatively flat
        is_up = hip_angle < 100 # Crunched up

        if is_up:
            crunch_stage = "up"
        if is_down and crunch_stage == 'up':
            crunch_stage = "down"
            rep_count += 1
            session_stats["crunches"]["reps"] = rep_count
            print(f"Crunch Count: {rep_count}")
            say_feedback(str(rep_count))
            reset_feedback_state() # Reset for next rep

        if crunch_stage == 'up' and hip_angle > 100:
             # Only give feedback if not fully crunched
             say_feedback("Lift shoulders higher", "crunches")


        return rep_count, crunch_stage

    except Exception:
        # print(f"Error processing crunches: {e}")
        return rep_count, crunch_stage


# --- Report Generation ---

def generate_report():
    """Generates a final session report image."""
    # Create a 1080p image
    report_img = np.full((1080, 1920, 3), (255, 255, 255), dtype=np.uint8)
    y_pos = 100
    line_height = 50
    left_margin = 100

    def put_text(text, y, size=1.0, color=(0, 0, 0), bold=False):
        thickness = 3 if bold else 2
        cv2.putText(report_img, text, (left_margin, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

    def draw_line(y):
        cv2.line(report_img, (left_margin, y), (1920 - left_margin, y), (200, 200, 200), 2)

    put_text("AI Physiotherapist - Session Report", y_pos, 2.5, (0, 165, 255), bold=True)
    y_pos += 80
    draw_line(y_pos)
    y_pos += 80

    total_reps = 0
    exercises_done = 0
    for exercise, data in session_stats.items():
        if data["reps"] > 0:
            exercises_done +=1
            total_reps += data["reps"]
            put_text(f"Exercise: {exercise.upper()}", y_pos, 1.5, bold=True)
            y_pos += line_height + 20
            put_text(f"Total Reps: {data['reps']}", y_pos, 1.2)
            y_pos += line_height

            if data["feedback"]:
                try:
                    feedback_counts = Counter(data["feedback"])
                    most_common_tip = feedback_counts.most_common(1)[0]
                    tip_text = f"'{most_common_tip[0]}' (repeated {most_common_tip[1]} times)"

                    put_text("Your Top Tip:", y_pos, 1.2)
                    y_pos += line_height
                    put_text(tip_text, y_pos, 1.1, (0, 0, 180)) # Red text for tips
                except IndexError: # Handle case where feedback list might be empty unexpectedly
                     put_text("Top Tip: No specific feedback recorded.", y_pos + line_height, 1.1, (100, 100, 100))

            else:
                put_text("Your Top Tip: Perfect form! No feedback given.", y_pos, 1.1, (0, 180, 0)) # Green text for good form

            y_pos += 100
            # Add line only if there's more space and more exercises
            if y_pos < 900 and exercises_done < sum(1 for d in session_stats.values() if d["reps"] > 0):
                draw_line(y_pos - 30)
            elif y_pos >= 900:
                 break # Stop adding exercises if report is full


    if total_reps == 0:
        put_text("No exercises were completed in this session.", y_pos, 1.2)
        y_pos += 60

    put_text("Press 'q' to close this report.", (report_img.shape[0] - 60), 1.0, (100, 100, 100))
    return report_img

# --- Main Application Loop ---

def main():
    global current_exercise, exercise_active

    # --- CAMERA SELECTION ---
    # Try 1, 0, 2 in order
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Warning: Could not open camera at index 1. Trying index 0.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam at index 0 or 1. Please check camera.")
            return

    print("AI Physiotherapist starting...")

    window_name = 'AI Physiotherapist'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- Initial Pose Suggestion ---
    suggestion_made = False
    suggestion_start_time = time.time()

    while cap.isOpened() and not suggestion_made:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera frame empty during initial check.")
            time.sleep(1) # Wait a bit before retrying or exiting
            if time.time() - suggestion_start_time > 10: # Timeout after 10s
                 print("Exiting due to persistent camera read error.")
                 cap.release()
                 cv2.destroyAllWindows()
                 return
            continue

        frame = cv2.flip(frame, 1) # Flip horizontally

        # Quick inference for pose check
        img = cv2.resize(frame, (192, 192))
        input_image = tf.expand_dims(tf.cast(img, dtype=tf.int32), axis=0)
        try:
            results = movenet(input_image); keypoints = results['output_0'].numpy()
            landmarks = np.squeeze(keypoints)
            # Check confidence before suggesting
            if np.mean(landmarks[:, 2]) > 0.3:
                initial_pose = analyze_initial_pose(landmarks)
                suggest_exercises(initial_pose)
                suggestion_made = True
                time.sleep(4) # Give user time to hear suggestion
            else:
                 # Display "Detecting..." message
                 h, w, _ = frame.shape
                 text = "Detecting User..."
                 (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                 cv2.putText(frame, text, ((w-tw)//2, (h+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                 cv2.imshow(window_name, frame)
                 if cv2.waitKey(10) & 0xFF == ord('q'): # Allow quitting during detection
                      cap.release()
                      cv2.destroyAllWindows()
                      return


        except Exception as e:
            print(f"Warning: Initial inference failed. {e}")
            time.sleep(0.5) # Wait before retrying


        # Exit if it takes too long to detect someone
        if time.time() - suggestion_start_time > 15 and not suggestion_made:
             print("No user detected after 15 seconds. Starting with default.")
             say_feedback(f"Welcome. Starting with {current_exercise}. Check controls.", None, force_say=True)
             suggestion_made = True # Proceed anyway
             time.sleep(2)


    print("Starting main exercise loop. Check UI for controls. Press 'q' to quit.")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera frame is empty during main loop. Exiting.")
            break # Exit loop if camera fails
        frame = cv2.flip(frame, 1)

        # --- Model Inference ---
        img = cv2.resize(frame, (192, 192))
        input_image = tf.expand_dims(tf.cast(img, dtype=tf.int32), axis=0)
        try:
            results = movenet(input_image); keypoints = results['output_0'].numpy()
        except Exception as e:
            print(f"Warning: Model inference failed. Skipping frame. {e}")
            continue # Skip frame on error

        landmarks = np.squeeze(keypoints)
        # Check overall confidence, more robustly
        visible_keypoints = landmarks[landmarks[:, 2] > 0.3]
        exercise_active = len(visible_keypoints) >= 10 # Require at least 10 visible keypoints


        rep_count, stage = 0, ""

        # Process exercise logic only if active
        if exercise_active:
            try:
                # Dynamically call the correct process_ function based on current_exercise
                process_func_name = f"process_{current_exercise}"
                if process_func_name in globals():
                    process_func = globals()[process_func_name]
                    rep_count, stage = process_func(landmarks, frame.shape)
                else:
                    print(f"Warning: No processing function named {process_func_name} found.")
                    stage = "N/A" # Indicate exercise cannot be processed
            except Exception as e:
                 # print(f"Error in processing {current_exercise}: {e}") # Optional: for debugging
                 # Don't reset stage here, let it persist or be N/A
                 pass # Silently ignore errors during processing, keep last known stage/rep
                 rep_count = session_stats[current_exercise]['reps'] # Keep last rep count
                 stage = globals().get(f"{current_exercise}_stage", "Error") # Keep last stage or show Error

        # --- Draw Visuals ---
        # Draw skeleton only if active to avoid drawing on empty frames
        if exercise_active:
            draw_keypoints_and_skeleton(frame, keypoints, 0.3) # Lower threshold slightly
        # Always draw UI
        draw_ui(frame, rep_count, stage, current_exercise)
        cv2.imshow(window_name, frame)

        # --- Key Controls ---
        key = cv2.waitKey(10) & 0xFF
        def switch_exercise(ex_name, ex_speech):
            global current_exercise, last_feedback_message
            # Reset stage for the *old* exercise before switching
            globals()[f"{current_exercise}_stage"] = "up" if current_exercise in ["squats", "pushups", "lunges", "press", "goodmornings", "crunches"] else "down"

            if current_exercise != ex_name:
                current_exercise = ex_name
                reset_feedback_state() # Reset feedback when switching
                say_feedback(ex_speech, None, force_say=True) # Force say new exercise name

        if key == ord('q'): say_feedback("Workout complete. Generating report.", None, force_say=True); time.sleep(2); break
        elif key == ord('s'): switch_exercise("squats", "Squats")
        elif key == ord('c'): switch_exercise("curls", "Bicep Curls. Left arm.")
        elif key == ord('j'): switch_exercise("jacks", "Jumping Jacks")
        elif key == ord('p'): switch_exercise("pushups", "Pushups. Side view.")
        elif key == ord('l'): switch_exercise("lunges", "Lunges. Left leg forward.")
        elif key == ord('o'): switch_exercise("press", "Overhead Press")
        elif key == ord('r'): switch_exercise("raises", "Lateral Raises")
        elif key == ord('g'): switch_exercise("goodmornings", "Good Mornings. Side view.")
        elif key == ord('h'): switch_exercise("highknees", "High Knees")
        elif key == ord('n'): switch_exercise("crunches", "Crunches. Side view.")

    # --- Cleanup and Report ---
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

    # --- Display Final Report ---
    print("\n--- SESSION REPORT ---")
    total_reps_all = 0
    for ex, data in session_stats.items():
        if data["reps"] > 0:
            total_reps_all += data["reps"]
            print(f"\n{ex.upper()}: Reps: {data['reps']}")
            if data["feedback"]:
                try:
                    counts = Counter(data["feedback"]); tip, count = counts.most_common(1)[0]
                    print(f"  Top Tip: '{tip}' ({count} times)")
                except IndexError:
                    print("  Top Tip: No specific feedback recorded.")
            else: print("  Top Tip: Perfect form!")
    if total_reps_all == 0: print("No exercises completed.")
    print("\nClose report window to exit.")

    # Generate and display report only if some reps were done or if forced (e.g., debug)
    if total_reps_all >= 0: # Show report even if 0 reps
        try:
            report_image = generate_report()
            report_window = 'Session Report'; cv2.namedWindow(report_window, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(report_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(report_window, report_image)
            while cv2.waitKey(10) & 0xFF != ord('q'): pass
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying final report: {e}")
            cv2.destroyAllWindows() # Ensure all windows close on error

if __name__ == '__main__':
    main()

