import cv2
import numpy as np
import pyttsx3
import threading
import time
import tensorflow as tf
import tensorflow_hub as hub


# --- Text-to-Speech Setup ---
class TTS:
    """Handles Text-to-Speech in a non-blocking manner."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
        self.last_spoken_time = {}

    def say(self, text, cooldown=3):
        """Speaks text if the cooldown for that specific text has passed."""
        with self.lock:
            current_time = time.time()
            if text not in self.last_spoken_time or (current_time - self.last_spoken_time[text]) > cooldown:
                self.last_spoken_time[text] = current_time
                threading.Thread(target=self._speak_text, args=(text,)).start()

    def _speak_text(self, text):
        """Internal method to synthesize speech, handles runtime errors."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError:
            pass  # Ignore "run loop already started" error
        except Exception as e:
            print(f"Error in TTS: {e}")


# --- Pose Estimation Helper Functions ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


# --- MoveNet Keypoint and Drawing Utilities ---
KEYPOINT_DICT = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(keypoints)
    for edge in edges:
        p1, p2 = edge
        if p1 < shaped.shape[0] and p2 < shaped.shape[0]:
            y1, x1, c1 = shaped[p1];
            y2, x2, c2 = shaped[p2]
            if c1 > confidence_threshold and c2 > confidence_threshold:
                cv2.line(frame, (int(x1 * x), int(y1 * y)), (int(x2 * x), int(y2 * y)), (255, 0, 0), 2)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(keypoints)
    for kp in shaped:
        ky, kx, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 4, (0, 255, 0), -1)


# --- Exercise Analysis Functions ---
def analyze_squats(landmarks, stage, counter):
    feedback = ""
    shoulder, hip, knee, ankle = (landmarks.get('left_shoulder'), landmarks.get('left_hip'), landmarks.get('left_knee'),
                                  landmarks.get('left_ankle'))
    if not all([shoulder, hip, knee, ankle]):
        shoulder, hip, knee, ankle = (landmarks.get('right_shoulder'), landmarks.get('right_hip'),
                                      landmarks.get('right_knee'), landmarks.get('right_ankle'))

    if all([shoulder, hip, knee, ankle]):
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        if knee_angle < 100 and hip_angle < 100:
            stage = "down"
            feedback = "Go deeper" if hip[1] > knee[1] else ""
        if knee_angle > 160 and hip_angle > 160 and stage == 'down':
            stage = "up"
            counter += 1
            feedback = "Good Rep!"
    else:
        feedback = "Show full body"
    return stage, counter, feedback


def analyze_bicep_curls(landmarks, stage, counter, initial_elbow):
    feedback = ""
    shoulder, elbow, wrist = landmarks.get('left_shoulder'), landmarks.get('left_elbow'), landmarks.get('left_wrist')
    if not all([shoulder, elbow, wrist]):
        shoulder, elbow, wrist = landmarks.get('right_shoulder'), landmarks.get('right_elbow'), landmarks.get(
            'right_wrist')

    if all([shoulder, elbow, wrist]):
        angle = calculate_angle(shoulder, elbow, wrist)

        if angle > 160:
            stage = "down"
            if initial_elbow is None: initial_elbow = elbow
        if angle < 30 and stage == 'down':
            stage = "up"
            counter += 1
            feedback = "Good Curl!"
            initial_elbow = None  # Reset for next rep

        if initial_elbow:
            elbow_movement = np.linalg.norm(np.array(elbow) - np.array(initial_elbow))
            if elbow_movement > 0.1:
                feedback = "Keep elbow still"
    else:
        feedback = "Show your arm"
        initial_elbow = None
    return stage, counter, feedback, initial_elbow


def analyze_overhead_press(landmarks, stage, counter):
    feedback = ""
    shoulder_l, elbow_l, wrist_l = landmarks.get('left_shoulder'), landmarks.get('left_elbow'), landmarks.get(
        'left_wrist')
    shoulder_r, elbow_r, wrist_r = landmarks.get('right_shoulder'), landmarks.get('right_elbow'), landmarks.get(
        'right_wrist')

    if all([shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]):
        elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
        elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

        if wrist_l[1] < shoulder_l[1] and wrist_r[1] < shoulder_r[1]:
            stage = "up"
        if stage == 'up' and elbow_angle_l < 90 and elbow_angle_r < 90:
            stage = "down"
            counter += 1
            feedback = "Great Press!"

        if stage == "up" and (elbow_angle_l < 160 or elbow_angle_r < 160):
            feedback = "Extend fully"
    else:
        feedback = "Show upper body"
    return stage, counter, feedback


# --- Main Application ---
def main():
    print("Loading MoveNet model...")
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # State variables
    counters = {'squats': 0, 'bicep_curls': 0, 'overhead_press': 0}
    stage = None
    current_exercise = 'squats'
    feedback = ""
    initial_elbow_pos = None  # For bicep curl stabilization

    tts = TTS()
    print("Starting AI Physiotherapist. Press 's', 'b', or 'p' to switch exercises. Press 'q' to quit.")
    tts.say(f"Welcome! Let's start with {current_exercise}.")

    window_name = 'AI Physiotherapist Pro'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.int32)
        results = movenet(input=input_image)
        keypoints = results['output_0'].numpy()

        try:
            landmarks = {KEYPOINT_DICT[i]: [c[1], c[0]] for i, c in enumerate(np.squeeze(keypoints)) if c[2] > 0.5}

            if current_exercise == 'squats':
                stage, counters['squats'], feedback = analyze_squats(landmarks, stage, counters['squats'])
            elif current_exercise == 'bicep_curls':
                stage, counters['bicep_curls'], feedback, initial_elbow_pos = analyze_bicep_curls(landmarks, stage,
                                                                                                  counters[
                                                                                                      'bicep_curls'],
                                                                                                  initial_elbow_pos)
            elif current_exercise == 'overhead_press':
                stage, counters['overhead_press'], feedback = analyze_overhead_press(landmarks, stage,
                                                                                     counters['overhead_press'])
        except Exception as e:
            feedback = "Detection error"
            print(f"Error processing landmarks: {e}")

        if feedback:
            tts.say(feedback, cooldown=2)

        # --- NEW: Enhanced UI Rendering ---
        overlay = frame.copy()
        alpha = 0.6

        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        BLUE = (255, 120, 0)

        # Draw connections and keypoints first
        draw_connections(frame, keypoints, EDGES, 0.5)
        draw_keypoints(frame, keypoints, 0.5)

        # Draw semi-transparent status bar
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), BLUE, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Determine feedback color
        feedback_color = GREEN if "Good" in feedback or "Great" in feedback else YELLOW

        # Display UI elements
        # Exercise
        cv2.putText(frame, 'EXERCISE', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, current_exercise.upper(), (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)

        # Reps
        rep_text_size = cv2.getTextSize(str(counters[current_exercise]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.putText(frame, 'REPS', (frame.shape[1] - 15 - rep_text_size[0] - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, str(counters[current_exercise]), (frame.shape[1] - 15 - rep_text_size[0], 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)

        # Feedback
        feedback_text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        feedback_x = int((frame.shape[1] - feedback_text_size[0]) / 2)
        cv2.putText(frame, feedback, (feedback_x, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color,
                    2, cv2.LINE_AA)

        # Controls instructions
        controls_text = "[S]quats | [B]icep Curls | [P]ress | [Q]uit"
        controls_text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, controls_text, (frame.shape[1] - controls_text_size[0] - 15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('s'), ord('b'), ord('p')]:
            if key == ord('s'):
                new_exercise = 'squats'
            elif key == ord('b'):
                new_exercise = 'bicep_curls'
            elif key == ord('p'):
                new_exercise = 'overhead_press'

            if current_exercise != new_exercise:
                current_exercise = new_exercise
                stage = None
                initial_elbow_pos = None  # Reset for bicep curls
                tts.say(f"Switched to {current_exercise.replace('_', ' ')}.")

    cap.release()
    cv2.destroyAllWindows()
    total_reps = sum(counters.values())
    print(f"Session ended. Total reps: {total_reps}")
    tts.say(f"Great workout! You completed {total_reps} total reps.")


if __name__ == '__main__':
    main()

