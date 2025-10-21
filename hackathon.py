import cv2
import numpy as np
import pyttsx3
import threading
import time
import tensorflow as tf
import tensorflow_hub as hub


# --- Text-to-Speech Setup ---
# This class will handle text-to-speech in a separate thread to avoid blocking the main video processing loop.
class TTS:
    """
    A class to handle Text-to-Speech operations in a non-blocking manner.
    """

    def __init__(self):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
        self.last_spoken_time = {}

    def say(self, text, cooldown=3):
        """
        Speaks the given text if the cooldown period has passed for that specific text.
        """
        with self.lock:
            current_time = time.time()
            if text not in self.last_spoken_time or (current_time - self.last_spoken_time[text]) > cooldown:
                self.last_spoken_time[text] = current_time
                # Run the speech in a new thread to avoid blocking
                threading.Thread(target=self._speak_text, args=(text,)).start()

    def _speak_text(self, text):
        """
        The actual speech synthesis method.
        Now handles the 'run loop already started' error gracefully.
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError:
            # This catches the "run loop already started" error.
            # We can safely ignore it and let the current speech finish.
            pass
        except Exception as e:
            print(f"Error in TTS: {e}")


# Initialize the TTS engine
tts = TTS()


# --- Pose Estimation Helper Function ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (landmarks).
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# --- MoveNet Keypoint and Drawing Utilities ---
# Dictionary to map keypoint indices from MoveNet to names
KEYPOINT_DICT = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

# Defines the connections between keypoints for drawing the skeleton
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]


def draw_connections(frame, keypoints, edges, confidence_threshold):
    """Draws lines connecting the keypoints to form a skeleton."""
    y, x, _ = frame.shape
    shaped = np.squeeze(keypoints)

    for edge in edges:
        p1, p2 = edge
        if p1 < shaped.shape[0] and p2 < shaped.shape[0]:
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if c1 > confidence_threshold and c2 > confidence_threshold:
                cv2.line(frame, (int(x1 * x), int(y1 * y)), (int(x2 * x), int(y2 * y)), (255, 0, 0), 2)


def draw_keypoints(frame, keypoints, confidence_threshold):
    """Draws circles at the location of detected keypoints."""
    y, x, _ = frame.shape
    shaped = np.squeeze(keypoints)

    for kp in shaped:
        ky, kx, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 4, (0, 255, 0), -1)


# --- Main Application ---
def main():
    # Load MoveNet model from TensorFlow Hub
    print("Loading MoveNet model...")
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']
    print("Model loaded successfully.")

    # Start webcam capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables for squat counter and exercise state
    squat_counter = 0
    stage = None  # Can be "up" or "down"
    feedback_message = ""

    print("Starting AI Physiotherapist. Press 'q' to quit.")
    tts.say("Welcome to the AI Physiotherapist. Let's begin your squat analysis.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If the frame could not be grabbed, break the loop.
            print("Error: Failed to capture image. Exiting.")
            break

        # --- MoveNet Processing ---
        # Resize and format the frame for the model
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.int32)

        # --- MODIFIED SECTION ---
        # Run inference
        # The model expects a keyword argument 'input', not a positional one.
        results = movenet(input=input_image)
        # --- END OF MODIFIED SECTION ---
        keypoints = results['output_0'].numpy()

        # --- Landmark and Angle Calculation ---
        try:
            # Create a dictionary of landmarks from the keypoints array
            landmarks = {}
            keypoints_squeezed = np.squeeze(keypoints)
            for i in range(keypoints_squeezed.shape[0]):
                y, x, conf = keypoints_squeezed[i]
                if conf > 0.5:
                    landmarks[KEYPOINT_DICT[i]] = [x, y]
                else:
                    landmarks[KEYPOINT_DICT[i]] = None

            # Get coordinates for key joints
            shoulder_l = landmarks.get('left_shoulder')
            hip_l = landmarks.get('left_hip')
            knee_l = landmarks.get('left_knee')
            ankle_l = landmarks.get('left_ankle')

            shoulder_r = landmarks.get('right_shoulder')
            hip_r = landmarks.get('right_hip')
            knee_r = landmarks.get('right_knee')
            ankle_r = landmarks.get('right_ankle')

            # Use left side by default, fallback to right if left is not visible
            if shoulder_l and hip_l and knee_l and ankle_l:
                shoulder, hip, knee, ankle = shoulder_l, hip_l, knee_l, ankle_l
            elif shoulder_r and hip_r and knee_r and ankle_r:
                shoulder, hip, knee, ankle = shoulder_r, hip_r, knee_r, ankle_r
            else:
                shoulder, hip, knee, ankle = None, None, None, None

            if all([shoulder, hip, knee, ankle]):
                # Calculate angles
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)

                # --- Squat logic ---
                feedback_message = ""

                if knee_angle < 100 and hip_angle < 100:
                    stage = "down"
                    if hip[1] < knee[1]:  # A simple check for depth: hip is below knee
                        pass
                    else:
                        feedback_message = "Go deeper"
                        tts.say(feedback_message)

                if knee_angle > 160 and hip_angle > 160 and stage == 'down':
                    stage = "up"
                    squat_counter += 1
                    feedback_message = "Good Rep!"
            else:
                feedback_message = "Please make sure your full body is visible."
                tts.say(feedback_message, cooldown=5)

        except Exception as e:
            feedback_message = "No pose detected. Stand in frame."
            print(f"Error processing landmarks: {e}")
            pass

        # --- Render UI on the screen ---
        # Draw skeleton
        draw_connections(frame, keypoints, EDGES, 0.5)
        draw_keypoints(frame, keypoints, 0.5)

        # Status box
        cv2.rectangle(frame, (0, 0), (300, 120), (245, 117, 16), -1)

        # Rep data
        cv2.putText(frame, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(squat_counter), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(frame, 'STAGE', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, stage, (90, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Feedback message
        cv2.putText(frame, feedback_message, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('AI Physiotherapist', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total squats: {squat_counter}")
    tts.say(f"Great job! You completed {squat_counter} squats.")


if __name__ == '__main__':
    main()

