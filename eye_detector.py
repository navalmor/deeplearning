import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import argparse
import time

class EyeDetector:
    def __init__(self, ear_threshold=0.3, ear_consec_frames=3):
        """
        Initialize the Eye Detector
        
        Args:
            ear_threshold (float): Eye Aspect Ratio threshold for detecting closed eyes
            ear_consec_frames (int): Number of consecutive frames below threshold to consider eyes closed
        """
        self.EAR_THRESHOLD = ear_threshold
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        
        # Counters
        self.COUNTER = 0
        self.blink_count = 0
        
        # Initialize dlib's face detector and facial landmark predictor
        print("Initializing face detector and landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Grab the indexes of the facial landmarks for the left and right eye
        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 42
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 48
    
    def eye_aspect_ratio(self, eye):
        """
        Calculate the Eye Aspect Ratio (EAR) for given eye landmarks
        
        Args:
            eye: Array of (x, y) coordinates for eye landmarks
            
        Returns:
            float: Eye Aspect Ratio value
        """
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, shape, eye_start, eye_end):
        """
        Extract eye landmarks from facial landmarks
        
        Args:
            shape: dlib shape object containing all facial landmarks
            eye_start: Starting index for eye landmarks
            eye_end: Ending index for eye landmarks
            
        Returns:
            numpy array: Eye landmark coordinates
        """
        eye_points = []
        for i in range(eye_start, eye_end):
            eye_points.append([shape.part(i).x, shape.part(i).y])
        return np.array(eye_points)
    
    def draw_eye_landmarks(self, frame, eye_landmarks):
        """
        Draw eye landmarks on the frame
        
        Args:
            frame: Input frame
            eye_landmarks: Eye landmark coordinates
        """
        cv2.polylines(frame, [eye_landmarks], True, (0, 255, 0), 1)
        for point in eye_landmarks:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
    
    def detect_eyes(self, frame):
        """
        Main eye detection function
        
        Args:
            frame: Input frame from camera
            
        Returns:
            tuple: (processed_frame, eye_status, left_ear, right_ear)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        rects = self.detector(gray, 0)
        
        eye_status = "Eyes: Not Detected"
        left_ear = 0
        right_ear = 0
        
        # Loop over the face detections
        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = self.predictor(gray, rect)
            
            # Extract the left and right eye coordinates
            left_eye = self.extract_eye_landmarks(shape, self.LEFT_EYE_START, self.LEFT_EYE_END)
            right_eye = self.extract_eye_landmarks(shape, self.RIGHT_EYE_START, self.RIGHT_EYE_END)
            
            # Calculate the eye aspect ratio for both eyes
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio together for both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Draw the eye landmarks
            self.draw_eye_landmarks(frame, left_eye)
            self.draw_eye_landmarks(frame, right_eye)
            
            # Draw face rectangle
            cv2.rectangle(frame, (rect.left(), rect.top()), 
                         (rect.right(), rect.bottom()), (255, 0, 0), 2)
            
            # Check if the eye aspect ratio is below the blink threshold
            if ear < self.EAR_THRESHOLD:
                self.COUNTER += 1
                eye_status = "Eyes: CLOSED"
            else:
                # If the eyes were closed for a sufficient number of frames
                if self.COUNTER >= self.EAR_CONSEC_FRAMES:
                    self.blink_count += 1
                
                # Reset the eye frame counter
                self.COUNTER = 0
                eye_status = "Eyes: OPEN"
        
        return frame, eye_status, left_ear, right_ear
    
    def run_camera(self):
        """
        Main function to run the eye detection with camera feed
        """
        print("Starting camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera started successfully!")
        print("Press 'q' to quit, 'r' to reset blink counter")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect eyes
            frame, eye_status, left_ear, right_ear = self.detect_eyes(frame)
            
            # Display information on frame
            cv2.putText(frame, eye_status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Threshold: {self.EAR_THRESHOLD}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Eye Detection - Open/Closed', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.blink_count = 0
                print("Blink counter reset!")
            elif key == ord('+') or key == ord('='):
                self.EAR_THRESHOLD += 0.01
                print(f"EAR Threshold: {self.EAR_THRESHOLD:.3f}")
            elif key == ord('-'):
                self.EAR_THRESHOLD = max(0.1, self.EAR_THRESHOLD - 0.01)
                print(f"EAR Threshold: {self.EAR_THRESHOLD:.3f}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Application closed. Total blinks detected: {self.blink_count}")

def main():
    parser = argparse.ArgumentParser(description='Real-time Eye Detection - Open/Closed')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Eye Aspect Ratio threshold for detecting closed eyes (default: 0.3)')
    parser.add_argument('--frames', type=int, default=3,
                       help='Number of consecutive frames below threshold to consider eyes closed (default: 3)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("EYE DETECTION APPLICATION")
    print("=" * 50)
    print(f"EAR Threshold: {args.threshold}")
    print(f"Consecutive Frames: {args.frames}")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset blink counter")
    print("  - Press '+' to increase threshold")
    print("  - Press '-' to decrease threshold")
    print("=" * 50)
    
    try:
        detector = EyeDetector(ear_threshold=args.threshold, ear_consec_frames=args.frames)
        detector.run_camera()
    except FileNotFoundError:
        print("Error: shape_predictor_68_face_landmarks.dat not found!")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place it in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()