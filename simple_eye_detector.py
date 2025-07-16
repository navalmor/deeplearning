import cv2
import numpy as np
import argparse
import time

class SimpleEyeDetector:
    def __init__(self, blink_threshold=30, blink_consecutive_frames=3):
        """
        Initialize the Simple Eye Detector using OpenCV Haar Cascades
        
        Args:
            blink_threshold (int): Threshold for eye area change to detect blinks
            blink_consecutive_frames (int): Number of consecutive frames to consider blink
        """
        self.BLINK_THRESHOLD = blink_threshold
        self.BLINK_CONSECUTIVE_FRAMES = blink_consecutive_frames
        
        # Counters
        self.closed_counter = 0
        self.blink_count = 0
        self.last_eye_area = 0
        
        # Initialize Haar cascade classifiers
        print("Initializing Haar cascade classifiers...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise Exception("Error loading Haar cascade files!")
        
        print("Haar cascades loaded successfully!")
    
    def calculate_eye_area(self, eyes):
        """
        Calculate the total area of detected eyes
        
        Args:
            eyes: List of detected eye rectangles
            
        Returns:
            int: Total area of all detected eyes
        """
        total_area = 0
        for (x, y, w, h) in eyes:
            total_area += w * h
        return total_area
    
    def detect_eye_state(self, current_eye_area):
        """
        Determine if eyes are open or closed based on eye area changes
        
        Args:
            current_eye_area (int): Current total eye area
            
        Returns:
            str: Eye state ("OPEN", "CLOSED", or "NOT DETECTED")
        """
        if current_eye_area == 0:
            return "NOT DETECTED"
        
        # If this is the first frame, just store the area
        if self.last_eye_area == 0:
            self.last_eye_area = current_eye_area
            return "OPEN"
        
        # Calculate the percentage change in eye area
        area_change = abs(current_eye_area - self.last_eye_area)
        area_change_percent = (area_change / self.last_eye_area) * 100 if self.last_eye_area > 0 else 0
        
        # If eye area decreased significantly, eyes might be closing
        if current_eye_area < self.last_eye_area * 0.7:  # 30% decrease
            self.closed_counter += 1
            if self.closed_counter >= self.BLINK_CONSECUTIVE_FRAMES:
                return "CLOSED"
            else:
                return "CLOSING"
        else:
            # Eyes are open
            if self.closed_counter >= self.BLINK_CONSECUTIVE_FRAMES:
                self.blink_count += 1
                print(f"Blink detected! Total blinks: {self.blink_count}")
            
            self.closed_counter = 0
            self.last_eye_area = current_eye_area
            return "OPEN"
    
    def detect_eyes(self, frame):
        """
        Main eye detection function
        
        Args:
            frame: Input frame from camera
            
        Returns:
            tuple: (processed_frame, eye_status, eye_count, eye_area)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        eye_status = "Eyes: Not Detected"
        eye_count = 0
        current_eye_area = 0
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes (upper half of face)
            roi_gray = gray[y:y+int(h*0.6), x:x+w]
            roi_color = frame[y:y+int(h*0.6), x:x+w]
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            eye_count = len(eyes)
            
            # Draw eye rectangles and calculate area
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Draw a circle at the center of each eye
                center_x = ex + ew // 2
                center_y = ey + eh // 2
                cv2.circle(roi_color, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # Calculate total eye area
            current_eye_area = self.calculate_eye_area(eyes)
            
            # Determine eye state
            state = self.detect_eye_state(current_eye_area)
            
            if eye_count >= 2:
                eye_status = f"Eyes: {state}"
            elif eye_count == 1:
                eye_status = f"Eye: {state} (Only 1 detected)"
            else:
                eye_status = "Eyes: Not Detected"
                self.closed_counter = 0
        
        return frame, eye_status, eye_count, current_eye_area
    
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
        print("Press 'q' to quit, 'r' to reset blink counter, '+'/'-' to adjust sensitivity")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect eyes
            frame, eye_status, eye_count, eye_area = self.detect_eyes(frame)
            
            # Display information on frame
            cv2.putText(frame, eye_status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Eyes detected: {eye_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Eye area: {eye_area}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Threshold: {self.BLINK_THRESHOLD}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' quit, 'r' reset, '+'/'-' adjust", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Simple Eye Detection - Open/Closed', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.blink_count = 0
                print("Blink counter reset!")
            elif key == ord('+') or key == ord('='):
                self.BLINK_THRESHOLD += 5
                print(f"Blink Threshold: {self.BLINK_THRESHOLD}")
            elif key == ord('-'):
                self.BLINK_THRESHOLD = max(5, self.BLINK_THRESHOLD - 5)
                print(f"Blink Threshold: {self.BLINK_THRESHOLD}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Application closed. Total blinks detected: {self.blink_count}")

def main():
    parser = argparse.ArgumentParser(description='Simple Real-time Eye Detection - Open/Closed')
    parser.add_argument('--threshold', type=int, default=30,
                       help='Blink detection threshold (default: 30)')
    parser.add_argument('--frames', type=int, default=3,
                       help='Number of consecutive frames for blink detection (default: 3)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLE EYE DETECTION APPLICATION")
    print("=" * 60)
    print(f"Blink Threshold: {args.threshold}")
    print(f"Consecutive Frames: {args.frames}")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset blink counter")
    print("  - Press '+' to increase threshold")
    print("  - Press '-' to decrease threshold")
    print("=" * 60)
    print("NOTE: This version uses OpenCV Haar cascades")
    print("      For more accurate detection, use the dlib version")
    print("=" * 60)
    
    try:
        detector = SimpleEyeDetector(blink_threshold=args.threshold, 
                                   blink_consecutive_frames=args.frames)
        detector.run_camera()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()