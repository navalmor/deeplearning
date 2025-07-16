#!/usr/bin/env python3
"""
Advanced Eye-Controlled Mouse System
=====================================
Features:
- Real-time gaze direction detection
- Mouse cursor control via eye movement
- Blink-to-click functionality
- Enhanced accuracy with MediaPipe
- Visual feedback overlay
- Optimized for 20+ FPS performance
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math
import argparse
from collections import deque
from typing import Tuple, Optional, List
import threading
import queue

# Disable pyautogui failsafe for smooth operation
pyautogui.FAILSAFE = False

class EyeGazeController:
    def __init__(self, config):
        """Initialize the Eye-Controlled Mouse System"""
        self.config = config
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh with optimized parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Smoothing and filtering
        self.gaze_history = deque(maxlen=config['smoothing_frames'])
        self.blink_history = deque(maxlen=config['blink_frames'])
        
        # State tracking
        self.last_blink_time = 0
        self.click_pending = False
        self.calibration_points = []
        self.is_calibrated = False
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.fps = 0
        
        # Visual feedback
        self.cursor_pos = (self.screen_w // 2, self.screen_h // 2)
        self.gaze_direction = "CENTER"
        self.eye_state = "OPEN"
        
        # Threading for mouse control
        self.mouse_queue = queue.Queue()
        self.mouse_thread = threading.Thread(target=self._mouse_worker, daemon=True)
        self.mouse_thread.start()
        
        print("🎯 Advanced Eye-Controlled Mouse System initialized!")
        print(f"📺 Screen resolution: {self.screen_w}x{self.screen_h}")
    
    def _mouse_worker(self):
        """Background thread for smooth mouse movement"""
        while True:
            try:
                action, data = self.mouse_queue.get(timeout=0.1)
                if action == "move":
                    x, y = data
                    pyautogui.moveTo(x, y)
                elif action == "click":
                    pyautogui.click()
                self.mouse_queue.task_done()
            except queue.Empty:
                continue
    
    def extract_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmarks from MediaPipe results"""
        eye_points = []
        for idx in eye_indices:
            point = landmarks[idx]
            eye_points.append([point.x, point.y, point.z])
        return np.array(eye_points)
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio with improved accuracy"""
        # Get the eye landmarks
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate EAR
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_gaze_direction(self, left_eye, right_eye, iris_left, iris_right):
        """Detect gaze direction using iris position relative to eye corners"""
        try:
            # Calculate iris position relative to eye corners
            left_ratio = self._calculate_iris_ratio(left_eye, iris_left)
            right_ratio = self._calculate_iris_ratio(right_eye, iris_right)
            
            # Average the ratios for more stability
            horizontal_ratio = (left_ratio[0] + right_ratio[0]) / 2
            vertical_ratio = (left_ratio[1] + right_ratio[1]) / 2
            
            # Determine gaze direction
            direction = self._ratio_to_direction(horizontal_ratio, vertical_ratio)
            
            return direction, horizontal_ratio, vertical_ratio
            
        except Exception as e:
            return "CENTER", 0.5, 0.5
    
    def _calculate_iris_ratio(self, eye_landmarks, iris_landmarks):
        """Calculate iris position ratio within the eye"""
        if len(eye_landmarks) < 6 or len(iris_landmarks) < 5:
            return (0.5, 0.5)
        
        # Eye corners
        left_corner = eye_landmarks[0]
        right_corner = eye_landmarks[3]
        top_point = eye_landmarks[1]
        bottom_point = eye_landmarks[4]
        
        # Iris center
        iris_center = np.mean(iris_landmarks, axis=0)
        
        # Calculate ratios
        eye_width = np.linalg.norm(right_corner - left_corner)
        eye_height = np.linalg.norm(top_point - bottom_point)
        
        if eye_width == 0 or eye_height == 0:
            return (0.5, 0.5)
        
        # Horizontal ratio (0 = left, 1 = right)
        h_ratio = np.linalg.norm(iris_center - left_corner) / eye_width
        
        # Vertical ratio (0 = up, 1 = down)
        v_ratio = np.linalg.norm(iris_center - top_point) / eye_height
        
        return (h_ratio, v_ratio)
    
    def _ratio_to_direction(self, h_ratio, v_ratio):
        """Convert ratios to gaze direction"""
        # Thresholds for direction detection
        h_left_thresh = 0.35
        h_right_thresh = 0.65
        v_up_thresh = 0.35
        v_down_thresh = 0.65
        
        # Determine direction
        if h_ratio < h_left_thresh:
            if v_ratio < v_up_thresh:
                return "UP_LEFT"
            elif v_ratio > v_down_thresh:
                return "DOWN_LEFT"
            else:
                return "LEFT"
        elif h_ratio > h_right_thresh:
            if v_ratio < v_up_thresh:
                return "UP_RIGHT"
            elif v_ratio > v_down_thresh:
                return "DOWN_RIGHT"
            else:
                return "RIGHT"
        else:
            if v_ratio < v_up_thresh:
                return "UP"
            elif v_ratio > v_down_thresh:
                return "DOWN"
            else:
                return "CENTER"
    
    def gaze_to_screen_coordinates(self, h_ratio, v_ratio):
        """Convert gaze ratios to screen coordinates"""
        # Apply sensitivity and smoothing
        sensitivity = self.config['mouse_sensitivity']
        
        # Map ratios to screen coordinates
        x = int(h_ratio * self.screen_w * sensitivity)
        y = int(v_ratio * self.screen_h * sensitivity)
        
        # Clamp to screen bounds
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))
        
        return x, y
    
    def smooth_gaze_position(self, new_pos):
        """Apply smoothing to gaze position"""
        self.gaze_history.append(new_pos)
        
        if len(self.gaze_history) < 3:
            return new_pos
        
        # Weighted average with more weight on recent positions
        weights = np.linspace(0.5, 1.0, len(self.gaze_history))
        weights /= weights.sum()
        
        smoothed_x = sum(pos[0] * w for pos, w in zip(self.gaze_history, weights))
        smoothed_y = sum(pos[1] * w for pos, w in zip(self.gaze_history, weights))
        
        return (int(smoothed_x), int(smoothed_y))
    
    def detect_blink_click(self, left_ear, right_ear):
        """Detect blink patterns for mouse clicks"""
        avg_ear = (left_ear + right_ear) / 2
        self.blink_history.append(avg_ear)
        
        if len(self.blink_history) < self.config['blink_frames']:
            return False
        
        # Check for blink pattern
        recent_avg = np.mean(list(self.blink_history)[-5:])
        baseline_avg = np.mean(list(self.blink_history))
        
        # Detect significant drop in EAR (blink)
        if recent_avg < self.config['blink_threshold'] and baseline_avg > self.config['blink_threshold']:
            current_time = time.time()
            if current_time - self.last_blink_time > self.config['click_cooldown']:
                self.last_blink_time = current_time
                return True
        
        return False
    
    def draw_visual_feedback(self, frame, gaze_pos, direction, eye_state, fps):
        """Draw visual feedback overlay"""
        h, w = frame.shape[:2]
        
        # Draw crosshair at gaze position
        gaze_x = int(gaze_pos[0] * w / self.screen_w)
        gaze_y = int(gaze_pos[1] * h / self.screen_h)
        
        # Crosshair
        cv2.line(frame, (gaze_x - 20, gaze_y), (gaze_x + 20, gaze_y), (0, 255, 0), 2)
        cv2.line(frame, (gaze_x, gaze_y - 20), (gaze_x, gaze_y + 20), (0, 255, 0), 2)
        cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 255, 0), -1)
        
        # Status information
        status_y = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_y += 30
        cv2.putText(frame, f"Gaze: {direction}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_y += 30
        color = (0, 255, 0) if eye_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Eyes: {eye_state}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        status_y += 30
        cv2.putText(frame, f"Mouse: ({gaze_pos[0]}, {gaze_pos[1]})", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Controls:",
            "ESC: Exit",
            "SPACE: Toggle mouse control",
            "C: Calibrate",
            "R: Reset"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blink indicator
        if self.click_pending:
            cv2.circle(frame, (w - 50, 50), 20, (0, 255, 255), -1)
            cv2.putText(frame, "CLICK", (w - 80, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def process_frame(self, frame):
        """Process a single frame for eye tracking and mouse control"""
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        gaze_pos = self.cursor_pos
        direction = "CENTER"
        eye_state = "OPEN"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                h, w = frame.shape[:2]
                
                # Convert normalized coordinates to pixel coordinates
                points = []
                for landmark in landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append([x, y])
                points = np.array(points)
                
                # Eye landmark indices for MediaPipe
                LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
                RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
                LEFT_IRIS_INDICES = [474, 475, 476, 477]
                RIGHT_IRIS_INDICES = [469, 470, 471, 472]
                
                # Extract eye landmarks
                left_eye = self.extract_eye_landmarks(landmarks, LEFT_EYE_INDICES)
                right_eye = self.extract_eye_landmarks(landmarks, RIGHT_EYE_INDICES)
                left_iris = self.extract_eye_landmarks(landmarks, LEFT_IRIS_INDICES)
                right_iris = self.extract_eye_landmarks(landmarks, RIGHT_IRIS_INDICES)
                
                # Calculate EAR for blink detection
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                
                # Determine eye state
                avg_ear = (left_ear + right_ear) / 2
                eye_state = "CLOSED" if avg_ear < self.config['ear_threshold'] else "OPEN"
                
                # Detect gaze direction
                direction, h_ratio, v_ratio = self.detect_gaze_direction(
                    left_eye, right_eye, left_iris, right_iris
                )
                
                # Convert to screen coordinates
                raw_pos = self.gaze_to_screen_coordinates(h_ratio, v_ratio)
                gaze_pos = self.smooth_gaze_position(raw_pos)
                
                # Update cursor position
                self.cursor_pos = gaze_pos
                self.gaze_direction = direction
                self.eye_state = eye_state
                
                # Mouse control
                if self.config['mouse_control_enabled']:
                    self.mouse_queue.put(("move", gaze_pos))
                
                # Blink click detection
                if self.detect_blink_click(left_ear, right_ear):
                    if self.config['click_enabled']:
                        self.mouse_queue.put(("click", None))
                        self.click_pending = True
                        # Reset click pending after short delay
                        threading.Timer(0.5, lambda: setattr(self, 'click_pending', False)).start()
                
                # Draw eye landmarks
                if self.config['show_landmarks']:
                    for point in points[LEFT_EYE_INDICES + RIGHT_EYE_INDICES]:
                        cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        
        # Calculate FPS
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Draw visual feedback
        self.draw_visual_feedback(frame, gaze_pos, direction, eye_state, self.fps)
        
        return frame
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        # Optimize camera settings for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🚀 Starting Eye-Controlled Mouse System...")
        print("📋 Controls:")
        print("  ESC: Exit")
        print("  SPACE: Toggle mouse control")
        print("  C: Calibrate")
        print("  R: Reset")
        print("  Blink to click!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Advanced Eye-Controlled Mouse', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    self.config['mouse_control_enabled'] = not self.config['mouse_control_enabled']
                    status = "enabled" if self.config['mouse_control_enabled'] else "disabled"
                    print(f"🖱️  Mouse control {status}")
                elif key == ord('c'):  # C
                    print("🎯 Calibration not implemented yet")
                elif key == ord('r'):  # R
                    self.gaze_history.clear()
                    self.blink_history.clear()
                    print("🔄 System reset")
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Eye-Controlled Mouse System stopped")

def load_config():
    """Load configuration parameters"""
    return {
        # Performance settings
        'smoothing_frames': 5,
        'blink_frames': 10,
        
        # Detection thresholds
        'ear_threshold': 0.25,
        'blink_threshold': 0.2,
        
        # Mouse control settings
        'mouse_sensitivity': 1.2,
        'mouse_control_enabled': True,
        'click_enabled': True,
        'click_cooldown': 1.0,  # seconds
        
        # Visual settings
        'show_landmarks': True,
        'show_crosshair': True,
    }

def main():
    parser = argparse.ArgumentParser(description='Advanced Eye-Controlled Mouse System')
    parser.add_argument('--no-mouse', action='store_true', 
                       help='Disable mouse control (visualization only)')
    parser.add_argument('--no-click', action='store_true', 
                       help='Disable blink-to-click functionality')
    parser.add_argument('--sensitivity', type=float, default=1.2,
                       help='Mouse sensitivity (default: 1.2)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Apply command line arguments
    if args.no_mouse:
        config['mouse_control_enabled'] = False
    if args.no_click:
        config['click_enabled'] = False
    config['mouse_sensitivity'] = args.sensitivity
    
    print("🎯 Advanced Eye-Controlled Mouse System")
    print("=" * 50)
    print(f"🖱️  Mouse control: {'ON' if config['mouse_control_enabled'] else 'OFF'}")
    print(f"👆 Click control: {'ON' if config['click_enabled'] else 'OFF'}")
    print(f"🎚️  Sensitivity: {config['mouse_sensitivity']}")
    print("=" * 50)
    
    try:
        controller = EyeGazeController(config)
        controller.run()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()