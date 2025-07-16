#!/usr/bin/env python3
"""
Calibration System for Eye-Controlled Mouse
===========================================
Provides calibration functionality to improve gaze tracking accuracy
by personalizing the system to individual users.
"""

import cv2
import numpy as np
import json
import time
from collections import defaultdict
import math

class GazeCalibrationSystem:
    """Calibration system for eye gaze tracking"""
    
    def __init__(self, screen_width, screen_height):
        self.screen_w = screen_width
        self.screen_h = screen_height
        
        # Calibration points (9-point calibration)
        self.calibration_points = [
            (0.1, 0.1),   # Top-left
            (0.5, 0.1),   # Top-center  
            (0.9, 0.1),   # Top-right
            (0.1, 0.5),   # Middle-left
            (0.5, 0.5),   # Center
            (0.9, 0.5),   # Middle-right
            (0.1, 0.9),   # Bottom-left
            (0.5, 0.9),   # Bottom-center
            (0.9, 0.9),   # Bottom-right
        ]
        
        # Convert to pixel coordinates
        self.pixel_points = [
            (int(x * screen_width), int(y * screen_height))
            for x, y in self.calibration_points
        ]
        
        # Calibration data storage
        self.calibration_data = defaultdict(list)
        self.current_point_index = 0
        self.samples_per_point = 30
        self.current_samples = 0
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_complete = False
        
        # Transformation matrix
        self.transform_matrix = None
        self.calibration_accuracy = 0.0
        
    def start_calibration(self):
        """Start the calibration process"""
        self.is_calibrating = True
        self.calibration_complete = False
        self.current_point_index = 0
        self.current_samples = 0
        self.calibration_data.clear()
        print("🎯 Starting calibration process...")
        print(f"📊 Will calibrate {len(self.calibration_points)} points with {self.samples_per_point} samples each")
    
    def get_current_calibration_point(self):
        """Get the current calibration point being displayed"""
        if not self.is_calibrating or self.current_point_index >= len(self.pixel_points):
            return None
        return self.pixel_points[self.current_point_index]
    
    def add_gaze_sample(self, gaze_x, gaze_y):
        """Add a gaze sample for the current calibration point"""
        if not self.is_calibrating:
            return False
        
        if self.current_point_index >= len(self.calibration_points):
            return False
        
        # Store the sample
        point_id = self.current_point_index
        self.calibration_data[point_id].append((gaze_x, gaze_y))
        self.current_samples += 1
        
        # Check if we have enough samples for this point
        if self.current_samples >= self.samples_per_point:
            print(f"✅ Point {point_id + 1}/{len(self.calibration_points)} completed")
            self.current_point_index += 1
            self.current_samples = 0
            
            # Check if calibration is complete
            if self.current_point_index >= len(self.calibration_points):
                self.complete_calibration()
                return True
        
        return False
    
    def complete_calibration(self):
        """Complete the calibration process and calculate transformation"""
        self.is_calibrating = False
        
        print("🔄 Processing calibration data...")
        
        # Average the samples for each point
        averaged_points = {}
        for point_id, samples in self.calibration_data.items():
            if len(samples) > 0:
                avg_x = np.mean([s[0] for s in samples])
                avg_y = np.mean([s[1] for s in samples])
                averaged_points[point_id] = (avg_x, avg_y)
        
        # Calculate transformation matrix
        if len(averaged_points) >= 4:  # Need at least 4 points for transformation
            self.calculate_transformation_matrix(averaged_points)
            self.calibration_complete = True
            print(f"✅ Calibration completed! Accuracy: {self.calibration_accuracy:.2f}%")
        else:
            print("❌ Calibration failed - insufficient data points")
            self.calibration_complete = False
    
    def calculate_transformation_matrix(self, averaged_points):
        """Calculate transformation matrix from gaze to screen coordinates"""
        try:
            # Prepare source and destination points
            src_points = []
            dst_points = []
            
            for point_id, gaze_pos in averaged_points.items():
                if point_id < len(self.pixel_points):
                    src_points.append(gaze_pos)
                    dst_points.append(self.pixel_points[point_id])
            
            src_points = np.float32(src_points)
            dst_points = np.float32(dst_points)
            
            # Calculate transformation matrix using perspective transform
            if len(src_points) >= 4:
                self.transform_matrix = cv2.getPerspectiveTransform(
                    src_points[:4], dst_points[:4]
                )
                
                # Calculate calibration accuracy
                self.calibration_accuracy = self.calculate_accuracy(
                    src_points, dst_points
                )
            else:
                # Use affine transform for fewer points
                self.transform_matrix = cv2.getAffineTransform(
                    src_points[:3], dst_points[:3]
                )
                self.calibration_accuracy = 75.0  # Estimated accuracy
                
        except Exception as e:
            print(f"❌ Error calculating transformation matrix: {e}")
            self.transform_matrix = None
            self.calibration_accuracy = 0.0
    
    def calculate_accuracy(self, src_points, dst_points):
        """Calculate calibration accuracy as percentage"""
        if self.transform_matrix is None:
            return 0.0
        
        try:
            # Transform source points using calculated matrix
            if self.transform_matrix.shape[0] == 3:  # Perspective transform
                transformed = cv2.perspectiveTransform(
                    src_points.reshape(-1, 1, 2), self.transform_matrix
                ).reshape(-1, 2)
            else:  # Affine transform
                ones = np.ones((src_points.shape[0], 1))
                src_homogeneous = np.hstack([src_points, ones])
                transformed = (self.transform_matrix @ src_homogeneous.T).T
            
            # Calculate average error
            errors = np.linalg.norm(transformed - dst_points, axis=1)
            avg_error = np.mean(errors)
            
            # Convert to accuracy percentage (lower error = higher accuracy)
            max_screen_distance = math.sqrt(self.screen_w**2 + self.screen_h**2)
            accuracy = max(0, 100 * (1 - avg_error / max_screen_distance))
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Error calculating accuracy: {e}")
            return 0.0
    
    def transform_gaze_point(self, gaze_x, gaze_y):
        """Transform raw gaze coordinates to calibrated screen coordinates"""
        if not self.calibration_complete or self.transform_matrix is None:
            return gaze_x, gaze_y
        
        try:
            src_point = np.float32([[gaze_x, gaze_y]])
            
            if self.transform_matrix.shape[0] == 3:  # Perspective transform
                transformed = cv2.perspectiveTransform(
                    src_point.reshape(-1, 1, 2), self.transform_matrix
                ).reshape(-1, 2)
            else:  # Affine transform
                src_homogeneous = np.array([gaze_x, gaze_y, 1])
                transformed = self.transform_matrix @ src_homogeneous
                transformed = transformed[:2]
            
            # Clamp to screen bounds
            x = max(0, min(self.screen_w - 1, int(transformed[0])))
            y = max(0, min(self.screen_h - 1, int(transformed[1])))
            
            return x, y
            
        except Exception as e:
            print(f"❌ Error transforming gaze point: {e}")
            return gaze_x, gaze_y
    
    def save_calibration(self, filename="calibration_data.json"):
        """Save calibration data to file"""
        if not self.calibration_complete:
            print("⚠️  No calibration data to save")
            return False
        
        try:
            calibration_data = {
                'screen_width': self.screen_w,
                'screen_height': self.screen_h,
                'calibration_points': self.calibration_points,
                'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None,
                'accuracy': self.calibration_accuracy,
                'samples_per_point': self.samples_per_point,
                'timestamp': time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"💾 Calibration saved to {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filename="calibration_data.json"):
        """Load calibration data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Verify screen dimensions match
            if (data.get('screen_width') != self.screen_w or 
                data.get('screen_height') != self.screen_h):
                print("⚠️  Screen dimensions don't match saved calibration")
                return False
            
            # Load transformation matrix
            if data.get('transform_matrix'):
                self.transform_matrix = np.array(data['transform_matrix'])
                self.calibration_accuracy = data.get('accuracy', 0.0)
                self.calibration_complete = True
                
                print(f"✅ Calibration loaded from {filename}")
                print(f"📊 Accuracy: {self.calibration_accuracy:.2f}%")
                return True
            else:
                print("❌ No valid transformation matrix in calibration file")
                return False
                
        except FileNotFoundError:
            print(f"⚠️  Calibration file {filename} not found")
            return False
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def draw_calibration_ui(self, frame):
        """Draw calibration UI on frame"""
        if not self.is_calibrating:
            return frame
        
        current_point = self.get_current_calibration_point()
        if current_point is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Scale coordinates to frame size
        point_x = int(current_point[0] * w / self.screen_w)
        point_y = int(current_point[1] * h / self.screen_h)
        
        # Draw calibration target
        radius = 30
        
        # Outer circle (white)
        cv2.circle(frame, (point_x, point_y), radius, (255, 255, 255), 3)
        
        # Inner circle (red)
        cv2.circle(frame, (point_x, point_y), radius // 3, (0, 0, 255), -1)
        
        # Center dot (white)
        cv2.circle(frame, (point_x, point_y), 3, (255, 255, 255), -1)
        
        # Progress indicator
        progress = (self.current_point_index * self.samples_per_point + self.current_samples)
        total_samples = len(self.calibration_points) * self.samples_per_point
        progress_percent = (progress / total_samples) * 100
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 60
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        fill_width = int((progress_percent / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Progress text
        progress_text = f"Calibration Progress: {progress_percent:.1f}%"
        cv2.putText(frame, progress_text, (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        point_text = f"Look at point {self.current_point_index + 1}/{len(self.calibration_points)}"
        sample_text = f"Sample {self.current_samples + 1}/{self.samples_per_point}"
        
        cv2.putText(frame, point_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, sample_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Look at the red dot and stay still", (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

class CalibrationValidation:
    """Validation system to test calibration accuracy"""
    
    def __init__(self, calibration_system):
        self.calibration_system = calibration_system
        self.validation_points = [
            (0.25, 0.25), (0.75, 0.25),
            (0.25, 0.75), (0.75, 0.75),
            (0.5, 0.5)
        ]
        self.validation_data = []
        self.current_validation_point = 0
        self.is_validating = False
    
    def start_validation(self):
        """Start validation process"""
        if not self.calibration_system.calibration_complete:
            print("❌ Cannot validate - calibration not complete")
            return False
        
        self.is_validating = True
        self.current_validation_point = 0
        self.validation_data.clear()
        print("🔍 Starting calibration validation...")
        return True
    
    def add_validation_sample(self, raw_gaze_x, raw_gaze_y):
        """Add validation sample"""
        if not self.is_validating:
            return False
        
        if self.current_validation_point >= len(self.validation_points):
            return False
        
        # Transform gaze point using calibration
        transformed_x, transformed_y = self.calibration_system.transform_gaze_point(
            raw_gaze_x, raw_gaze_y
        )
        
        # Expected point
        expected_x = self.validation_points[self.current_validation_point][0] * self.calibration_system.screen_w
        expected_y = self.validation_points[self.current_validation_point][1] * self.calibration_system.screen_h
        
        # Calculate error
        error = math.sqrt((transformed_x - expected_x)**2 + (transformed_y - expected_y)**2)
        
        self.validation_data.append({
            'expected': (expected_x, expected_y),
            'measured': (transformed_x, transformed_y),
            'error': error
        })
        
        self.current_validation_point += 1
        
        if self.current_validation_point >= len(self.validation_points):
            self.complete_validation()
            return True
        
        return False
    
    def complete_validation(self):
        """Complete validation and show results"""
        self.is_validating = False
        
        if not self.validation_data:
            print("❌ No validation data collected")
            return
        
        errors = [data['error'] for data in self.validation_data]
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"📊 Validation Results:")
        print(f"   Average error: {avg_error:.1f} pixels")
        print(f"   Maximum error: {max_error:.1f} pixels")
        
        # Determine accuracy rating
        if avg_error < 50:
            rating = "Excellent"
        elif avg_error < 100:
            rating = "Good"
        elif avg_error < 150:
            rating = "Fair"
        else:
            rating = "Poor - Consider recalibrating"
        
        print(f"   Rating: {rating}")

def main():
    """Test the calibration system"""
    print("🎯 Calibration System Test")
    print("=" * 40)
    
    # Initialize calibration system
    screen_w, screen_h = 1920, 1080  # Example screen resolution
    calibration = GazeCalibrationSystem(screen_w, screen_h)
    
    # Test calibration save/load
    print("Testing save/load functionality...")
    
    # Create dummy calibration data
    calibration.transform_matrix = np.eye(3, dtype=np.float32)
    calibration.calibration_accuracy = 85.5
    calibration.calibration_complete = True
    
    # Save and load
    if calibration.save_calibration("test_calibration.json"):
        calibration.calibration_complete = False
        if calibration.load_calibration("test_calibration.json"):
            print("✅ Save/load test passed")
        else:
            print("❌ Load test failed")
    else:
        print("❌ Save test failed")

if __name__ == "__main__":
    main()