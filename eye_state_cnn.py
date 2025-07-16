#!/usr/bin/env python3
"""
CNN-Based Eye State Detector
============================
A deep learning approach for more accurate eye open/closed detection.
Uses a lightweight CNN model for real-time performance.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from pathlib import Path

class EyeStateCNN(nn.Module):
    """Lightweight CNN for eye state classification"""
    
    def __init__(self, input_size=(24, 24)):
        super(EyeStateCNN, self).__init__()
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes: open, closed
        
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_size)
            dummy_output = self._forward_conv(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers only"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x
    
    def forward(self, x):
        """Forward pass through the entire network"""
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

class CNNEyeStateDetector:
    """CNN-based eye state detector with real-time optimization"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = EyeStateCNN()
        self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((24, 24)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"✅ Loaded CNN model from {model_path}")
        else:
            print("⚠️  No pre-trained model found. Using untrained model.")
            print("   Consider training the model for better accuracy.")
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def save_model(self, model_path):
        """Save the current model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_architecture': 'EyeStateCNN',
                'input_size': self.model.input_size
            }, model_path)
            print(f"✅ Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def preprocess_eye_region(self, eye_region):
        """Preprocess eye region for CNN input"""
        if eye_region is None or eye_region.size == 0:
            return None
        
        try:
            # Ensure the eye region is properly sized
            if len(eye_region.shape) == 3:
                eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            eye_region = cv2.equalizeHist(eye_region)
            
            # Apply preprocessing transforms
            eye_tensor = self.transform(eye_region)
            eye_tensor = eye_tensor.unsqueeze(0)  # Add batch dimension
            
            return eye_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing eye region: {e}")
            return None
    
    def predict_eye_state(self, eye_region):
        """Predict eye state using CNN"""
        preprocessed = self.preprocess_eye_region(eye_region)
        
        if preprocessed is None:
            return 0.5, "UNKNOWN"  # Return neutral probability
        
        try:
            with torch.no_grad():
                outputs = self.model(preprocessed)
                probabilities = outputs.cpu().numpy()[0]
                
                # probabilities[0] = closed, probabilities[1] = open
                open_prob = probabilities[1]
                closed_prob = probabilities[0]
                
                state = "OPEN" if open_prob > closed_prob else "CLOSED"
                confidence = max(open_prob, closed_prob)
                
                return confidence, state
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return 0.5, "UNKNOWN"
    
    def extract_eye_region(self, frame, eye_landmarks, padding=10):
        """Extract eye region from frame using landmarks"""
        try:
            if len(eye_landmarks) < 4:
                return None
            
            # Convert normalized coordinates to pixel coordinates if needed
            h, w = frame.shape[:2]
            if eye_landmarks.max() <= 1.0:
                eye_landmarks = eye_landmarks * [w, h]
            
            # Get bounding box
            x_min = max(0, int(eye_landmarks[:, 0].min()) - padding)
            x_max = min(w, int(eye_landmarks[:, 0].max()) + padding)
            y_min = max(0, int(eye_landmarks[:, 1].min()) - padding)
            y_max = min(h, int(eye_landmarks[:, 1].max()) + padding)
            
            # Extract eye region
            eye_region = frame[y_min:y_max, x_min:x_max]
            
            return eye_region if eye_region.size > 0 else None
            
        except Exception as e:
            print(f"❌ Error extracting eye region: {e}")
            return None

class HybridEyeStateDetector:
    """Hybrid detector combining CNN and traditional EAR methods"""
    
    def __init__(self, cnn_model_path=None, cnn_weight=0.7, ear_weight=0.3):
        self.cnn_detector = CNNEyeStateDetector(cnn_model_path)
        self.cnn_weight = cnn_weight
        self.ear_weight = ear_weight
        
        # EAR threshold for traditional method
        self.ear_threshold = 0.25
        
        print(f"🔀 Hybrid detector initialized (CNN: {cnn_weight}, EAR: {ear_weight})")
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if h == 0:
            return 0.0
        
        return (v1 + v2) / (2.0 * h)
    
    def detect_eye_state(self, frame, left_eye_landmarks, right_eye_landmarks):
        """Detect eye state using hybrid approach"""
        # Extract eye regions
        left_eye_region = self.cnn_detector.extract_eye_region(frame, left_eye_landmarks)
        right_eye_region = self.cnn_detector.extract_eye_region(frame, right_eye_landmarks)
        
        # CNN predictions
        left_cnn_conf, left_cnn_state = self.cnn_detector.predict_eye_state(left_eye_region)
        right_cnn_conf, right_cnn_state = self.cnn_detector.predict_eye_state(right_eye_region)
        
        # EAR calculations
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2
        
        # EAR-based prediction
        ear_state = "OPEN" if avg_ear > self.ear_threshold else "CLOSED"
        ear_confidence = abs(avg_ear - self.ear_threshold) * 4  # Normalize to 0-1 range
        
        # Combine predictions
        cnn_open_votes = sum([1 for state in [left_cnn_state, right_cnn_state] if state == "OPEN"])
        cnn_confidence = (left_cnn_conf + right_cnn_conf) / 2
        
        # Weighted combination
        if cnn_confidence > 0.7:  # High confidence CNN prediction
            final_state = left_cnn_state if left_cnn_conf > right_cnn_conf else right_cnn_state
            final_confidence = cnn_confidence
        else:  # Use hybrid approach
            cnn_score = cnn_open_votes / 2.0  # 0-1 scale
            ear_score = 1.0 if ear_state == "OPEN" else 0.0
            
            combined_score = (self.cnn_weight * cnn_score + 
                            self.ear_weight * ear_score)
            
            final_state = "OPEN" if combined_score > 0.5 else "CLOSED"
            final_confidence = abs(combined_score - 0.5) * 2  # Normalize confidence
        
        return {
            'state': final_state,
            'confidence': final_confidence,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'cnn_state': f"L:{left_cnn_state}, R:{right_cnn_state}",
            'cnn_confidence': cnn_confidence
        }

def create_sample_model():
    """Create and save a sample model structure"""
    model = EyeStateCNN()
    
    # Initialize with reasonable random weights
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    # Save the model
    model_path = "eye_state_cnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'EyeStateCNN',
        'input_size': (24, 24)
    }, model_path)
    
    print(f"📦 Sample CNN model created: {model_path}")
    print("   Note: This is an untrained model. For best results, train with your own data.")
    
    return model_path

def main():
    """Test the CNN eye state detector"""
    print("🧠 CNN Eye State Detector Test")
    print("=" * 40)
    
    # Create sample model if it doesn't exist
    model_path = "eye_state_cnn_model.pth"
    if not os.path.exists(model_path):
        model_path = create_sample_model()
    
    # Initialize detector
    detector = CNNEyeStateDetector(model_path)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("📷 Camera test started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simple test: use center region as "eye"
        h, w = frame.shape[:2]
        test_region = frame[h//3:2*h//3, w//3:2*w//3]
        
        confidence, state = detector.predict_eye_state(test_region)
        
        # Display results
        cv2.putText(frame, f"State: {state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), (255, 0, 0), 2)
        
        cv2.imshow('CNN Eye State Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()