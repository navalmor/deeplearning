# Eye Detection Application

A real-time eye detection application that can determine if your eyes are open or closed using computer vision techniques. The application uses your camera feed to detect faces, locate eyes, and analyze eye states in real-time.

## Features

- **Real-time Eye Detection**: Detects eyes in live camera feed
- **Open/Closed State Recognition**: Determines if eyes are open or closed
- **Blink Counter**: Counts the number of blinks detected
- **Eye Aspect Ratio (EAR) Display**: Shows the calculated EAR values for both eyes
- **Adjustable Sensitivity**: Modify detection threshold in real-time
- **Visual Feedback**: Draws eye landmarks and face boundaries
- **Mirror Mode**: Camera feed is flipped horizontally for natural interaction

## How It Works

The application uses the **Eye Aspect Ratio (EAR)** method for eye state detection:

1. **Face Detection**: Uses dlib's frontal face detector to locate faces
2. **Landmark Detection**: Identifies 68 facial landmarks including eye points
3. **EAR Calculation**: Computes the ratio of eye height to width
4. **State Determination**: Compares EAR against threshold to determine if eyes are open/closed
5. **Blink Detection**: Tracks consecutive closed frames to register blinks

### Eye Aspect Ratio Formula

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```

Where p1-p6 are the 6 landmark points around each eye.

## Installation

### Quick Setup

Run the setup script to automatically install dependencies and download required files:

```bash
python setup.py
```

### Manual Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Facial Landmarks Predictor**:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

## Requirements

- Python 3.7 or higher
- Webcam/Camera
- The following Python packages:
  - opencv-python
  - dlib
  - numpy
  - scipy
  - imutils

## Usage

### Basic Usage

```bash
python eye_detector.py
```

### Advanced Options

```bash
# Adjust eye closure sensitivity (lower = more sensitive)
python eye_detector.py --threshold 0.25

# Change consecutive frames needed for blink detection
python eye_detector.py --frames 5

# Combine options
python eye_detector.py --threshold 0.3 --frames 3
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `r` | Reset blink counter |
| `+` or `=` | Increase EAR threshold (less sensitive) |
| `-` | Decrease EAR threshold (more sensitive) |

## Display Information

The application shows the following information on screen:

- **Eye Status**: "Eyes: OPEN", "Eyes: CLOSED", or "Eyes: Not Detected"
- **Blink Count**: Total number of blinks detected
- **Left/Right EAR**: Eye Aspect Ratio values for each eye
- **Current Threshold**: The EAR threshold being used
- **Visual Landmarks**: Green circles and lines around detected eyes
- **Face Boundary**: Blue rectangle around detected face

## Troubleshooting

### Common Issues

1. **"shape_predictor_68_face_landmarks.dat not found"**
   - Run `python setup.py` or manually download the file
   - Ensure the file is in the same directory as `eye_detector.py`

2. **"Could not open camera"**
   - Check if camera is connected and not being used by another application
   - Try different camera indices if you have multiple cameras

3. **Poor Detection Accuracy**
   - Ensure good lighting conditions
   - Adjust the threshold using `+`/`-` keys
   - Face should be clearly visible and not too far from camera

4. **High False Positive Rate**
   - Increase threshold value (less sensitive)
   - Increase consecutive frames requirement

### Performance Tips

- Ensure good lighting for better face detection
- Keep face relatively centered in the frame
- Avoid extreme head angles
- Close other applications using the camera

## Technical Details

### Eye Aspect Ratio Thresholds

- **Default**: 0.3 (works for most people)
- **Range**: 0.1 - 0.5
- **Lower values**: More sensitive (detects partial closures)
- **Higher values**: Less sensitive (only full closures)

### Algorithm Flow

1. Capture frame from camera
2. Convert to grayscale
3. Detect faces using HOG + Linear SVM
4. For each face:
   - Predict 68 facial landmarks
   - Extract eye regions (points 36-47)
   - Calculate EAR for both eyes
   - Average the EAR values
   - Compare against threshold
   - Update eye state and blink counter

## Applications

This eye detection technology can be used for:

- **Drowsiness Detection**: Alert systems for drivers
- **Accessibility**: Eye-controlled interfaces
- **Health Monitoring**: Blink rate analysis
- **Gaming**: Eye-based game controls
- **Research**: Human-computer interaction studies

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## Acknowledgments

- [dlib](http://dlib.net/) for face detection and landmark prediction
- [OpenCV](https://opencv.org/) for computer vision operations
- The Eye Aspect Ratio method is based on research by Soukupová and Čech