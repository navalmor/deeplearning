#!/usr/bin/env python3
"""
Setup script for Eye Detection Application
This script will download the required facial landmark predictor file
and install necessary dependencies.
"""

import os
import sys
import subprocess
import urllib.request
import bz2
import shutil

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def download_landmarks_file():
    """Download and extract the facial landmarks predictor file"""
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
    extracted_file = "shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(extracted_file):
        print(f"✓ {extracted_file} already exists!")
        return True
    
    print("Downloading facial landmarks predictor file...")
    print(f"URL: {url}")
    
    try:
        # Download the compressed file
        print("Downloading... (this may take a few minutes)")
        urllib.request.urlretrieve(url, compressed_file)
        print("✓ Download completed!")
        
        # Extract the file
        print("Extracting...")
        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(extracted_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the compressed file
        os.remove(compressed_file)
        print("✓ Extraction completed!")
        print(f"✓ {extracted_file} is ready!")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading/extracting file: {str(e)}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Check if pip is available
    success, _ = run_command("pip --version")
    if not success:
        print("✗ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    success, output = run_command("pip install -r requirements.txt")
    if success:
        print("✓ Dependencies installed successfully!")
        return True
    else:
        print("✗ Error installing dependencies:")
        print(output)
        return False

def check_camera():
    """Check if camera is available"""
    print("Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera is working!")
                return True
            else:
                print("⚠ Camera detected but cannot read frames")
                return False
        else:
            print("⚠ No camera detected or camera is in use")
            return False
    except ImportError:
        print("⚠ OpenCV not installed yet, will check camera after installation")
        return True

def main():
    print("=" * 60)
    print("EYE DETECTION APPLICATION SETUP")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required!")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed!")
        sys.exit(1)
    
    # Download landmarks file
    if not download_landmarks_file():
        print("Setup failed!")
        sys.exit(1)
    
    # Check camera (after installing OpenCV)
    check_camera()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("You can now run the eye detection application:")
    print("  python eye_detector.py")
    print("\nOptions:")
    print("  python eye_detector.py --threshold 0.25  # Adjust sensitivity")
    print("  python eye_detector.py --frames 5        # Adjust blink detection")
    print("=" * 60)

if __name__ == "__main__":
    main()