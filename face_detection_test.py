# face_detection_test.py
# Use this script to test your face detection setup

import cv2
import face_recognition
import sys
import os
from pathlib import Path
import numpy as np


def test_image_loading(image_path):
    """Test if an image can be loaded properly"""
    print(f"\n--- Testing image loading for: {image_path} ---")

    # Method 1: OpenCV imread
    img = cv2.imread(str(image_path))
    if img is None or img.size == 0:
        print("❌ OpenCV imread failed to load the image")
    else:
        print(f"✅ OpenCV imread loaded the image successfully: {img.shape}")

    # Method 2: Alternative loading with imdecode
    try:
        with open(image_path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img2 is None or img2.size == 0:
            print("❌ OpenCV imdecode failed to load the image")
        else:
            print(f"✅ OpenCV imdecode loaded the image successfully: {img2.shape}")
    except Exception as e:
        print(f"❌ Error with alternative loading method: {e}")

    return img is not None or (img2 is not None)


def test_face_detection(image_path):
    """Test face detection on an image"""
    print(f"\n--- Testing face detection for: {image_path} ---")

    # Try to load the image
    try:
        with open(image_path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            print("❌ Failed to load image for face detection")
            return False

        # Convert to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Test 1: HOG face detection (default method)
        print("\n1. Testing HOG face detection (faster but less accurate):")
        start = cv2.getTickCount()
        hog_face_locations = face_recognition.face_locations(rgb_image, model="hog", number_of_times_to_upsample=1)
        end = cv2.getTickCount()
        time_hog = (end - start) / cv2.getTickFrequency()
        print(f"   Found {len(hog_face_locations)} faces in {time_hog:.2f} seconds")

        # Test 2: HOG with increased sensitivity
        print("\n2. Testing HOG with increased sensitivity (upsample=2):")
        start = cv2.getTickCount()
        hog_sensitive_face_locations = face_recognition.face_locations(rgb_image, model="hog",
                                                                       number_of_times_to_upsample=2)
        end = cv2.getTickCount()
        time_hog_sensitive = (end - start) / cv2.getTickFrequency()
        print(f"   Found {len(hog_sensitive_face_locations)} faces in {time_hog_sensitive:.2f} seconds")

        # Test 3: Try CNN model if available (more accurate but slower)
        try:
            print("\n3. Testing CNN face detection (more accurate but slower):")
            start = cv2.getTickCount()
            cnn_face_locations = face_recognition.face_locations(rgb_image, model="cnn", number_of_times_to_upsample=1)
            end = cv2.getTickCount()
            time_cnn = (end - start) / cv2.getTickFrequency()
            print(f"   Found {len(cnn_face_locations)} faces in {time_cnn:.2f} seconds")
        except Exception as e:
            print(f"   ❌ CNN detection failed (likely not installed): {e}")

        # Test 4: OpenCV Haar Cascade
        print("\n4. Testing OpenCV Haar Cascade:")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try to load the face cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                print("   ❌ Failed to load Haar Cascade classifier")
            else:
                start = cv2.getTickCount()
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                end = cv2.getTickCount()
                time_cascade = (end - start) / cv2.getTickFrequency()
                print(f"   Found {len(faces)} faces in {time_cascade:.2f} seconds")

                # Create debug image with rectangles
                debug_img = image.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                debug_path = f"{image_path.stem}_debug.jpg"
                cv2.imwrite(debug_path, debug_img)
                print(f"   ✅ Saved debug image with face rectangles to {debug_path}")

        except Exception as e:
            print(f"   ❌ Error with OpenCV Haar Cascade: {e}")

        # Summary
        print("\n--- Summary ---")
        if len(hog_face_locations) > 0 or len(hog_sensitive_face_locations) > 0:
            print("✅ HOG detection found faces")
        else:
            print("❌ HOG detection found no faces")

        try:
            if 'cnn_face_locations' in locals() and len(cnn_face_locations) > 0:
                print("✅ CNN detection found faces")
            else:
                print("❌ CNN detection found no faces")
        except:
            pass

        if 'faces' in locals() and len(faces) > 0:
            print("✅ OpenCV Haar Cascade found faces")
        else:
            print("❌ OpenCV Haar Cascade found no faces")

        return True

    except Exception as e:
        print(f"❌ Error during face detection: {e}")
        return False


def main():
    print("Face Detection Diagnostic Tool")
    print("=============================")

    if len(sys.argv) > 1:
        # Test a specific image
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            print(f"Error: Image {image_path} does not exist")
            return

        loaded = test_image_loading(image_path)
        if loaded:
            test_face_detection(image_path)
    else:
        # Test with a sample directory
        print("No image specified. Searching for images in static/uploads/")
        upload_dir = Path("static/uploads")

        if not upload_dir.exists():
            print(f"Error: Directory {upload_dir} does not exist")
            return

        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(upload_dir.glob(ext))

        if not image_paths:
            print("No images found in upload directory")
            return

        print(f"Found {len(image_paths)} images")
        for img_path in image_paths:
            loaded = test_image_loading(img_path)
            if loaded:
                test_face_detection(img_path)


if __name__ == "__main__":
    main()