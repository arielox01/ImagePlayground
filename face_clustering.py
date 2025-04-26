# face_clustering.py - Optimized for speed and memory efficiency on laptops

import os
import shutil
import face_recognition
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import sys
import time
import psutil  # For memory usage tracking

# Set the console encoding to UTF-8 if possible
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    return memory_mb


def estimate_memory_needs(image_shape, detection_model):
    """Estimate memory needs for face detection based on image size and model"""
    height, width = image_shape[:2]
    pixels = height * width

    # Base memory needed - rough estimate
    base_memory = pixels * 3 * 4 / (1024 * 1024)  # MB

    # Memory multiplier based on model
    if detection_model == "cnn":
        multiplier = 15  # CNN needs significantly more memory
    else:
        multiplier = 5  # HOG is more memory efficient

    return base_memory * multiplier


def resize_image_if_needed(image, max_size=800, force_resize=False):
    """
    Resize large images to speed up processing and reduce memory usage

    Args:
        image: Image to resize
        max_size: Maximum dimension size
        force_resize: If True, resize even if image is below max_size
    """
    if image is None:
        return None

    height, width = image.shape[:2]

    # If force resize or image is larger than max_size
    if force_resize or max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height))
        print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        return resized
    return image


def process_image(img_path, detection_model="hog", upsample_times=1, min_face_size=20,
                  low_memory_mode=False, status_callback=None):
    """
    Process a single image and extract faces with improved memory management

    Args:
        img_path: Path to the image
        detection_model: 'hog' (faster) or 'cnn' (more accurate)
        upsample_times: Number of times to upsample (1-3)
        min_face_size: Minimum face size in pixels
        low_memory_mode: Whether to use memory-saving techniques
        status_callback: Function to call with status updates
    """
    try:
        img_path = Path(img_path)
        if status_callback:
            status_callback(f"Processing image: {img_path.name}")
        print(f"Processing image: {img_path.name}")

        # Use the more robust image loading method that worked in the diagnostic test
        try:
            # Try alternative loading method with numpy first (for non-ASCII filenames)
            import numpy as np
            with open(img_path, 'rb') as f:
                img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None or image.size == 0:
                print(f"Alternative loading failed for {img_path}, trying standard method")
                # Fall back to standard loading as a backup
                image = cv2.imread(str(img_path))

            # If still failed, raise an exception
            if image is None or image.size == 0:
                raise Exception("Both loading methods failed")

        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            if status_callback:
                status_callback(f"Failed to load image {img_path.name}")
            return []

        print(f"Successfully loaded image: {img_path.name}, shape: {image.shape}")

        # Memory management: resize based on mode and model
        max_size = 600 if (low_memory_mode or detection_model == "cnn") else 1200
        image = resize_image_if_needed(image, max_size=max_size)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a copy of the original image for saving later
        original_image = image.copy()

        # Estimate memory needs and check if we need to adapt
        est_memory = estimate_memory_needs(image.shape, detection_model)
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        current_memory = get_memory_usage()

        print(
            f"Memory - Current: {current_memory:.1f}MB, Est. needed: {est_memory:.1f}MB, Available: {available_memory:.1f}MB")

        # If low memory mode or we don't have enough memory for CNN, fallback to HOG
        if detection_model == "cnn" and (low_memory_mode or est_memory > available_memory * 0.7):
            if status_callback:
                status_callback(f"Low memory - switching to HOG detection for {img_path.name}")
            print(f"Low memory - switching to HOG detection for {img_path.name}")
            detection_model = "hog"
            upsample_times = min(upsample_times, 1)  # Limit upsampling in low memory

        # Use specified face detection model with specified upsampling
        print(f"Using {detection_model} detection with upsample_times={upsample_times}")
        if status_callback:
            status_callback(f"Detecting faces with {detection_model}...")

        try:
            face_locations = face_recognition.face_locations(
                image,
                model=detection_model,
                number_of_times_to_upsample=upsample_times
            )
        except RuntimeError as e:
            # Handle memory errors or other runtime errors
            if "bad allocation" in str(e) or "allocation of" in str(e):
                print(f"Memory error during detection: {e}")
                if status_callback:
                    status_callback("Memory error - trying with reduced settings...")

                # Further resize the image and try again with HOG
                image = resize_image_if_needed(image, max_size=400, force_resize=True)
                detection_model = "hog"
                upsample_times = 1

                face_locations = face_recognition.face_locations(
                    image,
                    model=detection_model,
                    number_of_times_to_upsample=upsample_times
                )
            else:
                raise

        print(f"{detection_model.upper()} detection found {len(face_locations)} faces in {img_path.name}")
        if status_callback:
            status_callback(f"Found {len(face_locations)} faces in {img_path.name}")

        # If no faces found with primary method and it's not already CNN, try CNN as backup
        # Only if we're not in low memory mode and we have enough memory
        if len(face_locations) == 0 and detection_model != "cnn" and not low_memory_mode:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory > est_memory * 3:  # Only try CNN if we have plenty of memory
                print(f"Trying CNN detector for {img_path.name}")
                if status_callback:
                    status_callback(f"Trying CNN detector as backup...")
                try:
                    face_locations = face_recognition.face_locations(
                        image,
                        model="cnn",  # More accurate but slower
                        number_of_times_to_upsample=1
                    )
                    print(f"CNN detection found {len(face_locations)} faces in {img_path.name}")
                    if status_callback:
                        status_callback(f"CNN found {len(face_locations)} faces in {img_path.name}")
                except Exception as e:
                    print(f"CNN detection failed: {e}")
                    if status_callback:
                        status_callback("CNN detection failed, using HOG results")

        # Get face encodings for found locations
        face_encodings = []
        if face_locations:
            if status_callback:
                status_callback("Generating face encodings...")
            try:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if len(face_encodings) < len(face_locations):
                    print(f"Warning: Only encoded {len(face_encodings)} of {len(face_locations)} faces")
            except RuntimeError as e:
                print(f"Error during encoding: {e}")
                if "bad allocation" in str(e):
                    # If encoding fails, try with further reduced image
                    if status_callback:
                        status_callback("Memory error during encoding - reducing image size...")
                    image = resize_image_if_needed(image, max_size=300, force_resize=True)
                    face_locations = face_recognition.face_locations(
                        image, model="hog", number_of_times_to_upsample=1
                    )
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(image, face_locations)

        results = []
        for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
            top, right, bottom, left = location
            face_image = image[top:bottom, left:right]

            # Apply minimum face size filter based on user parameter
            face_width = right - left
            face_height = bottom - top
            if face_width < min_face_size or face_height < min_face_size:
                print(f"Face too small ({face_width}x{face_height}), skipping")
                continue

            # Quality score based on size and position
            quality_score = (face_width * face_height) / (image.shape[0] * image.shape[1])

            # Improve score for centered faces
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            img_center_x = image.shape[1] / 2
            img_center_y = image.shape[0] / 2

            # Distance from center (normalized)
            center_dist = np.sqrt(((center_x - img_center_x) / image.shape[1]) ** 2 +
                                  ((center_y - img_center_y) / image.shape[0]) ** 2)

            # Higher score for more centered faces
            center_bonus = 1 - min(center_dist, 0.5)
            quality_score *= (1 + center_bonus)

            results.append({
                'path': str(img_path),
                'encoding': encoding,
                'location': location,
                'face_image': face_image,
                'original_image': original_image,  # Store original image
                'quality': quality_score,
                'index': i
            })

        # If no faces were found, print message
        if not results:
            print(f"No valid faces found in {img_path.name}")
            if status_callback:
                status_callback(f"No valid faces found in {img_path.name}")

        return results
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        if status_callback:
            status_callback(f"Error processing {img_path.name}: {str(e)}")
        return []


def simple_clustering(encodings, tolerance=0.6):
    """Simple and fast clustering algorithm for face encodings"""
    if len(encodings) == 0:
        return []

    # Initialize with first face in its own cluster
    labels = [0]

    # For each remaining face, compare to existing clusters
    for i in range(1, len(encodings)):
        matched = False

        # Compare with faces from each existing cluster
        for j in range(i):
            # Calculate face distance
            face_distance = face_recognition.face_distance([encodings[j]], encodings[i])[0]

            if face_distance < tolerance:
                # Match found, assign to same cluster
                labels.append(labels[j])
                matched = True
                break

        if not matched:
            # No match found, create new cluster
            labels.append(max(labels) + 1)

    return labels


def cluster_faces(source_dir="static/uploads", output_root="categorized_runs", tolerance=0.6,
                  min_cluster_size=1, save_mode="both", detection_model="hog",
                  upsample_times=1, min_face_size=20, low_memory_mode=False,
                  batch_size=5, status_callback=None):
    """
    Clustering function with improved memory management and progress reporting

    Parameters:
    - save_mode: "both", "faces", or "originals" - what to save in each cluster
    - detection_model: "hog" (faster) or "cnn" (more accurate)
    - upsample_times: How many times to upsample image (1-3), higher finds smaller faces but slower
    - min_face_size: Minimum face size in pixels to consider valid
    - low_memory_mode: Whether to use memory-saving techniques
    - batch_size: Number of images to process at once (smaller batches use less memory)
    - status_callback: Function to call with status updates
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = Path(output_root) / timestamp
    original_dir = run_path / "OriginalImages"
    no_faces_dir = run_path / "No_Faces_Detected"

    # Create directories
    for directory in [run_path, original_dir, no_faces_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Get image paths
    source_path = Path(source_dir)
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(source_path.glob(ext))

    print(f"Found {len(image_paths)} images in upload directory")
    if status_callback:
        status_callback(f"Found {len(image_paths)} images in upload directory")

    if not image_paths:
        print("No images found in upload directory")
        if status_callback:
            status_callback("No images found in upload directory")
        return str(run_path)

    # Copy original images to original directory
    for img_path in image_paths:
        try:
            shutil.copy(img_path, original_dir / img_path.name)
        except Exception as e:
            print(f"Error copying {img_path.name}: {e}")
            # Create a simpler filename if needed
            safe_name = f"img_{hash(str(img_path))}.jpg"
            shutil.copy(img_path, original_dir / safe_name)

    print(f"Processing {len(image_paths)} images...")
    if status_callback:
        status_callback(f"Processing {len(image_paths)} images...")

    # Process images in batches to manage memory
    all_faces = []
    images_with_no_faces = set()

    # Adjust batch size based on memory mode and detection model
    if low_memory_mode:
        batch_size = min(batch_size, 3)  # Very small batches in low memory mode
    elif detection_model == "cnn":
        batch_size = min(batch_size, 5)  # Smaller batches for CNN model

    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch = image_paths[batch_start:batch_end]

        print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} images)")
        if status_callback:
            status_callback(f"Processing batch {batch_idx + 1}/{total_batches}")

        # Process each image in the batch
        for img_idx, img_path in enumerate(batch):
            if status_callback:
                overall_progress = (batch_idx * batch_size + img_idx) / len(image_paths) * 100
                status_callback(f"Overall progress: {overall_progress:.1f}% - Image {img_idx + 1}/{len(batch)}")

            faces = process_image(
                str(img_path),
                detection_model=detection_model,
                upsample_times=upsample_times,
                min_face_size=min_face_size,
                low_memory_mode=low_memory_mode,
                status_callback=status_callback
            )

            if faces:
                all_faces.extend(faces)
            else:
                images_with_no_faces.add(str(img_path))
                print(f"No faces found in {img_path.name}, adding to No_Faces_Detected folder")
                try:
                    # Copy to No_Faces_Detected folder
                    shutil.copy(img_path, no_faces_dir / img_path.name)
                except Exception as e:
                    print(f"Error copying to no_faces_dir: {e}")
                    # Create a simpler filename if needed
                    safe_name = f"noface_{hash(str(img_path))}.jpg"
                    shutil.copy(img_path, no_faces_dir / safe_name)

        # Force garbage collection after each batch to free memory
        import gc
        gc.collect()
        print(f"Current memory usage: {get_memory_usage():.1f} MB")

    total_faces = len(all_faces)
    print(f"Found {total_faces} faces total across {len(image_paths) - len(images_with_no_faces)} images")
    if status_callback:
        status_callback(f"Found {total_faces} faces total across {len(image_paths) - len(images_with_no_faces)} images")

    # If no faces found in any images, we're done
    if not all_faces:
        print("No faces found in any images")
        if status_callback:
            status_callback("No faces found in any images")
        return str(run_path)

    # Get all encodings
    encodings = [face['encoding'] for face in all_faces]

    print(f"Clustering {len(encodings)} face encodings...")
    if status_callback:
        status_callback(f"Clustering {len(encodings)} face encodings...")

    # Use simple clustering for small datasets
    cluster_labels = simple_clustering(encodings, tolerance)

    # Get unique clusters
    unique_labels = sorted(set(cluster_labels))
    cluster_count = len(unique_labels)
    print(f"Found {cluster_count} clusters")
    if status_callback:
        status_callback(f"Found {cluster_count} clusters")

    # Create clusters
    for label_idx, label in enumerate(unique_labels):
        if status_callback:
            cluster_progress = (label_idx + 1) / len(unique_labels) * 100
            status_callback(f"Saving clusters: {cluster_progress:.1f}% - Cluster {label_idx + 1}/{len(unique_labels)}")

        cluster_name = f"Face_{label_idx + 1}"
        cluster_folder = run_path / cluster_name
        cluster_folder.mkdir(exist_ok=True)

        # Get faces in this cluster
        cluster_faces = [face for face, face_label in zip(all_faces, cluster_labels) if face_label == label]
        print(f"Cluster {label_idx + 1} has {len(cluster_faces)} faces")

        # Sort by quality
        cluster_faces.sort(key=lambda x: x['quality'], reverse=True)

        # Save best face as preview
        if cluster_faces:
            preview_face = cluster_faces[0]
            preview_path = cluster_folder / "preview.jpg"
            preview_img = cv2.cvtColor(preview_face['face_image'], cv2.COLOR_RGB2BGR)
            # Use imdecode/imencode for saving to handle non-ASCII paths
            cv2.imwrite(str(preview_path), preview_img)

        # Track originals we've already saved (to avoid duplicates)
        originals_saved = set()

        # Save cropped faces and their originals based on save_mode
        for i, face in enumerate(cluster_faces):
            try:
                img_path = Path(face['path'])
                face_img = face['face_image']
                original_img = face['original_image']  # Get the original image

                # Create safe filenames using just the index and hash for files with non-ASCII names
                base_name = ''.join(c for c in img_path.stem if c.isalnum() or c in '._- ')
                if not base_name:
                    base_name = f"image_{hash(str(img_path))}"

                # Save based on selected mode
                if save_mode in ["both", "faces"]:
                    # Save the face image
                    filename = f"{base_name}_face{face['index'] + 1}.jpg"
                    dest_path = cluster_folder / filename
                    cv2.imwrite(str(dest_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

                    # Save reference to original
                    with open(str(cluster_folder / f"{filename}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Original: {img_path.name}\nLocation: {face['location']}")

                # Handle original images
                original_filename = f"original_{base_name}.jpg"
                if save_mode in ["both", "originals"] and original_filename not in originals_saved:
                    # Save the original image (if not already saved)
                    original_path = cluster_folder / original_filename
                    cv2.imwrite(str(original_path), cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
                    originals_saved.add(original_filename)

                    # Create a reference file for the original
                    with open(str(cluster_folder / f"{original_filename}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Source: {img_path.name}")

            except Exception as e:
                print(f"Error saving face: {e}")
                # Use a simpler filename
                simple_name = f"face_{i}.jpg"
                dest_path = cluster_folder / simple_name
                cv2.imwrite(str(dest_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    if status_callback:
        status_callback(f"Processing completed in {processing_time:.2f} seconds")

    return str(run_path)