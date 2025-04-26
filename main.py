# main.py - Optimized for laptops with memory management and progress tracking

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from pathlib import Path
import os
import shutil
import time
import sys
import threading
import queue
import cv2
import numpy as np
import io
from PIL import Image
import psutil

# Import the face clustering functionality
from face_clustering import cluster_faces, resize_image_if_needed

# Set UTF-8 encoding for Windows if possible
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

app = Flask(__name__)
app.secret_key = "face_clustering_app"  # For flash messages and session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'face_clustering_secret_key'

UPLOAD_FOLDER = Path("static/uploads")
FACES_DIR = Path("static/faces")
RUNS_DIR = Path("categorized_runs")
MAX_UPLOAD_SIZE_MB = 15  # Maximum size for uploaded images in MB

# Create necessary directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for tracking clustering progress
clustering_progress = {
    'status': 'idle',
    'message': '',
    'progress': 0,
    'started_at': None,
    'completed_at': None,
    'result': None,
    'error': None
}

# Queue for passing status updates between threads
status_queue = queue.Queue()


def status_callback(message):
    """Callback function for receiving status updates from clustering process"""
    status_queue.put(message)


def resize_image_for_upload(image_file, max_size=1600, max_file_size_mb=2):
    """
    Resize image during upload if it exceeds max dimensions or file size

    Args:
        image_file: File-like object from request.files
        max_size: Maximum width/height allowed
        max_file_size_mb: Maximum file size in MB

    Returns:
        BytesIO object containing the possibly resized image
    """
    img_data = image_file.read()
    file_size_mb = len(img_data) / (1024 * 1024)

    # Check if file size exceeds maximum
    if file_size_mb <= max_file_size_mb:
        try:
            # Check dimensions
            img = Image.open(io.BytesIO(img_data))
            width, height = img.size

            # If dimensions are fine, return original
            if width <= max_size and height <= max_size:
                return io.BytesIO(img_data)

        except Exception as e:
            print(f"Error checking image dimensions: {e}")
            # Return original if we can't check dimensions
            return io.BytesIO(img_data)

    # If we reach here, we need to resize the image
    try:
        # Convert to OpenCV format for resizing
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            # If OpenCV couldn't read, try with PIL
            img_pil = Image.open(io.BytesIO(img_data))
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Resize using our helper function
        img_resized = resize_image_if_needed(img, max_size=max_size, force_resize=(file_size_mb > max_file_size_mb))

        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        resized_data = io.BytesIO(buffer)

        print(f"Resized image from {len(img_data) / (1024 * 1024):.1f}MB to {len(buffer) / (1024 * 1024):.1f}MB")
        return resized_data

    except Exception as e:
        print(f"Error resizing image: {e}")
        # Return original if resize fails
        return io.BytesIO(img_data)


def clustering_worker(tolerance=0.6, min_cluster_size=1, save_mode="both",
                      detection_model="hog", upsample_times=1, min_face_size=20,
                      low_memory_mode=False, batch_size=10):
    """Background worker for clustering process"""
    global clustering_progress

    try:
        clustering_progress['status'] = 'running'
        clustering_progress['message'] = 'Starting clustering process...'
        clustering_progress['started_at'] = time.time()

        # Run clustering with status callback
        result_dir = cluster_faces(
            tolerance=tolerance,
            min_cluster_size=min_cluster_size,
            save_mode=save_mode,
            detection_model=detection_model,
            upsample_times=upsample_times,
            min_face_size=min_face_size,
            low_memory_mode=low_memory_mode,
            batch_size=batch_size,
            status_callback=status_callback
        )

        # Update symlink to new output
        if FACES_DIR.exists() or os.path.islink(str(FACES_DIR)):
            try:
                if os.path.islink(str(FACES_DIR)):
                    os.unlink(str(FACES_DIR))
                elif FACES_DIR.exists():
                    shutil.rmtree(str(FACES_DIR))
            except Exception as e:
                clustering_progress['message'] = f"Error removing old symlink: {e}"

        # Create new symlink
        try:
            os.symlink(Path(result_dir).resolve(), str(FACES_DIR), target_is_directory=True)
        except Exception as e:
            # Windows may need admin rights for symlinks, try a directory copy instead
            if sys.platform == 'win32':
                clustering_progress[
                    'message'] = "Could not create symlink (Windows may require admin privileges). Using directory copy instead."
                if not FACES_DIR.exists():
                    FACES_DIR.mkdir(parents=True)
                for item in Path(result_dir).iterdir():
                    if item.is_dir():
                        dest_dir = FACES_DIR / item.name
                        if dest_dir.exists():
                            shutil.rmtree(dest_dir)
                        shutil.copytree(item, dest_dir)
            else:
                clustering_progress['message'] = f"Error creating symlink: {e}"

        # Mark as completed
        clustering_progress['completed_at'] = time.time()
        duration = clustering_progress['completed_at'] - clustering_progress['started_at']
        clustering_progress['message'] = f"Clustering completed in {duration:.1f} seconds"
        clustering_progress['progress'] = 100
        clustering_progress['status'] = 'completed'
        clustering_progress['result'] = result_dir

    except Exception as e:
        clustering_progress['status'] = 'error'
        clustering_progress['error'] = str(e)
        clustering_progress['message'] = f"Error during clustering: {e}"
        print(f"Error in clustering worker: {e}")


def update_progress():
    """Update progress from status queue"""
    global clustering_progress

    while not status_queue.empty():
        message = status_queue.get_nowait()
        clustering_progress['message'] = message

        # Extract progress percentage if available
        if "progress" in message.lower() and "%" in message:
            try:
                progress_text = message.split("%")[0].split(":")[-1].strip()
                clustering_progress['progress'] = float(progress_text)
            except:
                # If we can't parse progress, just continue
                pass


@app.route("/", methods=["GET", "POST"])
def index():
    """Main page route"""
    # Update progress from queue
    update_progress()

    # Get upload count
    upload_count = len(list(UPLOAD_FOLDER.glob("*.*")))
    show_cluster_button = upload_count > 0

    # Process uploaded files
    if request.method == "POST" and 'images' in request.files:
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            flash("No files selected")
            return redirect(request.url)

        # Create upload directory if it doesn't exist
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

        # Save uploaded files with resize
        upload_count = 0
        too_large_count = 0
        for file in files:
            if file and file.filename:
                # Check file extension
                ext = os.path.splitext(file.filename)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    continue

                try:
                    # Check file size
                    file.seek(0, os.SEEK_END)
                    file_size_mb = file.tell() / (1024 * 1024)
                    file.seek(0)

                    if file_size_mb > MAX_UPLOAD_SIZE_MB:
                        too_large_count += 1
                        continue

                    # Process file with potential resize
                    resized_data = resize_image_for_upload(file)
                    resized_data.seek(0)

                    # Save to disk with safe filename
                    filename = os.path.basename(file.filename)
                    with open(UPLOAD_FOLDER / filename, 'wb') as f:
                        f.write(resized_data.getvalue())

                    upload_count += 1
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")

        if upload_count > 0:
            flash(f"Successfully uploaded {upload_count} images")
        if too_large_count > 0:
            flash(f"Skipped {too_large_count} images that exceeded the {MAX_UPLOAD_SIZE_MB}MB size limit")

        return redirect(url_for('index'))

    # Get list of runs
    runs = []
    try:
        for run_dir in sorted(RUNS_DIR.glob("*"), reverse=True):
            if run_dir.is_dir():
                run_date = run_dir.name.replace("_", " ").replace("-", "/")
                runs.append({
                    'date': run_date,
                    'path': run_dir.name
                })
    except Exception as e:
        print(f"Error listing runs: {e}")

    # Get face clusters
    clusters = []
    if FACES_DIR.exists():
        try:
            for cluster_dir in sorted(FACES_DIR.glob("*")):
                if cluster_dir.is_dir() and not cluster_dir.name.startswith(
                        ".") and not cluster_dir.name == "No_Faces_Detected":
                    preview_path = cluster_dir / "preview.jpg"

                    # Count files
                    file_count = len([f for f in cluster_dir.glob("*") if
                                      f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

                    # Check if preview exists
                    preview_rel_path = None
                    if preview_path.exists():
                        # Get relative path from static folder
                        parts = preview_path.parts
                        if 'static' in parts:
                            static_idx = parts.index('static')
                            preview_rel_path = '/'.join(parts[static_idx + 1:])
                        else:
                            preview_rel_path = str(preview_path)

                    # Default to first image if no preview
                    if not preview_rel_path:
                        for img in cluster_dir.glob("*.jpg"):
                            parts = img.parts
                            if 'static' in parts:
                                static_idx = parts.index('static')
                                preview_rel_path = '/'.join(parts[static_idx + 1:])
                            break

                    clusters.append({
                        'name': cluster_dir.name,
                        'preview': preview_rel_path,
                        'count': file_count
                    })
        except Exception as e:
            print(f"Error getting clusters: {e}")

    return render_template(
        'index.html',
        upload_count=upload_count,
        clusters=clusters,
        runs=runs,
        show_cluster_button=show_cluster_button
    )


@app.route('/cluster_now', methods=['POST'])
def cluster_now():
    """Start the clustering process"""
    global clustering_progress

    # Reset progress
    clustering_progress = {
        'status': 'starting',
        'message': 'Initializing...',
        'progress': 0,
        'started_at': None,
        'completed_at': None,
        'result': None,
        'error': None
    }

    # Get parameters from form
    tolerance = float(request.form.get('tolerance', 0.6))
    min_cluster_size = int(request.form.get('min_cluster_size', 1))
    save_mode = request.form.get('save_mode', 'both')
    detection_model = request.form.get('detection_model', 'hog')
    upsample_times = int(request.form.get('upsample_times', 1))
    min_face_size = int(request.form.get('min_face_size', 20))

    # Check memory to determine low memory mode
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    low_memory_mode = available_memory < 2000  # Under 2GB is low memory

    if low_memory_mode:
        print("Low memory mode enabled due to available memory under 2GB")

    # Determine batch size based on available memory
    if available_memory < 1000:  # Under 1GB
        batch_size = 2
    elif available_memory < 2000:  # Under 2GB
        batch_size = 5
    elif available_memory < 4000:  # Under 4GB
        batch_size = 10
    else:  # More than 4GB
        batch_size = 20

    # Start clustering in background thread
    cluster_thread = threading.Thread(
        target=clustering_worker,
        kwargs={
            'tolerance': tolerance,
            'min_cluster_size': min_cluster_size,
            'save_mode': save_mode,
            'detection_model': detection_model,
            'upsample_times': upsample_times,
            'min_face_size': min_face_size,
            'low_memory_mode': low_memory_mode,
            'batch_size': batch_size
        }
    )
    cluster_thread.daemon = True
    cluster_thread.start()

    return redirect(url_for('progress_page'))


@app.route('/progress')
def progress_page():
    """Show progress of clustering"""
    return render_template('progress.html')


@app.route('/get_progress')
def get_progress():
    """API endpoint to get current progress"""
    update_progress()

    return jsonify(clustering_progress)


@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """Clear all uploaded images"""
    try:
        for file in UPLOAD_FOLDER.glob("*.*"):
            file.unlink()
        flash("All uploaded images have been cleared")
    except Exception as e:
        flash(f"Error clearing uploads: {e}")

    return redirect(url_for('index'))


@app.route('/view_cluster/<cluster_name>')
def view_cluster(cluster_name):
    """View a specific cluster's images"""
    if not FACES_DIR.exists():
        flash("No face clusters available")
        return redirect(url_for('index'))

    # Get path to cluster
    cluster_dir = FACES_DIR / cluster_name
    if not cluster_dir.exists() or not cluster_dir.is_dir():
        flash(f"Cluster '{cluster_name}' not found")
        return redirect(url_for('index'))

    # Get images in cluster
    images = []
    try:
        for img_path in sorted(cluster_dir.glob("*.jpg")):
            if img_path.name == "preview.jpg":
                continue

            # Get relative path from static folder
            parts = img_path.parts
            if 'static' in parts:
                static_idx = parts.index('static')
                img_rel_path = '/'.join(parts[static_idx + 1:])

                # Check if it's a face or original
                is_face = "face" in img_path.name.lower() and not "original" in img_path.name.lower()

                images.append({
                    'path': img_rel_path,
                    'name': img_path.name,
                    'is_face': is_face
                })
    except Exception as e:
        flash(f"Error getting images: {e}")

    return render_template(
        'gallery.html',
        cluster_name=cluster_name,
        images=images
    )


@app.route('/switch_run/<run_path>')
def switch_run(run_path):
    """Switch to a different clustering run"""
    run_dir = RUNS_DIR / run_path

    if not run_dir.exists() or not run_dir.is_dir():
        flash(f"Run '{run_path}' not found")
        return redirect(url_for('index'))

    # Update symlink or copy directory
    if FACES_DIR.exists() or os.path.islink(str(FACES_DIR)):
        try:
            if os.path.islink(str(FACES_DIR)):
                os.unlink(str(FACES_DIR))
            elif FACES_DIR.exists():
                shutil.rmtree(str(FACES_DIR))
        except Exception as e:
            flash(f"Error removing old faces dir: {e}")

    try:
        # Create symlink
        os.symlink(run_dir.resolve(), str(FACES_DIR), target_is_directory=True)
        flash(f"Switched to run: {run_path}")
    except Exception as e:
        # Windows fallback - copy directory
        if sys.platform == 'win32':
            if not FACES_DIR.exists():
                FACES_DIR.mkdir(parents=True)
            for item in run_dir.iterdir():
                if item.is_dir():
                    dest_dir = FACES_DIR / item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(item, dest_dir)
            flash(f"Switched to run: {run_path} (directory copy mode)")
        else:
            flash(f"Error switching runs: {e}")

    return redirect(url_for('index'))


if __name__ == "__main__":
    # Create necessary directories if they don't exist
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Make sure static/faces exists (even if just an empty dir)
    if not FACES_DIR.exists() and not os.path.islink(str(FACES_DIR)):
        FACES_DIR.mkdir(parents=True, exist_ok=True)

    # Start Flask
    app.run(debug=True, host='0.0.0.0', port=5000)