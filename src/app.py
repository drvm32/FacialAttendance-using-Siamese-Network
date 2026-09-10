from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from inference import mark_attendance, load_attendance, reset_daily_status, add_new_person_to_records, train_with_new_image
from preprocess import load_reference_images
import threading
import time
import os

app = Flask(__name__)

# Global state
ref_images = load_reference_images('D:\\New folder\\data\\reference_images')
attendance_data = load_attendance(ref_images)
model_file = 'D:\\New folder\\models\\siamese_model.h5'
cap = None  # Initialize cap as None
frame = None
result = "Idle"  # Initial state
history = []
has_trained = False
data_lock = threading.Lock()
camera_active = False  # Track camera state
video_thread = None  # Track the video feed thread

def initialize_training():
    global attendance_data, has_trained
    if not has_trained:
        existing_people = set(attendance_data.keys())
        current_people = set(ref_images.keys())
        new_people = current_people - existing_people

        if new_people:
            for person in new_people:
                print(f"New person detected: {person}")
                with data_lock:
                    add_new_person_to_records(person, attendance_data)
                print(f"Please add training pairs for {person} in 'data/training_pairs/'")
                input("Press Enter after adding pairs to retrain...")
                train_with_new_image(person)
        else:
            print("No new people detected; using existing model.")
        has_trained = True

def video_feed_thread():
    global frame, result, history, attendance_data, cap, camera_active
    while camera_active:  # Run only when camera_active is True
        ret, frame = cap.read()
        if not ret:
            result = "Error: Could not read frame."
            camera_active = False  # Stop if frame read fails
            break
        with data_lock:
            result, history = mark_attendance(frame, attendance_data, history)
        time.sleep(0.1)
    if cap is not None:
        cap.release()
        print("Camera released")

def gen_frames():
    global frame, result, camera_active
    while camera_active:
        if frame is not None:
            display_frame = frame.copy()
            cv2.putText(display_frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)
    print("Frame generation stopped")
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')  # Send an empty frame to stop the stream

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, camera_active, video_thread, result
    if not camera_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open webcam'}), 500
        camera_active = True
        result = "Camera Active"
        video_thread = threading.Thread(target=video_feed_thread, daemon=True)
        video_thread.start()
        return jsonify({'status': 'Camera started'}), 200
    else:
        return jsonify({'status': 'Camera already active'}), 200

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, result
    if camera_active:
        camera_active = False  # This will stop the threads
        result = "Camera Stopped"
        return jsonify({'status': 'Camera stopped'}), 200
    else:
        return jsonify({'status': 'Camera not active'}), 200

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def get_attendance():
    with data_lock:
        reset_daily_status(attendance_data)
        return jsonify(attendance_data)
#here
@app.route('/download_report/late_attendance')
def download_late_report():
    try:
        # Call the function from utils.py to generate the file
        report_path = generate_late_attendance_report()

        # Send the generated file to the user for download
        return send_file(
            report_path,
            as_attachment=True,
            download_name='Late_Attendance_Report.xlsx', # The filename the user will see
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        # Handle potential errors during report generation
        print(f"Error generating report: {e}")
        return "Error generating report.", 500
    # Optional: Clean up the temporary file after sending if needed
    # finally:
    #     if os.path.exists(report_path):
    #         os.remove(report_path)
#here
def close_app():
    global cap, camera_active
    camera_active = False  # Ensure threads stop
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    initialize_training()
    try:
        app.run(host='127.0.0.1', port=5000, debug=True)
    finally:
        close_app()