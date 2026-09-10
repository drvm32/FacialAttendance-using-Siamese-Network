# inference PY
import cv2
import numpy as np
from siamese_model import siamese_network
from preprocess import detect_and_align_face, load_reference_images
from datetime import datetime, timedelta
import json
import os
import tensorflow as tf
from train import load_training_pairs, contrastive_loss

attendance_file = '../data/attendance.json'
model_file = '../models/siamese_model.h5'


def load_attendance(ref_images=None):
    if os.path.exists(attendance_file) and ref_images is None:
        with open(attendance_file, 'r') as f:
            data = json.load(f)
            print(f"Loaded attendance data: {data}")
            return data
    ref_images = ref_images or load_reference_images('../data/reference_images/')
    data = {person: {"last_marked": None, "status": "absent", "history": []} for person in ref_images.keys()}
    save_attendance(data)
    return data


def save_attendance(attendance_data):
    try:
        with open(attendance_file, 'w') as f:
            json.dump(attendance_data, f, indent=4)
        print(f"Saved attendance data: {attendance_data}")
    except Exception as e:
        print(f"Error saving attendance: {e}")


def can_mark_attendance(last_marked):
    if last_marked is None:
        return True
    last_time = datetime.strptime(last_marked, "%Y-%m-%d %H:%M:%S")
    return datetime.now() - last_time > timedelta(hours=24)


def reset_daily_status(attendance_data):
    today = datetime.now().date()
    for person, data in attendance_data.items():
        if data["last_marked"]:
            last_date = datetime.strptime(data["last_marked"], "%Y-%m-%d %H:%M:%S").date()
            if today > last_date:
                data["status"] = "absent"
    save_attendance(attendance_data)


def add_new_person_to_records(person_name, attendance_data):
    if person_name not in attendance_data:
        attendance_data[person_name] = {"last_marked": None, "status": "absent", "history": []}
        save_attendance(attendance_data)
        print(f"Added {person_name} to attendance records")


def train_with_new_image(new_person_name):
    if os.path.exists(model_file):
        model, base_network = siamese_network((128, 128, 3))
        model.load_weights(model_file)
    else:
        model, base_network = siamese_network((128, 128, 3))

    pairs, labels, pair_ids = load_training_pairs('../data/training_pairs/')
    pair_a, pair_b = pairs[:, 0], pairs[:, 1]
    print(f"Training with {len(pairs)} pairs including new data for {new_person_name}")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=contrastive_loss)
    model.fit([pair_a, pair_b], labels, epochs=50, batch_size=2, validation_split=0.2, verbose=5)
    model.save(model_file)
    print(f"Retrained model saved to {model_file}")


def mark_attendance(live_img, attendance_data, history=[]):
    ref_images = load_reference_images('../data/reference_images/')
    model, _ = siamese_network((128, 128, 3))
    model.load_weights(model_file)

    live_face = detect_and_align_face(live_img)
    if live_face is None:
        return "No face detected", history

    live_face = np.expand_dims(live_face, axis=0)
    min_dist = float('inf')
    person_id = "Unknown"

    for ref_id, ref_img in ref_images.items():
        ref_img = np.expand_dims(ref_img, axis=0)
        dist = model.predict([live_face, ref_img], verbose=0)[0][0]
        if dist < min_dist:
            min_dist = dist
            person_id = ref_id if dist < 0.7 else "Unknown"

    history.append((person_id, min_dist))
    if len(history) > 5:
        history.pop(0)
    avg_dist = np.mean([d for _, d in history])
    final_id = max(set([p for p, _ in history]), key=[p for p, _ in history].count) if avg_dist < 0.7 else "Unknown"

    if final_id != "Unknown":
        add_new_person_to_records(final_id, attendance_data)
        if can_mark_attendance(attendance_data[final_id]["last_marked"]):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance_data[final_id]["last_marked"] = timestamp
            attendance_data[final_id]["status"] = "present"
            attendance_data[final_id]["history"].append({"timestamp": timestamp, "status": "present"})
            save_attendance(attendance_data)
            return f"Attendance marked for {final_id} at {timestamp}", history
        return f"Attendance already marked for {final_id} today", history
    return "Unknown person", history


if __name__ == "__main__":
    ref_images = load_reference_images('../data/reference_images/')
    attendance_data = load_attendance(ref_images)

    existing_people = set(attendance_data.keys())
    current_people = set(ref_images.keys())
    new_people = current_people - existing_people

    for person in new_people:
        print(f"New person detected: {person}")
        add_new_person_to_records(person, attendance_data)
        print(
            f"Please add training pairs for {person} in '../data/training_pairs/' (e.g., same_person/pair5/, diff_person/pair6/)")
        input("Press Enter after adding pairs to retrain...")
        train_with_new_image(person)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    history = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        result, history = mark_attendance(frame, attendance_data, history)
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reset_daily_status(attendance_data)
    cap.release()
    cv2.destroyAllWindows()