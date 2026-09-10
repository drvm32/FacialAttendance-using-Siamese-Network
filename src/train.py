# train.py
import os
import numpy as np
import tensorflow as tf
from siamese_model import siamese_network
from preprocess import detect_and_align_face
import pickle

# --- Function definitions are OK at module level ---
def contrastive_loss(y_true, y_pred):
    """
    Calculates the contrastive loss.

    Args:
        y_true: True labels (0 or 1). 1 means the pairs are similar, 0 means they are dissimilar.
        y_pred: Predicted distances between pairs by the Siamese network.

    Returns:
        The contrastive loss value.
    """
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32) # Ensure y_true is float32
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def load_training_pairs(directory):
    """
    Loads image pairs for training the Siamese network.

    Looks for subdirectories 'same_person' and 'diff_person' within the given directory.
    Each subdirectory should contain further subdirectories, each representing a pair
    with at least two .jpg images.

    Args:
        directory: The path to the directory containing 'same_person' and 'diff_person' folders.

    Returns:
        A tuple containing:
            - pairs (np.array): An array of shape (num_pairs, 2, height, width, channels)
            - labels (np.array): An array of shape (num_pairs,) with 1 for same, 0 for different.
            - pair_ids (list): A list of tuples with the filenames for each pair.
    """
    pairs = []
    labels = []
    pair_ids = []
    print(f"Looking for pairs in: {directory}")
    for subdir in ['same_person', 'diff_person']:
        path = os.path.join(directory, subdir)
        print(f"Checking subdirectory: {path}")
        if not os.path.exists(path):
            print(f"Warning: Subdirectory not found - {path}")
            continue
        if not os.path.isdir(path):
             print(f"Warning: Expected directory, found file - {path}")
             continue

        pair_folders_count = 0
        for pair_folder in os.listdir(path):
            pair_path = os.path.join(path, pair_folder)
            if not os.path.isdir(pair_path): # Skip if it's not a directory
                 print(f"Skipping non-directory item: {pair_path}")
                 continue

            pair_folders_count += 1
            images = [f for f in os.listdir(pair_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] # Accept more extensions
            #print(f"Found {len(images)} images in {pair_path}")
            if len(images) >= 2:
                # Try to load the first two suitable images
                img1_path = os.path.join(pair_path, images[0])
                img2_path = os.path.join(pair_path, images[1])
                img1 = detect_and_align_face(img1_path)
                img2 = detect_and_align_face(img2_path)

                if img1 is not None and img2 is not None:
                    pairs.append([img1, img2])
                    labels.append(1.0 if subdir == 'same_person' else 0.0) # Use floats for labels
                    pair_ids.append((images[0], images[1]))
                    #print(f"Successfully loaded pair from {pair_folder}")
                else:
                    print(f"Warning: Could not detect/align face in one or both images in {pair_path} ({images[0]}, {images[1]})")
            else:
                print(f"Warning: Found less than 2 images in pair folder {pair_path}")
        if pair_folders_count == 0:
             print(f"Warning: No pair folders found in {path}")

    if not pairs:
         print("Error: No valid image pairs were loaded. Check directory structure and image files.")
         return np.array([]), np.array([]), []

    return np.array(pairs), np.array(labels), pair_ids

# --- Wrap the main execution logic ---
if __name__ == "__main__":
    print("Running train.py directly...")

    # Define base directory relative to the script location might be more robust
    script_dir = os.path.dirname(__file__) # Gets the directory where train.py is located
    data_dir = os.path.abspath(os.path.join(script_dir, '../data/training_pairs/'))
    model_dir = os.path.abspath(os.path.join(script_dir, '../models/'))

    pairs, labels, pair_ids = load_training_pairs(data_dir)
    if len(pairs) == 0:
        print("No training pairs found. Exiting.")
        exit()

    # Ensure labels are float32, matching the expected type for the loss function
    labels = np.array(labels, dtype=np.float32)

    pair_a = pairs[:, 0]
    pair_b = pairs[:, 1]
    print(f"Loaded {len(pairs)} training pairs: {len(labels)} labels (1.0=same, 0.0=diff)")

    model, base_network = siamese_network((128, 128, 3)) # Assuming (128, 128, 3) is correct
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=contrastive_loss)
    model.summary() # Print model summary

    print("Starting model training...")
    history = model.fit(
        [pair_a, pair_b], labels,
        epochs=50,
        batch_size=16, # Increased batch size slightly, adjust based on memory
        validation_split=0.2, # Use 20% of data for validation
        verbose=1,
        shuffle=True # Shuffle training data each epoch
    )

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'siamese_model.h5')
    val_data_path = os.path.join(model_dir, 'val_data.pkl') # Path for validation data

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # --- Save validation data (Indices might be better than full pairs) ---
    # Keras shuffles data before splitting if shuffle=True. Getting the exact validation set
    # used by Keras requires careful index handling or using callbacks.
    # For simplicity here, we'll re-split and save that split. This might differ slightly
    # from the exact set Keras used internally if shuffling was different.
    from sklearn.model_selection import train_test_split
    # Re-split the original data to get a validation set to save
    # Use a fixed random_state for reproducibility of the split saved to file
    print(
        "Warning: Performing non-stratified split for saving validation data due to small dataset size.")  # Optional Warning
    X_train_save, X_val_save, y_train_save, y_val_save = train_test_split(
        pairs, labels, test_size=0.2, random_state=42  # <--- Removed stratify=labels
    )
    validation_data_to_save = {'X_val': X_val_save, 'y_val': y_val_save}
    with open(val_data_path, 'wb') as f:
        pickle.dump(validation_data_to_save, f)
    print(f"Validation split data saved to {val_data_path}")


    # --- Evaluation and Plotting Section ---
    print("\nEvaluating model and generating plots...")

    # Import plotting and metrics libraries here (only needed when __name__ == "__main__")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Load the validation data we just saved
    try:
        with open(val_data_path, 'rb') as f:
            val_data = pickle.load(f)
        X_val = val_data['X_val']
        val_labels = val_data['y_val'] # Already float32 from earlier conversion
        val_pair_a = X_val[:, 0]
        val_pair_b = X_val[:, 1]
        print(f"Loaded {len(val_labels)} pairs for validation plotting from {val_data_path}")
    except Exception as e:
        print(f"Could not load validation data from {val_data_path} for plotting: {e}")
        val_pair_a, val_pair_b, val_labels = None, None, None # Set to None if loading fails

    if val_labels is not None and len(val_labels) > 0: # Check if validation data is available
        # 1. Get Predictions (distances) on validation set
        predicted_distances = model.predict([val_pair_a, val_pair_b])
        predicted_distances = predicted_distances.flatten()

        # === Plot 1: Confusion Matrix ===
        eval_threshold = 0.7 # Threshold used for evaluation/inference
        # Convert predictions to binary based on threshold (1 if < threshold, 0 if >= threshold)
        binary_predictions = (predicted_distances < eval_threshold).astype(int)
        # Ensure true labels are also integer type for confusion matrix
        int_val_labels = val_labels.astype(int)

        cm = confusion_matrix(int_val_labels, binary_predictions)
        accuracy = accuracy_score(int_val_labels, binary_predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Diff (0)', 'Predicted Same (1)'],
                    yticklabels=['Actual Diff (0)', 'Actual Same (1)'])
        plt.title(f'Confusion Matrix (Threshold={eval_threshold}, Accuracy={accuracy:.3f})')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
        try:
            plt.savefig(confusion_matrix_path)
            print(f"Confusion matrix plot saved to {confusion_matrix_path}")
        except Exception as e:
            print(f"Error saving confusion matrix plot: {e}")
        plt.close() # Close the plot to free memory

        # === Plot 2: Accuracy vs. Threshold ===
        # Define thresholds relative to the observed distances
        min_dist = np.min(predicted_distances)
        max_dist = np.max(predicted_distances)
        # Avoid thresholds too far outside the range, add a small buffer
        thresholds = np.linspace(max(0, min_dist - 0.1), min(2.0, max_dist + 0.1), 100)
        accuracies = [accuracy_score(int_val_labels, (predicted_distances < t).astype(int)) for t in thresholds]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, marker='.', linestyle='-')
        # Mark the threshold used for the confusion matrix
        plt.axvline(x=eval_threshold, color='r', linestyle='--', label=f'Eval Threshold ({eval_threshold})')

        # Find best threshold based on validation accuracy
        if accuracies: # Ensure accuracies list is not empty
             best_threshold_idx = np.argmax(accuracies)
             best_threshold = thresholds[best_threshold_idx]
             best_accuracy = accuracies[best_threshold_idx]
             plt.axvline(x=best_threshold, color='g', linestyle=':', label=f'Best Valid Threshold ({best_threshold:.3f}, Acc={best_accuracy:.3f})')
             plt.scatter([best_threshold], [best_accuracy], color='g', s=100, label=f'Best Point') # Highlight best point

        plt.title('Accuracy vs. Distance Threshold on Validation Set')
        plt.xlabel('Distance Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1.05) # Ensure y-axis goes from 0 to slightly above 1
        accuracy_plot_path = os.path.join(model_dir, 'accuracy_vs_threshold.png')
        try:
            plt.savefig(accuracy_plot_path)
            print(f"Accuracy vs. Threshold plot saved to {accuracy_plot_path}")
        except Exception as e:
             print(f"Error saving accuracy plot: {e}")
        plt.close() # Close the plot

        # === Plot 3: Training History (Loss) ===
        if history is not None and hasattr(history, 'history'):
            plt.figure(figsize=(10, 6))
            if 'loss' in history.history:
                 plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                 plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (Contrastive)')
            plt.legend()
            plt.grid(True)
            loss_plot_path = os.path.join(model_dir, 'training_loss_history.png')
            try:
                 plt.savefig(loss_plot_path)
                 print(f"Training loss history plot saved to {loss_plot_path}")
            except Exception as e:
                 print(f"Error saving loss history plot: {e}")
            plt.close()

    else:
        print("Skipping plot generation due to missing or empty validation data.")

# --- End of main execution logic ---