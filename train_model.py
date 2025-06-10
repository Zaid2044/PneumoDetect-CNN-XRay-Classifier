# train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
BASE_PATH = 'chest_xray'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
VAL_PATH = os.path.join(BASE_PATH, 'val')
TEST_PATH = os.path.join(BASE_PATH, 'test')

# Define model parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# --- 1. Data Loading and Augmentation ---
print("[INFO] Preparing data generators...")

# Create an image data generator for the training set with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Create an image data generator for validation and test sets (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True)

val_generator = val_test_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)

test_generator = val_test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=1,
    class_mode='binary',
    shuffle=False)

print(f"[INFO] Found {train_generator.samples} images in training set.")
print(f"[INFO] Found {val_generator.samples} images in validation set.")
print(f"[INFO] Found {test_generator.samples} images in test set.")

# --- 2. Model Building (Transfer Learning with VGG16) ---
print("[INFO] Building model...")

# Load the VGG16 model, pre-trained on ImageNet, without the top classification layer
base_model = VGG16(weights="imagenet", include_top=False,
                   input_tensor=Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

# Freeze the layers of the base model so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Create our new classification head to put on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="sigmoid")(head_model)

# Combine the base model and our new head model
model = Model(inputs=base_model.input, outputs=head_model)

# --- 3. Model Compilation ---
print("[INFO] Compiling model...")
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.summary()

# --- 4. Model Training ---
print("[INFO] Training model...")

# Define callbacks for smarter training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr])

# --- 5. Model Evaluation ---
print("[INFO] Evaluating model on the test set...")

# Get predictions from the test set
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples)
binary_predictions = (predictions > 0.5).astype(int).flatten()

# Get true labels
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, binary_predictions, target_names=class_labels))

# --- 6. Visualization and Saving Results ---
print("[INFO] Generating and saving visualizations...")

# Plot training history (Accuracy and Loss)
plt.style.use("ggplot")
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("training_history.png")
print("[INFO] Saved training history plot to training_history.png")

# Generate and plot confusion matrix
cm = confusion_matrix(true_labels, binary_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
print("[INFO] Saved confusion matrix plot to confusion_matrix.png")

# --- 7. Save the Trained Model ---
print("[INFO] Saving trained model...")
model.save("pneumonia_detector_model.h5")
print("[INFO] Model saved as pneumonia_detector_model.h5")
print("\n--- PNEUMODETECT TRAINING COMPLETE ---")