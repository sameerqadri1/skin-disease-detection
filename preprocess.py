import os
import glob
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shutil

# Step 1 & 2: Load the dataset
dataset_path = "dataset"  # Use relative path to avoid escape issues
# Alternatively, can use:
# dataset_path = os.path.join(os.getcwd(), "dataset")  # Absolute path with os.path.join
images = []
labels = []

for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(class_folder)

images = np.array(images)
labels = np.array(labels)
print(f"Loaded {len(images)} images with shape: {images[0].shape}")

# Step 3: Identify classes
unique_classes = np.unique(labels)
print("Classes in the dataset:")
for i, class_name in enumerate(unique_classes):
    print(f"{i+1}. {class_name}")

# Step 4: Data augmentation
images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
images_rgb = np.array(images_rgb)

datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

augmented_images = []
augmented_labels = []
augmentations_per_image = 2

for i, (img, label) in enumerate(zip(images_rgb, labels)):
    img = np.expand_dims(img, axis=0)
    aug_iter = datagen.flow(img, batch_size=1)
    for _ in range(augmentations_per_image):
        aug_img = next(aug_iter)[0].astype(np.uint8)
        augmented_images.append(aug_img)
        augmented_labels.append(label)

all_images = np.concatenate([images_rgb, augmented_images])
all_labels = np.concatenate([labels, augmented_labels])
print(f"Total images after augmentation: {len(all_images)}")

# Step 5: Prepare images and labels [Img resizing & Normalization]
target_size = (224, 224)
resized_images = [cv2.resize(img, target_size) for img in all_images]
resized_images = np.array(resized_images)
# Don't normalize here since we're saving as images
# We'll normalize during training

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)
print(f"Image shape after resizing: {resized_images.shape}")
print(f"Class mapping: {dict(enumerate(label_encoder.classes_))}")

# Step 6: Data splitting
X_train, X_temp, y_train, y_temp = train_test_split(
    resized_images, 
    encoded_labels, 
    test_size=0.3,  # 70% for training, 30% for test+val 
    random_state=42,
    stratify=encoded_labels  # Ensure each split has the same class distribution
)

# Further split the test data into test and validation
X_test, X_val, y_test, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,  # 15% for test, 15% for validation
    random_state=42,
    stratify=y_temp
)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Step 7: Save preprocessed data
preprocessed_dir = "preprocessed_data"
os.makedirs(preprocessed_dir, exist_ok=True)

# Create train_ds, test_ds, val_ds directories
train_ds_dir = os.path.join(preprocessed_dir, "train_ds")
test_ds_dir = os.path.join(preprocessed_dir, "test_ds")
val_ds_dir = os.path.join(preprocessed_dir, "val_ds")

# Remove existing directories if they exist to ensure clean data
if os.path.exists(train_ds_dir):
    shutil.rmtree(train_ds_dir)
if os.path.exists(test_ds_dir):
    shutil.rmtree(test_ds_dir)
if os.path.exists(val_ds_dir):
    shutil.rmtree(val_ds_dir)

os.makedirs(train_ds_dir, exist_ok=True)
os.makedirs(test_ds_dir, exist_ok=True)
os.makedirs(val_ds_dir, exist_ok=True)

# Save class names
np.save(os.path.join(preprocessed_dir, "classes.npy"), label_encoder.classes_)
print(f"Classes saved to {os.path.join(preprocessed_dir, 'classes.npy')}")

# Create directories for each class in train_ds, test_ds, and val_ds
for cls in label_encoder.classes_:
    os.makedirs(os.path.join(train_ds_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_ds_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_ds_dir, cls), exist_ok=True)

# Save images to their respective directories
print("Saving preprocessed images to directories...")

# Helper function to save images
def save_image_set(images, labels, base_dir, class_names):
    for i, (img, label_idx) in enumerate(zip(images, labels)):
        class_name = class_names[label_idx]
        class_dir = os.path.join(base_dir, class_name)
        img_path = os.path.join(class_dir, f"img_{i}.jpg")
        # Convert back to BGR for OpenCV
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Print progress every 100 images
        if i % 100 == 0:
            print(f"Saved {i}/{len(images)} images to {base_dir}")

# Save training images
save_image_set(X_train, y_train, train_ds_dir, label_encoder.classes_)
print(f"Saved {len(X_train)} training images")

# Save validation images
save_image_set(X_val, y_val, val_ds_dir, label_encoder.classes_)
print(f"Saved {len(X_val)} validation images")

# Save test images
save_image_set(X_test, y_test, test_ds_dir, label_encoder.classes_)
print(f"Saved {len(X_test)} test images")

print("Preprocessing completed successfully!")