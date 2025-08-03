
import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Define constants that are used in the notebook
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 2
BATCH_SIZE = 32

def create_project_structure(processed_data_path, models_path):
    print("Creating project directory structure...")
    os.makedirs(os.path.join(processed_data_path, 'train', 'Engaged'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_path, 'train', 'Not Engaged'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_path, 'test', 'Engaged'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_path, 'test', 'Not Engaged'), exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    print("Project structure created successfully.")

def download_and_preprocess_data(kaggle_dataset_name, data_path, models_path):
    print("Downloading dataset from Kaggle...")
    try:
        download_path = kagglehub.dataset_download(kaggle_dataset_name)
        base_data_path = os.path.join(download_path, 'Student-engagement-dataset')

        raw_engaged_path = os.path.join(base_data_path, 'Engaged')
        raw_not_engaged_path = os.path.join(base_data_path, 'Not Engaged')

        engaged_images = glob(os.path.join(raw_engaged_path, '*', '*.jpg')) +                          glob(os.path.join(raw_engaged_path, '*', '*.jpeg')) +                          glob(os.path.join(raw_engaged_path, '*', '*.png'))

        not_engaged_images = glob(os.path.join(raw_not_engaged_path, '*', '*.jpg')) +                              glob(os.path.join(raw_not_engaged_path, '*', '*.jpeg')) +                              glob(os.path.join(raw_not_engaged_path, '*', '*.png'))

        all_images = engaged_images + not_engaged_images
        labels = [0] * len(engaged_images) + [1] * len(not_engaged_images)

        train_images, test_images, train_labels, test_labels = train_test_split(
            all_images, labels, test_size=0.2, random_state=42, stratify=labels)

        create_project_structure(data_path, models_path)

        print("\nSplitting images and copying to train/test directories...")
        for img_path, label in zip(train_images, train_labels):
            dest_dir = os.path.join(data_path, 'train', 'Engaged' if label == 0 else 'Not Engaged')
            shutil.copy(img_path, dest_dir)

        for img_path, label in zip(test_images, test_labels):
            dest_dir = os.path.join(data_path, 'test', 'Engaged' if label == 0 else 'Not Engaged')
            shutil.copy(img_path, dest_dir)

        print(f"Training data: {len(train_images)} images")
        print(f"Testing data: {len(test_images)} images")
        print("Data successfully processed and split.")

    except Exception as e:
        print(f"Error during data download and preprocessing: {e}")


def get_data_generators(train_path, test_path):
    print("\nSetting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    val_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    print("Data generators created.")
    return train_generator, val_generator, test_generator
