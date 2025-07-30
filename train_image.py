import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import wandb

# Initialize WandB
wandb.init(project='deepfake-detection')

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Split data into training, validation, and test sets
real_images = os.path.join(r'C:\PYTHON\train_images', 'real')
fake_images = os.path.join(r'C:\PYTHON\train_images', 'fake')

real_train, real_val = train_test_split(os.listdir(real_images), test_size=0.2, random_state=42)
fake_train, fake_val = train_test_split(os.listdir(fake_images), test_size=0.2, random_state=42)

# Training Data Generator
train_generator = train_datagen.flow_from_directory(
    r'C:\PYTHON\train_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Validation Data Generator
validation_generator = train_datagen.flow_from_directory(
    r'C:\PYTHON\train_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Model Architecture
def build_model(learning_rate):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )

    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Build and train the model
model = build_model(learning_rate=1e-3)

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# Save the trained model
model.save('C:\\PYTHON\\saved_model', save_format='tf')

# Log metrics to WandB
wandb.log({'accuracy': history.history['accuracy'][-1], 'val_accuracy': history.history['val_accuracy'][-1]})
