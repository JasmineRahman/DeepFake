import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
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

# Split validation set into validation and test sets
fake_val, fake_test = train_test_split(fake_val, test_size=0.5, random_state=42)
real_val, real_test = train_test_split(real_val, test_size=0.5, random_state=42)

# Training Data Generator
train_generator = train_datagen.flow_from_directory(
    r'C:\PYTHON\train_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation Data Generator
validation_generator = train_datagen.flow_from_directory(
    r'C:\PYTHON\train_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    r'C:\PYTHON\test_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Model Architecture
def build_model(learning_rate, batch_size):
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

    # Compile model with Learning Rate Schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )

    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Define the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'accuracy'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-2},
        'batch_size': {'min': 16, 'max': 128},
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project='deepfake-detection')

def train_evaluate_model():
    # Fetch hyperparameters from WandB
    config = wandb.init()
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    # Build and train the model
    model = build_model(learning_rate, batch_size)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    # Save the trained model
    model.save('C://PYTHON//saved_model', save_format='tf')

    # Model Evaluation Metrics on Test Set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')

    # Log metrics to WandB
    wandb.log({'accuracy': test_accuracy, 'loss': test_loss})

    # Prediction function
    def predict_image(image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)
        probability = prediction[0][0]

        print(f"Probability of being a deepfake: {probability * 100:.2f}%")
        if probability < 0.5:
            print("Image is likely real.")
        else:
            print("Image is likely deepfake.")

    # Example usage
    image_path_real = r'C:\Users\dhaks\OneDrive\Pictures\Saved Pictures\original1.jpeg'
    image_path_fake = r'C:\Users\dhaks\OneDrive\Pictures\Saved Pictures\fake1.jpeg'

    predict_image(image_path_real)
    predict_image(image_path_fake)

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the sweep
wandb.agent(sweep_id, function=train_evaluate_model)