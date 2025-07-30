import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(image_dimensions['height'], image_dimensions['width'], image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

def detect_deepfake(file):
    meso = Meso4()
    meso.load(r'C:/Users/Smile/Documents/image_fake_detection-main[1]/image_fake_detection-main/app/weights/Meso4_DF')

    # Get the file stream from the Werkzeug FileStorage object
    file_stream = file.stream

    # Use PIL to open the image from the file stream
    img = Image.open(file_stream)
    img = img.convert('RGB')  # Ensure that the image is in RGB format

    # Resize the image
    img = img.resize((256, 256))

    # Convert the image to a NumPy array
    img_array = img_to_array(img)

    # Create a single-image "batch" as a NumPy array with an extra dimension for batch size
    img_batch = np.expand_dims(img_array, axis=0)

    # Use the ImageDataGenerator to apply any additional preprocessing (optional)
    data_generator = ImageDataGenerator(rescale=1./255)
    preprocessed_img = data_generator.flow(img_batch)  # Generate a single-image batch

    # Access the preprocessed image
    processed_img = preprocessed_img[0]  # Extract the first (and only) image from the generator

    prediction = meso.predict(processed_img)[0][0]
    if prediction < 0.8:
        result_text = "Image is likely fake"
    else:
        result_text = "Image is likely real"
    print(prediction)

    # Create a figure and an axes object
    fig, ax = plt.subplots()

    # Display the image on the axes
    ax.imshow(img)

    # Set a descriptive title for the plot
    ax.set_title(prediction)
    ax.text(30,-7,"prediction :",size=12)
    ax.text(40,-23,result_text,size=14)
    # Optionally, remove axes for a cleaner look
    ax.axis('off')

    # Show the plot
    plt.show()

    return prediction
