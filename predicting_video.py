import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from efficientnet.tfkeras import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, Dropout
from mtcnn import MTCNN

input_size = 128
best_model_path = 'tmp_checkpoint/best_model.h5'


efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)

model = Sequential([
    efficient_net,
    Dense(units=512, activation='relu'),
    Dropout(0.5),
    Dense(units=128, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.load_weights(best_model_path)

def get_filename_only(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def predict_video(video_path):
    base_path = 'C:/Users/Smile/Desktop/Hack/DeepFake-Detect/testing_video'
    
    if video_path.endswith(".mp4"):
        tmp_path = os.path.join(base_path, get_filename_only(video_path)).replace('\\', '/')
        os.makedirs(tmp_path, exist_ok=True)
        
        print(f'Creating Directory: {tmp_path}')
        print('Converting Video to Images...')
        
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        
        while cap.isOpened():
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % int(frame_rate) == 0:
                scale_ratio = 1
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif frame.shape[1] > 1000:
                    scale_ratio = 0.5
                
                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                
                new_filename = f'{os.path.join(tmp_path, get_filename_only(video_path))}-{count:03d}.png'.format(count)
                count += 1
                cv2.imwrite(new_filename, new_frame)
        
        cap.release()
    
    print(f'Processing Directory: {tmp_path}')
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
    
    faces_path = os.path.join(tmp_path, 'faces').replace("\\", '/')
    os.makedirs(faces_path, exist_ok=True)
    
    print('Cropping Faces from Images...')
    detector = MTCNN()
    
    for frame in frame_images:
        image_path = os.path.join(tmp_path, frame).replace('\\', '/')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        
        count = 0
        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']
            
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3
                margin_y = bounding_box[3] * 0.3
                x1 = max(int(bounding_box[0] - margin_x), 0)
                x2 = min(int(bounding_box[0] + bounding_box[2] + margin_x), image.shape[1])
                y1 = max(int(bounding_box[1] - margin_y), 0)
                y2 = min(int(bounding_box[1] + bounding_box[3] + margin_y), image.shape[0])
                
                crop_image = image[y1:y2, x1:x2]
                new_filename = f'{os.path.join(faces_path, get_filename_only(frame))}-{count:02d}.png'.format(count)
                count += 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
    
    fake_count = 0
    real_count = 0
    predictions = []
    
    for img_name in os.listdir(faces_path):
        if img_name.endswith(".png"):
            img_path = os.path.join(faces_path, img_name).replace("\\", "/")
            img_array = cv2.imread(img_path)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, (128, 128))
            img_array = img_array / 255.0
            
            prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
            predictions.append(prediction)
            
            
            if prediction > 0.1:
                fake_count += 1
                result_text = "The video is classified as fake."
                break
            else:
                real_count += 1
    print(predictions)
    print("real",real_count)
    print("fake",fake_count)
    
    if fake_count==0:
        result_text = "The video is classified as real."
    
    print(result_text)
    
    # Display video with result text
    fig, ax = plt.subplots()
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.imshow(frame)
        ax.text(0.5, -0.1, result_text, size=14, ha='center')
        ax.axis('off')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=video_frame_generator(video_path), blit=False, interval=50)
    plt.show()

def video_frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb
    cap.release()
