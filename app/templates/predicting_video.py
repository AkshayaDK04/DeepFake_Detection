import tensorflow as tf
import os,re
import cv2
import math
from mtcnn import MTCNN
from keras.preprocessing.image import img_to_array, load_img
from efficientnet.tfkeras import EfficientNetB0 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.models import Model
import numpy as np
from efficientnet.tfkeras import EfficientNetB0 

'''
from tensorflow.keras.models import load_model

# Assuming 'best_model' is the path to your saved model
best_model = "path/to/your/best_model.h5"

# Define your model (replace this with your actual model architecture)
model = Sequential()
model.add(...)  # Add layers as per your architecture

# Load the best model weights
model.load_weights(best_model)

# Now, you can use the 'model' variable for predictions or further training

'''
input_size = 128
best_model='tmp_checkpoint/best_model.h5'
efficient_net = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (input_size, input_size, 3),
    include_top = False,
    pooling = 'max'
)
model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()
best_model=load_model(best_model)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

#loaded_model = tf.keras.models.load_model()
def predict_video(video_path):
    '''
    load_model.predict(video_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150,150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)'''
    base_path='C:/Users/Smile/Desktop/Hack/DeepFake-Detect/testing_video'
    if (video_path.endswith(".mp4")):
        tmp_path = os.path.join(base_path, get_filename_only(video_path))
        tmp_path=tmp_path.replace('\\','/')
        print('Creating Directory: ' + tmp_path)
        os.makedirs(tmp_path, exist_ok=True)
        print('Converting Video to Images...')
        count = 0
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(5)#frame rate
        while(cap.isOpened()):
            frame_id = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            print("asdfh")
            print(frame_id % math.floor(frame_rate))
            if (frame_id % math.floor(frame_rate) == 0):
                print('Original Dimensions: ', frame.shape)
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1
                print('Scale Ratio: ', scale_ratio)

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                print('Resized Dimensions: ', new_frame.shape)

                new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, get_filename_only(video_path)), count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
    print('Processing Directory: ' + tmp_path)
    print(os.listdir(tmp_path))
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x).replace('\\', '/'))]
    faces_path = os.path.join(tmp_path, 'faces').replace("\\", '/')
    print('Creating Directory: ' + faces_path)
    os.makedirs(faces_path, exist_ok=True)
    print('Cropping Faces from Images...')



    # Move the detector creation outside of the loop for efficiency
    detector = MTCNN()
    print("How it is going")
    print(frame_images)
    for frame in frame_images:
        print('Processing ', frame)
        image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        print('Face Detected: ', len(results))
        count = 0

        for result in results:
            bounding_box = result['box']
            print(bounding_box)
            confidence = result['confidence']
            print(confidence)
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3  # 30% as the margin
                margin_y = bounding_box[3] * 0.3  # 30% as the margin
                x1 = max(int(bounding_box[0] - margin_x), 0)
                x2 = min(int(bounding_box[0] + bounding_box[2] + margin_x), image.shape[1])
                y1 = max(int(bounding_box[1] - margin_y), 0)
                y2 = min(int(bounding_box[1] + bounding_box[3] + margin_y), image.shape[0])
                print(x1, y1, x2, y2)
                crop_image = image[y1:y2, x1:x2]
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)
                count = count + 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
            else:
                print('Skipped a face..')
    fakecount=0
    realcount=0
    l=[]
    for images in os.listdir(faces_path):
        if images.endswith(".png"):
            img_path = os.path.join(faces_path, images).replace("\\", "/")
            print("Image Path:", img_path)

            # Load the image directly from the face crop
            img_array = cv2.imread(img_path)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array=cv2.resize(img_array, (128, 128))
            img_array = img_array / 255.0  # Normalize the pixel values

            # Predict using the loaded model
            prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
            print(images, prediction)
            l.append(prediction)

            if prediction < 0.5:
                fakecount += 1
            else:
                realcount += 1

    if realcount > fakecount:
        return "The video is classified as real."
    else:
        return "The video is classified as fake."
       

# Example usage
#video_path = r'C:/Users/Smile/Desktop/Hack/DeepFake-Detect/testing_video/bmjmjmbglm.mp4'
#predict_video(video_path)