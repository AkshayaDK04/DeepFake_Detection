import os,re
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import custom_object_scope

# Define the FixedDropout layer
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)
    
    def call(self, inputs, training=None):
        return super(FixedDropout, self).call(inputs, training=False)

# Load the model with the custom object scope
with custom_object_scope({'FixedDropout': FixedDropout}):
    model = load_model('tmp_checkpoint/best_model.h5')

# Function to predict if an image is a deepfake
def predict_image(image_path):
    # Resize the image to match the input shape expected by the model
    img = load_img(image_path, target_size=(128, 128))  # Corrected target size
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    probability = prediction[0][0]

    print(f"Probability of being a deepfake: {probability * 100:.2f}%")
    if probability < 0.5:
        return "Image is likely real."
    else:
        return "Image is likely a deepfake."

# Example usage
#image_path_real = r'C:\Users\dhaks\OneDrive\Pictures\Saved Pictures\original1.jpeg'
# image_path_fake = r'C:\Users\dhaks\OneDrive\Pictures\Saved Pictures\fake1.jpeg'

#print(predict_image(image_path_real))
# print(predict_image(image_path_fake))
