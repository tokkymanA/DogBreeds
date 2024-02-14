import tensorflow as tf
import numpy as np 
import pandas as pd
import tensorflow_hub as hub
import os

os.environ['F_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = "Model/20240204-$H0804-full-train1-mobilenetV2.keras"
LABELS_PATH = "CSV/labels.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 32

class Iden():

    def __init__(self, input_byte):

        self.input_byte = input_byte
        self.file = pd.read_csv(LABELS_PATH)
        self.unique_breed = np.unique(np.array(self.file["breed"]))
        self.LoadModel = tf.keras.models.load_model(MODEL_PATH, custom_objects={"KerasLayer":hub.KerasLayer})
        print("Error Load")
        self.make_input = self.create_data_batches([self.input_byte])
        self.prediction = self.LoadModel.predict(self.make_input)
        self.result = (self.unique_breed[self.prediction[0].argmax()], np.max(self.prediction[0]), self.make_input)

    def process_image(self, byte_data, image_size=IMAGE_SIZE):

        # Turn byte_data into numerical tensor
        image = tf.io.decode_jpeg(contents=byte_data, channels=3) 
        # convert the color channel values from 0-255 into 0-1
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
        # resize the image to our desired value (244, 244)
        image = tf.image.resize(images=image, size=[image_size, image_size])

        return image
        
    def create_data_batches(self, byte_data:list):

        print("Batch data...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(byte_data)))
        data_batch = data.map(self.process_image).batch(BATCH_SIZE)
        return data_batch


        



    
