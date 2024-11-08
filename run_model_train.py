import os
import numpy as np
import json as js
import cv2

from model import VarEncoder
from model_1 import AeVersion1, AeModelCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam



MODELS = {
    "0.0": VarEncoder,
    "1.0": AeVersion1
}
generator = ImageDataGenerator()
with open("trainer_confs.json", "r") as json_f:
    training_confs = js.load(json_f)


gen_data = {}
for train_part in training_confs.keys():
    
    if train_part != "model_version":
        samples = []
        samples_ph = training_confs[train_part]
        for sample_ph in os.listdir(samples_ph):

            image = cv2.imread(os.path.join(samples_ph, sample_ph))
            image = cv2.resize(image, (128, 128))
            image = image / 255.0
            image = (image - np.mean(image)) / np.std(image)

            samples.append(image)

        samples = np.asarray(samples)
        gen_data[train_part] = samples
    


model = MODELS[training_confs["model_version"]](filepath="C:\\Users\\1\\Desktop\\tmp_model\\model_confs.json")
callback = AeModelCallback(data=gen_data["train_data_source"], 
                           model=model, input_sh=(128, 128, 3), 
                           run_folder="C:\\Users\\1\\Desktop\\tmp_model\\model_version_1_0")
# model.compile(optimizer=Adam(learning_rate=0.1), loss_fn=MeanSquaredError())
# print(model.model.summary())
# model.fit(gen_data["train_data_source"], 
#           gen_data["train_data_source"], 
#           epochs=100, 
#           batch_size=32)


run_folder = "C:\\Users\\1\\Desktop\\tmp_model\\model_saves"
train_tensor = gen_data["train_data_source"]
train_labels = gen_data["train_data_source"]
epochs = 100
batch_size = 32
epoch_per_save=10

model.train(run_folder,
        train_tensor,
        train_labels,
        epochs,
        batch_size,
        epoch_per_save,
        gen_encoded_sample=False,)


    




