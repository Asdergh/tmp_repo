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
    "0.0": {
        "model": VarEncoder,
        "epochs": 100,
        "batch_size": 32,
        "epoch_per_save": 10,
        "params_path": "C:\\Users\\1\\Desktop\\tmp_model\\model_confs.json"
    },

    "1.0": {
        "model": AeVersion1,
        "epochs": 100,
        "batch_size": 32
    }
}
with open("trainer_confs.json", "r") as json_f:
    training_confs = js.load(json_f)


gen_data = {}
for train_part in training_confs.keys():
    
    if train_part not in ["model_version", "pretrained"]:
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
    



epochs = MODELS[training_confs["model_version"]]["epochs"]
batch_size = MODELS[training_confs["model_version"]]["batch_size"]

if training_confs["model_version"] == "1.0":        

    model = MODELS[training_confs["model_version"]]["model"](input_sh=(128, 128, 3))
    if training_confs["pretrained"] == "true":
        model.load_weights("C:\\Users\\1\\Desktop\\tmp_model\\model_version_1_0\\model.weights.h5")

    callback = AeModelCallback(data=gen_data["train_data_source"], 
                           model=model, input_sh=(128, 128, 3), 
                           run_folder="C:\\Users\\1\\Desktop\\tmp_model\\model_version_1_0")
    
    model.compile(optimizer=Adam(learning_rate=0.01), loss_fn=MeanSquaredError())
    model.fit(gen_data["train_data_source"], 
            gen_data["train_data_source"], 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[callback])

elif training_confs["model_version"] == "0.0":

    params_path = MODELS[training_confs["model_version"]]["params_path"]
    epochs_per_save = MODELS[training_confs["model_version"]]["epoch_per_save"]

    model = MODELS[training_confs["model_version"]]["model"](filepath=params_path)
    if training_confs["pretrained"]:
        model.load_weights(filepath="C:\\Users\\1\\Desktop\\tmp_model\\model_saves\\entire_model_weights.weights.h5")

    model.train(run_folder="C:\\Users\\1\\Desktop\\tmp_model\\model_saves",
            train_tensor=gen_data["train_data_source"],
            train_labels=gen_data["train_data_source"],
            epochs=epochs,
            batch_size=batch_size,
            epoch_per_save=epochs_per_save,
            gen_encoded_sample=False)


    




