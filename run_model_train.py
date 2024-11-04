import os
import numpy as np
import json as js
import cv2

from model import VarEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

target_shape = (128, 128, 3)
json_params = {
    "input_shape": target_shape,
    "hiden_dim": 2,
    "encoder_out_activations": "tanh",
    "encoder_conv_filters": [32, 32, 64, 64],
    "encoder_conv_kernel_size": [3, 3, 3, 3],
    "encoder_conv_strides": [1, 2, 2, 1],
    "encoder_conv_padding": ["same", "same", "same", "same"],
    "encoder_conv_activations": ["linear", "linear", "linear", "linear"],
    "encoder_conv_dropout": [0.56, 0.56, 0.56, 0.56],
    "encoder_dense_units": [32, 32, 64, 64],
    "encoder_dense_dropout_rates": [0.56, 0.56, 0.56, 0.56],
    "encoder_dense_activations": ["linear", "linear", "linear", "linear"],

    "decoder_conv_filters": [32, 32, 64, 64],
    "decoder_conv_kernel_size": [3, 3, 3, 3],
    "decoder_conv_strides": [1, 2, 2, 1],
    "decoder_conv_padding": ["same", "same", "same", "same"],
    "decoder_conv_activations": ["linear", "linear", "linear", "linear"],
    "decoder_conv_dropout": [0.56, 0.56, 0.56, 0.56],
    "decoder_out_activations": "tanh",

    "weights_init": {
        "mean": 0.01,
        "stddev": 0.1
    }

}

generator = ImageDataGenerator()
with open("trainer_confs.json", "r") as json_f:
    gen_confs = js.load(json_f)


gen_data = {}
for train_part in gen_confs.keys():
    
    samples = []
    samples_ph = gen_confs[train_part]
    for sample_ph in os.listdir(samples_ph):

        image = cv2.imread(os.path.join(samples_ph, sample_ph))
        image = cv2.resize(image, (128, 128))
        image = image / 255.0

        samples.append(image)

    samples = np.asarray(samples)
    gen_data[train_part] = samples
     

    


print(gen_data["train_data_source"].shape)
model = VarEncoder(params_json=json_params)
model.train(train_tensor=gen_data["train_data_source"], train_labels=gen_data["train_data_source"], epochs=4, batch_size=32, run_folder="model_saves", epoch_per_save=1)


