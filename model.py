import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json as js
import tensorflow.keras.backend as K

from PIL import Image
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    Reshape,
    LayerNormalization,
    BatchNormalization,
    Add,
)
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Masking,
    Bidirectional,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Flatten,
)
from tensorflow.keras.layers import (
    Concatenate,
    Lambda,
    Embedding,
    Multiply,
    Layer,
    Conv1D,
    Attention,
    RepeatVector,
)
from tensorflow.keras.losses import (
    SparseCategoricalCrossentropy,  
    CategoricalCrossentropy,
)
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.random import categorical
from tensorflow import map_fn, expand_dims, convert_to_tensor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import RandomNormal


class VarEncoder:

    def __init__(self, params_json=None, filepath=None) -> None:

        self.params_json = params_json

        if self.params_json is None:
            if filepath is None:
                raise ValueError("You must pass aither parasm_json or filepath !!!")

            self.load_params(filepath=filepath)

        self.encoder_conv_layers_n = len(self.params_json["encoder_conv_filters"])
        self.encoder_dense_layers_n = len(self.params_json["encoder_dense_units"])
        self.decoder_layers_n = len(self.params_json["decoder_conv_filters"])
        self.epoch_iterator = 0
        self.weights_init = RandomNormal(
            mean=self.params_json["weights_init"]["mean"],
            stddev=self.params_json["weights_init"]["stddev"],
        )
        self.model_loss = []

        self._build_encoder_()
        self._build_decoder_()
        self._build_model_()
        #self._load_hiden_dim_()

        self.save_params(filepath=filepath)

    def _load_hiden_dim_(self, json_log=None):

        if json_log is None:
            json_log = (
                "model_confs.json"
            )

        with open(json_log, "r") as json_file:
            self.hiden_dim = js.load(json_file)

        for class_label in self.hiden_dim.keys():
            self.hiden_dim[class_label] = np.asarray(self.hiden_dim[class_label])

    def _build_encoder_(self):

        def sampling(args):

            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
            return mu + K.exp(log_var / 2) * epsilon

        input_layer = Input(shape=self.params_json["input_shape"])
        encoder_layer = input_layer

        for layer_number in range(self.encoder_conv_layers_n):

            encoder_layer = Conv2D(
                filters=self.params_json["encoder_conv_filters"][layer_number],
                kernel_size=self.params_json["encoder_conv_kernel_size"][layer_number],
                padding="same",
                strides=self.params_json["encoder_conv_strides"][layer_number],
                kernel_initializer=self.weights_init,
            )(encoder_layer)

            encoder_layer = Activation(
                self.params_json["encoder_conv_activations"][layer_number]
            )(encoder_layer)
            encoder_layer = Dropout(
                rate=self.params_json["encoder_conv_dropout"][layer_number]
            )(encoder_layer)
            encoder_layer = BatchNormalization()(encoder_layer)

        self.saved_shape = encoder_layer.shape[1:]
        dense_layer = Flatten()(encoder_layer)

        for layer_number in range(self.encoder_dense_layers_n):

            dense_layer = Dense(
                units=self.params_json["encoder_dense_units"][layer_number],
                activation=self.params_json["encoder_dense_activations"][layer_number],
            )(dense_layer)
            dense_layer = Dropout(
                rate=self.params_json["encoder_dense_dropout_rates"][layer_number]
            )(dense_layer)

        mean_layer = Dense(
            units=self.params_json["hiden_dim"],
            activation=self.params_json["encoder_out_activations"],
        )(dense_layer)
        std_layer = Dense(
            units=self.params_json["hiden_dim"],
            activation=self.params_json["encoder_out_activations"],
        )(dense_layer)
        output_layer = Lambda(sampling)([mean_layer, std_layer])

        self.encoder = Model(input_layer, output_layer)

    def _build_decoder_(self):

        input_layer = Input(shape=(self.params_json["hiden_dim"],))
        rec_layer = Dense(units=np.prod(self.saved_shape))(input_layer)
        rec_layer = Reshape(target_shape=self.saved_shape)(rec_layer)

        decoder_layer = rec_layer
        for layer_number in range(self.decoder_layers_n):

            decoder_layer = Conv2DTranspose(
                filters=self.params_json["decoder_conv_filters"][layer_number],
                kernel_size=self.params_json["decoder_conv_kernel_size"][layer_number],
                padding="same",
                strides=self.params_json["decoder_conv_strides"][layer_number],
                kernel_initializer=self.weights_init,
            )(decoder_layer)

            decoder_layer = Activation(
                self.params_json["decoder_conv_activations"][layer_number]
            )(decoder_layer)
            decoder_layer = Dropout(
                rate=self.params_json["decoder_conv_dropout"][layer_number]
            )(decoder_layer)
            decoder_layer = BatchNormalization()(decoder_layer)

        output_layer = Conv2D(
            filters=self.params_json["input_shape"][-1],
            strides=1,
            padding="same",
            kernel_size=3,
        )(decoder_layer)
        output_layer = Activation(self.params_json["decoder_out_activations"])(
            output_layer
        )
        self.decoder = Model(input_layer, output_layer)

    def _build_model_(self):

        model_input_layer = Input(shape=self.params_json["input_shape"])
        model_output_layer = self.decoder(self.encoder(model_input_layer))
        self.model = Model(model_input_layer, model_output_layer)
        self.model.compile(
            loss="mse", metrics=["mae"], optimizer=RMSprop(learning_rate=0.01)
        )

    def train(
        self,
        run_folder,
        train_tensor,
        train_labels,
        epochs,
        batch_size,
        epoch_per_save=None,
        gen_encoded_sample=False,
    ):

        run_folder = run_folder
        if not os.path.exists(run_folder):
            os.mkdir(run_folder)
        entire_model_weights_folder = os.path.join(
            run_folder, "entire_model_weights.weights.h5"
        )

        for epoch in range(epochs):

            random_idx = np.random.randint(0, train_tensor.shape[0] - 1, batch_size)
            train_batch = train_tensor[random_idx]

            self.model_loss.append(self.model.train_on_batch(train_batch, train_batch))

            if epoch_per_save is not None:
                if epoch % epoch_per_save == 0:
                    self.save_samples(
                        samples_number=100, data_tensor=train_tensor, run_folder=run_folder
                    )

            self.epoch_iterator += 1

        if gen_encoded_sample:
            hiden_dim_json = os.path.join(run_folder, "hiden_dim_storage.json")
            self.generate_encoded_dim(
                data_tensor=train_tensor,
                general_labels=train_labels,
                hiden_dim_json=hiden_dim_json,
            )

        self.model.save_weights(filepath=entire_model_weights_folder)

    def generate_encoded_dim(self, general_labels, data_tensor, hiden_dim_json):

        encoded_vectors = self.encoder.predict(data_tensor)
        self.hiden_dim = {
            class_name: [] for class_name in self.params_json["classes_dis"].values()
        }

        for encoded_point, class_label in zip(encoded_vectors, general_labels):

            class_name = self.params_json["classes_dis"][class_label]
            self.hiden_dim[class_name].append(encoded_point.tolist())

        with open(hiden_dim_json, "w") as json_file:
            js.dump(self.hiden_dim, json_file)

    def load_weights(self, filepath=None):

        if filepath is None:
            filepath = "c:\\Users\\1\\Desktop\\models_save\\AE_save\\entire_model_weights.weights.h5"
        self.model.load_weights(filepath)

    def save_params(self, filepath=None):

        if filepath is None:
            filepath = "model_confs.json"

        with open(filepath, "w") as file:
            js.dump(self.params_json, file)

    def load_params(self, filepath):

        with open(filepath, "r") as file:
            self.params_json = js.load(file)

    def save_samples(
        self,
        samples_number,
        data_tensor,
        run_folder,
        save_as_animation=False,
        animation_frames=None,
    ):

        def _generate_show_tensor_(samples_number_sq, decoded_images):

            if not (self.params_json["input_shape"][-1] == 1):
                show_tensor = np.zeros(
                    (
                        samples_number_sq * self.params_json["input_shape"][0],
                        samples_number_sq * self.params_json["input_shape"][1],
                        self.params_json["input_shape"][-1],
                    )
                )

            else:
                show_tensor = np.zeros(
                    (
                        samples_number_sq * self.params_json["input_shape"][0],
                        samples_number_sq * self.params_json["input_shape"][1],
                    )
                )

            sample_number = 0
            for i in range(samples_number_sq):
                for j in range(samples_number_sq):

                    if not (self.params_json["input_shape"][-1] == 1):
                        show_tensor[
                            i
                            * self.params_json["input_shape"][0] : (i + 1)
                            * self.params_json["input_shape"][0],
                            j
                            * self.params_json["input_shape"][1] : (j + 1)
                            * self.params_json["input_shape"][1],
                            :,
                        ] = decoded_images[sample_number]

                    else:

                        curent_sample = decoded_images[sample_number]
                        curent_sample = np.squeeze(curent_sample, axis=2)

                        show_tensor[
                            i
                            * self.params_json["input_shape"][0] : (i + 1)
                            * self.params_json["input_shape"][0],
                            j
                            * self.params_json["input_shape"][1] : (j + 1)
                            * self.params_json["input_shape"][1],
                        ] = curent_sample

                    sample_number += 1

            return show_tensor

        gen_samples_folder = os.path.join(run_folder, "generated_samples")
        if not os.path.exists(gen_samples_folder):
            os.mkdir(gen_samples_folder)
        samples_number_sq = int(np.sqrt(samples_number))

        if not save_as_animation:

            fig, axis = plt.subplots()

            random_idx = np.random.randint(0, data_tensor.shape[0] - 1, samples_number)
            encoded_vectors = self.encoder.predict(data_tensor[random_idx])
            decoded_images = self.decoder.predict(encoded_vectors)

            curent_epoch_samples = os.path.join(
                gen_samples_folder, f"generated_samples_{self.epoch_iterator}.png"
            )
            show_tensor = _generate_show_tensor_(
                samples_number_sq=samples_number_sq, decoded_images=decoded_images
            )
            axis.imshow(show_tensor, cmap="inferno")
            fig.savefig(curent_epoch_samples)

        else:

            gen_samples_frames = os.path.join(
                gen_samples_folder, f"samples_frames{self.epoch_iterator}"
            )
            if not os.path.exists(gen_samples_frames):
                os.mkdir(gen_samples_frames)

            gen_animation_path = os.path.join(
                gen_samples_folder, f"generated_samples_{self.epoch_iterator}.gif"
            )
            for frame_n in range(animation_frames):

                frame_path = os.path.join(
                    gen_samples_frames, f"samples_frame_{frame_n}.png"
                )

                random_idx = np.random.randint(
                    0, data_tensor.shape[0] - 1, samples_number
                )
                encoded_data = self.encoder.predict(data_tensor[random_idx])
                decoded_images = self.decoder.predict(encoded_data)

                show_tensor = _generate_show_tensor_(
                    samples_number_sq=samples_number_sq, decoded_images=decoded_images
                )
                cv2.imwrite(frame_path, show_tensor)

            images = []
            for image_path in os.listdir(gen_samples_frames):

                image_path = os.path.join(gen_samples_frames, image_path)
                image = Image.open(image_path)
                images.append(image)

                images[0].save(
                    gen_animation_path,
                    save_all=True,
                    append_images=images,
                    optimize=False,
                    duration=100,
                    loop=0,
                )