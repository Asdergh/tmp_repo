import numpy as np
import tensorflow as tf
import torch as th
import matplotlib.pyplot as plt
import os
plt.style.use("dark_background")

from tensorflow.keras.layers import Input, Multiply, Dense, Activation, Conv2D, UpSampling2D, AvgPool2D, GlobalAveragePooling2D, Concatenate, Dense, BatchNormalization, Flatten
from tensorflow.keras import Model, Sequential
from tensorflow.keras.metrics import Mean
from tensorflow import Module, GradientTape
from tensorflow.keras.callbacks import Callback



class BottleNeck2D(Module):

    def __init__(self, filters, kernel_size=2, activation="relu"):

        conv = Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=1, 
            padding="same", 
            activation=activation
        )
        con = Concatenate()

        self.layers = [
            conv, 
            con
        ]
    
    def __call__(self, input):

        x = input
        x = self.layers[0](x)
        x = self.layers[1]([input, x])

        return x
    

class SeMap2D(Module):

    def __init__(self, ch, ratio=16):

        gap = GlobalAveragePooling2D()
        se_map = Dense(units=(ch // ratio), activation="relu")
        tanh_gate = Dense(units=ch, activation="softmax")
        out = Multiply()

        self.layers = [
            gap, 
            se_map, 
            tanh_gate, 
            out
        ]
    
    def __call__(self, input):

        x = input
        for layer in self.layers[:-1]:
            x = layer(x)
        
        x = self.layers[-1]([input, x])
        return x

class DownSample(Module):

    def __init__(self, filters, kernel_size, activation="silu"):

        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding="same")
        norm = BatchNormalization()
        #pool = AvgPool2D(pool_size=2)
        acti = Activation(activation)

        self.layers = [
            conv, 
            norm, 
            #pool, 
            acti
        ] 

    def __call__(self, input):

        x = input
        for layer in self.layers:
            x = layer(x)
        
        return x


class UpSample(Module):

    def __init__(self, filters, kernel_size, activation="sigmoid"):

        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")
        norm = BatchNormalization()
        up = UpSampling2D(size=2)
        acti = Activation(activation)

        self.layers = [
            conv,
            norm, 
            up, 
            acti
        ]
    
    def __call__(self, input):

        x = input
        for layer in self.layers:
            x = layer(x)
        
        return x

class Encoding2D(Module):

    def __init__(self, filters, kernel_size, activation):

        bottle_neck = BottleNeck2D(filters=filters, kernel_size=kernel_size)
        down_sample = DownSample(filters=filters, kernel_size=kernel_size, activation=activation)
        se_map = SeMap2D(ch=filters, ratio=16)

        self.layers = [
            down_sample,
            se_map
            ]
    
    def __call__(self, inputs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return x

class Decoding2D(Module):

    def __init__(self, filters, kernel_size, activation):

        bottle_neck = BottleNeck2D(filters=filters, kernel_size=kernel_size)
        up_sample = UpSample(filters=filters, kernel_size=kernel_size, activation=activation)
        se_map = SeMap2D(ch=filters, ratio=16)

        self.layers = [ 
            up_sample,
            se_map
            ]
    
    def __call__(self, inputs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return x




class AeModelCallback(Callback):


    def __init__(self, run_folder, model, data, input_sh=(128, 128, 3), sp_save=25, **kwargs):
        
        super().__init__(**kwargs)
        self.model_gen = model
        self.input_sh = input_sh
        self.data = data
        self.sp_save = sp_save

        self.run_folder = run_folder
        if not os.path.exists(self.run_folder):
            os.mkdir(self.run_folder)

        self.sample_n = 0

    def on_epoch_end(self, epoch, logs=None):

        rd_idx = np.random.randint(0, self.data.shape[0], self.sp_save)
        batch = self.data[rd_idx]
        gen = self.model_gen.predict(batch)

        spr = int(np.sqrt(self.sp_save))
        fig, axis = plt.subplots()
        show_pipline = np.zeros(shape=(
            spr * self.input_sh[0],
            spr * self.input_sh[1],
            self.input_sh[-1]
        ))

        sn = 0
        for i in range(spr):
            for j in range(spr):

                show_pipline[i * self.input_sh[0]: (i + 1) * self.input_sh[0],
                             j * self.input_sh[1]: (j + 1) * self.input_sh[1], 
                             :] = gen[sn]
                sn += 1
        
        axis.imshow(show_pipline)
        gen_path = os.path.join(self.run_folder, "generated_samples")
        if not os.path.exists(gen_path):
            os.mkdir(gen_path)

        fig.savefig(fname=os.path.join(gen_path, f"gen_sample{self.sample_n}.png"))
        self.sample_n += 1


    def on_train_end(self, logs=None):

        weigths_path = os.path.join(self.run_folder, "model.weights.h5")
        model_path = os.path.join(self.run_folder, "model.keras")

        self.model_gen.save_weights(weigths_path)

        
class AeVersion1(Model):

    def __init__(self, input_sh, **kwargs):

        super().__init__(**kwargs)
        self.input_sh = input_sh
        self.model = self._build_model()
    
    def _build_model(self):

        input_layer = Input(shape=self.input_sh)

        en_layer = Encoding2D(filters=128, kernel_size=3, activation="linear")(input_layer)
        en_layer = Encoding2D(filters=64, kernel_size=3, activation="linear")(en_layer)
        en_layer = Encoding2D(filters=32, kernel_size=3, activation="linear")(en_layer)

        dec_layer = Decoding2D(filters=128, kernel_size=3, activation="linear")(en_layer)
        dec_layer = Decoding2D(filters=64, kernel_size=3, activation="linear")(dec_layer)
        dec_layer = Decoding2D(filters=32, kernel_size=3, activation="linear")(dec_layer)

        out = Conv2D(filters=self.input_sh[-1], kernel_size=1, strides=1, activation="linear")(dec_layer)
        return Model(inputs=input_layer, outputs=out)
    
    @property
    def metrics(self):

        return [
            self.model_loss_tracker
        ]
    
    def compile(self, optimizer, loss_fn):

        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model_loss_tracker = Mean(name="model_loss_tracker")
    
    
    @tf.function
    def call(self, inputs):
        return self.model(inputs)
    
    def train_step(self, inputs):

        train_sample, labels = inputs
        with GradientTape() as gr_tape:

            preds = self.model(train_sample)
            loss = self.loss_fn(labels, preds)
        
        tr_vars = self.model.trainable_variables
        grads = gr_tape.gradient(loss, tr_vars)
        self.optimizer.apply_gradients(zip(grads, tr_vars))
        
        self.model_loss_tracker.update_state(loss)
        return {
            "model_loss": self.model_loss_tracker.result()
        }



    

        