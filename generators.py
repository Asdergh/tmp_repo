import os
import cv2
import yaml
import random as rd 

from tensorflow.keras.utils import Sequence


class DenImageGenerator(Sequence):


    def __init__(self, yaml_conf, us_type="train"):
        
        with open(yaml_conf, "r") as ym:
            self.conf = yaml.load(ym)
        
        self.data_source = os.path.join(self.conf["datafolder"], us_type)
        self.images_ph = os.listidr(self.data_source, "image")
        self.labels_ph = os.listidr(self.data_source, "labels")
    
    def __len__(self):

        return sum([os.listdir(os.path.join(self.conf["datafolder"], batch)) for batch in [
            "train",
            "test",
            "validation"
        ]])

    def __getitem__(self, idx):

        
        if not self.conf["shuffle"]:    

            batch = [(image, label) for image, label in zip(
                os.path.join()
            )]
        
        else:

            images_batch = rd.sample(os.listdir(self.images_ph))
            labels_batch = rd.sample(os.listdir(self.labels_ph))

        
        for ()
        