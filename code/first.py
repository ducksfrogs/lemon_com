import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib

data_root_orig = '../input/train_images'

data_root = pathlib.Path(data_root_orig)



all_image_paths = list(data_root.glob('./*'))

import pandas as pd

train_labels = pd.read_csv("../input/train_images.csv")
