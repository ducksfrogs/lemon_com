import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib

data_root_orig = '../input/train_images'

data_root = pathlib.Path(data_root_orig)



all_image_paths = list(data_root.glob('./*'))

import pandas as pd

train_labels = pd.read_csv("../input/train_images.csv")

all_image_paths = [str(path) for path in all_image_paths]

image_count = len(all_image_paths)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(t))

BATCH_SIZE = 32

ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
