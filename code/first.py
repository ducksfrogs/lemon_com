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

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3), include_top=False)
mobile_net.trainable = False

def change_range(image, label):
    return 2*image-1, label

keras_ds = ds.map(change_range)
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4)
])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max Ligit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
steps_per_epoch

import time

default_timeit_steps = 2*steps_per_epoch+1

def timeit(ds, steps=default_timeit_steps):
    overall_start = time.time()
    it = iter(ds.take(steps+1))
    next(it)

    start = time.time()
    for i,(images, labels) in enumerate(it):
        if i%10 == 0:
            print('.', end=' ')
    print()
    end = time.time()
    duration = end - start

    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
    print("Total time: {}s".format(end-overall_start))


ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))

ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds

timeit(ds)

ds = image_label_ds.cache()
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))

ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


timeit(ds)

ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)
ds

timeit(ds)
