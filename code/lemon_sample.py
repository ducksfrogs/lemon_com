import cv2
import numpy as np
import pandas as pd
import time

from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from albumentations import *
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_images_path_list = sorted(glob('../input/train_images/*.jpg'))
test_images_path_list = sorted(glob('../input/test_images/*.jpg'))


print("==================")
print("Training images: ")
print(len(train_images_path_list))
print(train_images_path_list[:5])
print("==================")
print("Validation:")
print(len(test_images_path_list))
print(test_images_path_list[:5])


image_0000 = Image.open("../input/train_images/train_0000.jpg")
plt.imshow(image_0000)

train_df = pd.read_csv("../input/train_images.csv")
train_df['class_num'].values_counts().plot.bar(figsize=(10,3),rot=0)


train_df['fold'] = 0

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold, (train_index, test_index) in enumerate(kf.split(train_df, train_df['class_num'])):
    print("FOLD{}".format(fold))
    train_df.loc[test_index, 'fold'] = fold


    
