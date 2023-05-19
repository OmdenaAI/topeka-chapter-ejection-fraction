import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomCrop, RandomFlip, RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, concatenate, Dropout
from tqdm import tqdm 
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1-y_true)*y_pred, 0, 1)))
    return true_negatives / (true_negatives + false_positives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true*(1-y_pred), 0, 1)))
    return true_positives / (true_positives + false_negatives + K.epsilon())

def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

def step_decay(epoch):
    initial_lr = 0.00001
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * math.pow(drop, math.floor((epoch)/epochs_drop))
    print(f"Epoch: {epoch+1}, Learning rate: {lr}")
    return lr
