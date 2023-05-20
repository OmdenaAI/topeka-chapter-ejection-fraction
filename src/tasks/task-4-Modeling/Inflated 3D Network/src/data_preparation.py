import os
import numpy as np
import pandas as pd
import cv2
import skvideo.io

def load_data():
    df = pd.read_csv("/kaggle/input/heartdatabase/EchoNet-Dynamic/FileList.csv")
    print(f"Total videos for training: {len(df)}")

    train_df = df[df["Split"] == "TRAIN"]
    train_files = df.FileName[df["Split"] == "TRAIN"]+'.avi'

    train_df = train_df.reset_index(drop=True)
    train_files = list(train_files)

    val_df = df[df["Split"] == "VAL"]
    val_files = df.FileName[df["Split"] == "VAL"]+'.avi'
    val_df = val_df.reset_index(drop=True)
    val_files = list(val_files)

    return train_df, train_files, val_df, val_files


def batch_generator(batch_size, gen_x): 
    batch_features = np.zeros((batch_size, 28, 112, 112))
    batch_labels = np.zeros((batch_size, 1)) 

    while True:
        for i in range(batch_size):
            batch_features[i], batch_labels[i] = next(gen_x)
        yield np.expand_dims(batch_features, 4), batch_labels


def generate_data(filelist, img_path, gt_df):
    while True:
        for i in filelist:
            if i.endswith(".avi"):
                img = skvideo.io.vread(img_path + '/' + i)[:, :, :, 0] 
                img = img[:28]
                resized_img = np.zeros((28, 112, 112))

                for j, k in enumerate(img):
                    resized_img[j, :, :] = cv2.resize(k, (112, 112), interpolation= cv2.INTER_LINEAR) / 255
                y = round(gt_df.EF[np.where(gt_df.FileName == i[:-4])[0][0]])

                yield resized_img, y
