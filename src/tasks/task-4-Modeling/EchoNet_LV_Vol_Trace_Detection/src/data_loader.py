import numpy as np
import pandas as pd
import PIL
import cv2
from constants import NUM_KEYPOINTS, IMAGE_SIZE, OUTPUT_DIR

def LoadData(input_dir, type='TRAIN'):
    all_images = []
    all_points = []
    all_ids = []
    for j, p in enumerate(input_dir.glob(f"*.csv")):
        df = pd.read_csv(p)
        try:
            df_type = df.Split.unique()[0]
        except AttributeError:
            print(df)
            break
        if df_type == type:
            for i, x in enumerate(df.Image):
                img = PIL.Image.open(input_dir.joinpath(x))
                #plt.imshow(img)
                #plt.show()
                v = df.iloc[i][:NUM_KEYPOINTS]
                if len(v) != 84:
                    continue
                all_points.append(v)
                img = cv2.resize(np.asarray(img), (IMAGE_SIZE, IMAGE_SIZE))
                all_images.append(img)
                all_ids.append(p.name.split('.')[0])
    all_images = np.asarray(all_images)
    all_points = np.asarray(all_points)
    all_points = all_points.reshape(-1, 1, 1, NUM_KEYPOINTS) / IMAGE_SIZE
    all_ids = np.asarray(all_ids)
    return all_images, all_points, all_ids
