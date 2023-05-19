# calculation_script.py

import math
import numpy as np
import pandas as pd

def calculate_disk_area(x1, y1, x2, y2):
    dist = np.linalg.norm(np.array((x1, y1)) - np.array((x2, y2)))
    r = dist/2
    area = np.pi * r * r
    return area
  
def calculate_volume(keypoints):
    '''
        keypoints: shape is [1, NUM_KEYPOINTS]
    '''
    ## first 4 is the long axis points
    x1, y1, x2, y2 = keypoints[0][0], keypoints[0][1], keypoints[0][2], keypoints[0][3]
    distance = np.linalg.norm(np.array((x1, y1)) - np.array((x2, y2)))
    height_of_disk = distance/20
    accumalated_areas = []
    for i in range(4, NUM_KEYPOINTS, 4):
        accumalated_areas.append(calculate_disk_area(keypoints[0][i], keypoints[0][i+1], 
                                                     keypoints[0][i+2], keypoints[0][i+3]))
        
    xa, ya, xb, yb = keypoints[0][4], keypoints[0][5], keypoints[0][6], keypoints[0][7]
    xc, yc, xd, yd = keypoints[0][8], keypoints[0][9], keypoints[0][10], keypoints[0][11]
    ## Calculate the distance between the 2 adjacent parallel lines. This will be alternate height of 
    ## the disk
    m = (yb-ya)/(xb-xa)
    c1 = yb - m*xb
    c2 = yd - m*xd
    alt_height_of_disk = abs(c1-c2)/math.sqrt(1+m*m)
    volume = sum(accumalated_areas)*height_of_disk
    return volume

def calculate_EF(ED_keypoints, ES_keypoints):
    '''
        ED_keypoints: shape [1, NUM_KEYPOINTS]
        ES_keypoints: shape [1, NUM_KEYPOINTS]
    '''
    ED_volume = calculate_volume(ED_keypoints)
    ES_volume = calculate_volume(ES_keypoints)
    EF = ((ED_volume - ES_volume) / ED_volume) * 100
    return EF

def calculate_EFs(data_keypoints):
    '''
    data_keypoints: shape [None, 1, 1, NUM_KEYPOINTS]
    '''
    total = data_keypoints.shape[0]
    data_EFs = []
    for i in range(0, total, 2):
        ED_kps = data_keypoints[i].reshape(-1, NUM_KEYPOINTS) * IMAGE_SIZE
        ES_kps = data_keypoints[i+1].reshape(-1, NUM_KEYPOINTS) * IMAGE_SIZE
        EF = calculate_EF(ED_kps, ES_kps)
        data_EFs.append(EF)
    return data_EFs

def build_dataframe_EFs(calculated_kps, predicted_kps):
    '''
        calculated_kps: shape [None, 1, 1, NUM_KEYPOINTS]
        predicted_kps: shape [None, 1, 1, NUM_KEYPOINTS]
    '''
    cal_efs = calculate_EFs(calculated_kps)
    pred_efs = calculate_EFs(predicted_kps)
    d = {'Actual_EF': cal_efs, 'Pred_EF': pred_efs}
    df = pd.DataFrame(data=d)
    act_lvef_class = []
    for i in df.Actual_EF:
        if i >= 50:
            act_lvef_class.append('Normal')
        elif i > 40:
            act_lvef_class.append('Mild')
        else:
            act_lvef_class.append('Abnormal')
    act_lvef_class = pd.Series(act_lvef_class, name='Actual_HFClass')
    act_lvef_class = act_lvef_class.astype('category')
    act_lvef_class = act_lvef_class.cat.set_categories(["Normal", "Mild", "Abnormal"], ordered=True)
    df['Actual_HFClass'] = act_lvef_class
    pred_lvef_class = []
    for i in df.Pred_EF:
        if i >= 50:
            pred_lvef_class.append('Normal')
        elif i > 40:
            pred_lvef_class.append('Mild')
        else:
            pred_lvef_class.append('Abnormal')
    pred_lvef_class = pd.Series(pred_lvef_class, name='Actual_HFClass')
    pred_lvef_class = pred_lvef_class.astype('category')
    pred_lvef_class = pred_lvef_class.cat.set_categories(["Normal", "Mild", "Abnormal"], ordered=True)
    df['Pred_HFClass'] = pred_lvef_class
    df['Diff_EFs'] = np.abs(df.Actual_EF - df.Pred_EF)
    return df

def get_predicted_points(data_images, model):
    '''
    data_images: shape [None, 112, 112, 3]
    '''
    data_kps = model.predict(data_images)
    return data_kps
