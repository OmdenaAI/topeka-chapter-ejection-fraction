import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from keras.layers import Input, Dropout, SeparableConv2D, Dense, Flatten
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('/content/drive/MyDrive/LVEF_Streamlit/model/WeightsLV_Cavity_Volume_Trace-0030.h5', compile=False)

def cal_disk_area(x1,y1,x2,y2):
    dist = np.linalg.norm(np.array((x1,y1)) - np.array((x1,y2)))
    r = dist/2
    area = np.pi * r * r 
    return area

def cal_vol2(KP):
    num_KP = 84
    x1, y1, x2, y2 = KP[0], KP[1], KP[2], KP[3]
    distance = np.linalg.norm(np.array((x1, y1)) - np.array((x2, y2)))
    disk_height = distance / 20
    accumulated_areas = []
    for i in range(4, num_KP, 4):
        accumulated_areas.append(cal_disk_area(KP[i], KP[i + 1], KP[i + 2], KP[i + 3]))
    xa, ya, xb, yb = KP[4], KP[5], KP[6], KP[7]
    xc, yc, xd, yd = KP[8], KP[9], KP[10], KP[11]
    m = (yb - ya) / (xb - xa)
    c1 = yb - m * xb
    c2 = yd - m * xd
    alt_height_disk = abs(c1 - c2) / math.sqrt(1 + m * m)
    volume = sum(accumulated_areas) * disk_height
    return volume

def cal_vol(KP):
    num_KP = 84

    # Reshape the KP array to remove the extra dimension
    KP = KP.reshape(-1)

    # Check if the KP array has enough elements
    if len(KP) < num_KP:
        raise ValueError("Invalid number of keypoints")

    # Access the keypoints using the correct indices
    x1, y1, x2, y2 = KP[0], KP[1], KP[2], KP[3]
    distance = np.linalg.norm(np.array((x1, y1)) - np.array((x2, y2)))

    disk_height = distance / 20
    accumulated_areas = []
    for i in range(4, num_KP, 4):
        accumulated_areas.append(cal_disk_area(KP[i], KP[i + 1], KP[i + 2], KP[i + 3]))
    xa, ya, xb, yb = KP[4], KP[5], KP[6], KP[7]
    xc, yc, xd, yd = KP[8], KP[9], KP[10], KP[11]
    m = (yb - ya) / (xb - xa)
    c1 = yb - m * xb
    c2 = yd - m * xd
    alt_height_disk = abs(c1 - c2) / math.sqrt(1 + m * m)
    volume = sum(accumulated_areas) * disk_height

    return volume




def cal_EF(ED_KP,ES_KP):
    ED = cal_vol(ED_KP)
    ES = cal_vol(ES_KP)
    EF = ((ED - ES) / ED) * 100
    return EF

def calculate_EF(prediction):
    num = prediction.shape[0]
    num_KP = prediction.shape[3]
    IMG_SIZE = 112
    data_EF = []

    
    print("Prediction Shape:", prediction.shape)
    print("Num:", num)
    print("Num_KP:", num_KP)

    for i in range(0,num,2):
        ED_KP = prediction[i].reshape(-1, num_KP) * IMG_SIZE  
        ES_KP = prediction[i+1].reshape(-1, num_KP) * IMG_SIZE   

        EF = cal_EF(ED_KP,ES_KP)
        data_EF.append(EF)
    return data_EF

def predict(imgs):
    t_img=tf.convert_to_tensor(imgs)
    pred=model.predict(t_img)
    return pred

def severity(EF):
    EV = np.array(EF)
    if(EV>=50.0):
        return ('Normal')
    elif(EV>40.0 and EV<50.0):
        return ('Mild')
    elif(EV<40.0):
        return ('Abnormal')

def preprocess(dst, st):
    # Convert numpy arrays to tensors
    dst = tf.convert_to_tensor(dst, dtype=tf.float32)
    st = tf.convert_to_tensor(st, dtype=tf.float32)

    # Resize the images
    dst_img = tf.image.resize(dst, [112, 112])
    syst_img = tf.image.resize(st, [112, 112])
    
    p_img = np.array([dst_img, syst_img])
    prediction = predict(p_img)
    cal_EF = calculate_EF(prediction) 
    condition = severity(cal_EF)

    return cal_EF, condition

def preprocess2(dst, st):
    # Convert numpy arrays to tensors
    dst = tf.convert_to_tensor(dst, dtype=tf.float32)
    st = tf.convert_to_tensor(st, dtype=tf.float32)

    # Resize the images
    dst_img = tf.image.resize(dst, [112, 112])
    syst_img = tf.image.resize(st, [112, 112])
    
    # Predict keypoints
    data_images = np.array([dst_img, syst_img])
    predicted_kps = get_predicted_points(data_images, model)
    
    # Calculate volume
    volume = cal_vol(predicted_kps)
    
    # Plot the volume tracing over the image
    plt.figure(figsize=(10, 10))
    plt.imshow(dst_img)
    plt.plot(volume, color='red')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Volume Tracing')
    plt.show()
    
    # Calculate EF and condition
    cal_EF = calculate_EF(predicted_kps)
    condition = severity(cal_EF)
    
    return cal_EF, condition



import plotly.graph_objects as go

def run_trace():
    import streamlit as st
  
    # Setup file upload
    dst_image = st.sidebar.file_uploader("Upload DST image", type=["png", "jpg", "jpeg"])
    syst_image = st.sidebar.file_uploader("Upload ST image", type=["png", "jpg", "jpeg"])

    st_button = st.sidebar.button("Calculate EF")

    if st_button:
        # Upload systolic image
        if syst_image is not None:
            syst_image = Image.open(syst_image).convert("RGB")
            syst_img = np.array(syst_image)

        # Upload diastolic image
        if dst_image is not None:
            dst_image = Image.open(dst_image).convert("RGB")
            dst_img = np.array(dst_image)
    
        # Preprocess the images and get EF and condition
        if dst_image is not None and syst_image is not None:
            cal_EF, condition = preprocess(dst_img, syst_img)

            # Ensure that ef is a single value, not a list
            #ef = ef[0] * 100 if isinstance(ef, list) and len(ef) == 1 else ef
            ef = cal_EF[0] * 100 if isinstance(cal_EF, list) and len(cal_EF) == 1 else cal_EF

            
            # Display the images
            st.image(dst_image, use_column_width=0.5, caption="DST Image")
            st.image(syst_image, use_column_width=0.5, caption="ST Image")

        
            
            st.write("Ejection Fraction (EF): ", ef)
            #st.write("Condition: ", condition)

  


            # Define the EF prediction ranges and their corresponding colors
            ranges = {'Normal': (70, 70, 'green'), 'Borderline': (49, 49, 'orange'), 'Reduced': (0, 40, 'red')}
            ef_range = None
            for range_name, (lower, upper, color) in ranges.items():
                if lower <= ef <= upper:
                    ef_range = range_name
                    ef_color = color
                    break

            # Create a dictionary with the EF range names and values
            ef_dict = {'Predicted EF': ef}
            for range_name, (lower, upper, color) in ranges.items():
                if range_name == 'Normal':
                    ef_dict[range_name] = lower
                elif range_name == 'Borderline':
                    ef_dict[range_name] = lower - 1
                else:
                    ef_dict[range_name] = upper

            # Define the data labels for each EF range
            data_labels = {
                'Normal': 'NORMAL Ejection Fraction<br>≈50–70% is pumped out during<br>each contraction (Usually<br>comfortable during activity.)',
                'Borderline': 'BORDERLINE Ejection Fraction<br>≈41–49% is pumped out<br>during each contraction<br>(Symptoms may become<br>noticeable during activity.)',
                'Reduced': 'REDUCED Ejection Fraction<br>≤40% is pumped out during<br>each contraction (Symptoms<br>may become noticeable<br>even during rest.)'
            }

            # Set the order of the bars
            ordered_ef_dict = {k: ef_dict[k] for k in ['Predicted EF', 'Normal', 'Borderline', 'Reduced']}

            # Create the bar chart
            fig = go.Figure(data=[go.Bar(
                x=list(ordered_ef_dict.keys()),
                y=list(ordered_ef_dict.values()),
                marker={'color': ['blue', 'green', 'orange', 'red']},
                text=[data_labels.get(k, '') for k in ordered_ef_dict.keys()],
                textfont={'size': 18, 'family': 'Lilita One'},
                textposition='outside',
                hovertemplate='%{y:.2f}%<br>%{text}'
            )])

            # Customize the chart layout
            fig.update_layout(
                title={'text': 'EF Measurement', 'font': {'size': 28, 'family': 'Lilita One'}},
                xaxis={'title': 'EF Measurement', 'showticklabels': True, 'tickfont': {'size': 20, 'family': 'Lilita One'}},
                yaxis={'title': 'EF Value (%)', 'showticklabels': True, 'tickfont': {'size': 20,'family': 'Lilita One'},'showline': False, 'range': [0, 100]},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor = 'rgba(0,0,0,0)',
                width=800, height=600)
                
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            # Call preprocess function to visualize the volume tracing plot
            #cal_EF, condition = preprocess(dst_img, syst_img)

        else:
            st.warning("Please upload both DST and ST images.")
    else:
      st.warning("Please click 'Calculate EF' button.")
