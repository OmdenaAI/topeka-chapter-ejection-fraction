import streamlit as st
import streamlit.components.v1 as components
import base64
import cv2
import skvideo.io
import tempfile
import subprocess
import os
import io
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from keras.utils.data_utils import get_file
import plotly.graph_objects as go
from matplotlib.patches import Arc
import matplotlib.pyplot as plt

from tensorflow_addons.layers import InstanceNormalization
from io import BytesIO
import moviepy.editor as mp




import requests

def upload():

  #Set page title and favicon
  #st.set_page_config(page_title="LVEF Assessment App - Assessment", page_icon=":heart:")

  # Define the EF ranges
  EF_RANGES = {
      "Normal": (55, 75),
      "Mildly reduced": (40, 54),
      "Moderately reduced": (30, 39),
      "Severely reduced": (20, 29),
      "Extremely reduced": (0, 19)
  }
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

  # Load the trained model
  @st.cache_data()

  def predict_ef(video_bytes):
    # Save the video bytes to display later
    video_bytes_to_display = video_bytes.read()

    # Create a temporary file for the video
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    temp_video_file.write(video_bytes_to_display)
    temp_video_file.close()

    # Convert to .mp4
    clip = mp.VideoFileClip(temp_video_file.name)
    mp4_file_path = temp_video_file.name + '.mp4'
    clip.write_videofile(mp4_file_path)
    with tempfile.TemporaryFile() as f:
        # Write the contents of the BytesIO object to a temporary file
        f.write(video_bytes.read())
        f.seek(0)

        # Convert video to frames
        cap = cv2.VideoCapture(f.name)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames_array = np.array(frames)
        img = frames_array[0:26]
        resized_img = np.zeros((28, 112, 112))
        for j, k in enumerate(img):
            resized_img[j, :, :] = cv2.resize(k, (112, 112), interpolation=cv2.INTER_LINEAR) / 255

        spatial_data = resized_img[:, :, :56]
        temporal_data = np.zeros((28, 112, 56))
        for j in range(1, 28):
            subtracted_frame = resized_img[j, :, 56:] - resized_img[j - 1, :, 56:]
            std = np.std(subtracted_frame)
            if std == 0:
                std = 1e-6 # set a small epsilon value if standard deviation is zero
            normalized_frame = (subtracted_frame - np.mean(subtracted_frame)) / std
            temporal_data[j, :, :] = normalized_frame

        temporal_data = temporal_data[1:27, :, :]


        # Make EF prediction using the trained model
        model_file = '/content/drive/MyDrive/LVEF_Streamlit/model/best_two_stream.h5'
        custom_objects = {'specificity': specificity, 'sensitivity': sensitivity, 'r2_score': r2_score}
        get_custom_objects().update(custom_objects)
        model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
        prediction = model.predict([spatial_data[np.newaxis, :, :, :, np.newaxis], 
                                      temporal_data[np.newaxis, :, :, :, np.newaxis]])[0][0]
        ef_prediction = int(prediction * 100)


        # Define the EF prediction ranges and their corresponding colors
        ranges = {'Normal': (70, 70, 'green'), 'Borderline': (49, 49, 'orange'), 'Reduced': (0, 40, 'red')}
        ef_range = None
        for range_name, (lower, upper, color) in ranges.items():
            if lower <= ef_prediction <= upper:
                ef_range = range_name
                ef_color = color
                break

        # Create a dictionary with the EF range names and values
        ef_dict = {'Predicted EF': ef_prediction}
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
        #fig = go.Figure(data=[go.Bar(x=list(ordered_ef_dict.keys()), y=list(ordered_ef_dict.values()), 
                                    #marker={'color': ['blue', 'green', 'orange', 'red']})])
        # Create a Streamlit columns layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
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
          fig.update_layout(title={'text': 'EF Measurement', 'font': {'size': 28, 'family': 'Lilita One'}}, 
                            xaxis={'title': 'EF Measurement', 'showticklabels': True, 'tickfont': {'size': 20, 'family': 'Lilita One'}},
                            yaxis={'title': 'EF Value (%)', 'showticklabels': True, 'tickfont': {'size': 20,'family': 'Lilita One'},'showline': False, 'range': [0, 100]},
          
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor = 'rgba(0,0,0,0)',
                            width=800, height=600)

          # Show the chart
          st.plotly_chart(fig)

        # Display the video in the right column
        with col2:
          #st.video(mp4_file_path)

          # Open the video file, read the bytes and encode them to base64
          with open(mp4_file_path, "rb") as video_file:
              video_bytes = video_file.read()
          base64_video = base64.b64encode(video_bytes).decode()

          # Create an HTML string that embeds the video
          video_html = f'''
          <video width="640" height="480" controls autoplay>
              <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
          </video>
          '''

          # Display the HTML in the app
          st.markdown(video_html, unsafe_allow_html=True)
        

        # Hide the upload widget
        #st.session_state['uploaded_file'] = None

        # Hide the upload video widget and prediction button
        #st.session_state['hide_widget'] = True

        # Show the predicted EF value and range
        st.write(f'<span style="font-family: Lilita One; font-size: 24px;">EF Prediction: {ef_prediction}%</span>', unsafe_allow_html=True)




        os.remove(temp_video_file.name)
        os.remove(mp4_file_path)
        return ef_prediction


  #Add page title
  st.markdown(
  """
  <style>
  .big-font {
  font-size:50px !important;
  }
  </style>
  """,
  unsafe_allow_html=True
  )

  #st.markdown('<p class="big-font" style="text-align: center; font-family: Lilita One;">LVEF Assessment</p>', unsafe_allow_html=True)
  st.markdown('<p class="big-font" style="text-align: center; font-family: Lilita One; font-size: 36px;">LVEF Assessment</p>', unsafe_allow_html=True)


  #Get uploaded file
  uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
  # Hide the upload widget once the bar chart is plotted
  upload_placeholder = st.empty()

  #Prediction button
  if uploaded_file is not None:

      ef_prediction = predict_ef(uploaded_file)


      # Remove the upload video widget
      uploaded_file = None

      # Remove the upload widget placeholder
      upload_placeholder.empty()

  #Add footer
  st.markdown(
      """
      <div class="icons">
          <a href="https://www.linkedin.com/company/omdena-topeka-chapter/" target="_blank"><button class="iconButton">LinkedIn</button></a>
          <a href="https://omdena.com" target="_blank"><button class="iconButton">Omdena</button></a>
      </div>
      """,
      unsafe_allow_html=True,
  )


  st.markdown(
      f"""
      <style>
      .stApp {{
          background-image: url("https://raw.githubusercontent.com/explainable-ai/lvef_assessment_app/main/resources/blank%20background.jpg");
          background-attachment: fixed;
          background-size: cover;
      }}
      .icons {{
          position: fixed;
          bottom: 20px;
          right: 20px;
          display: flex;
          align-items: center;
      }}
      .iconButton, .continueButton {{
          background-color: #4477BB !important;
          border: none !important;
          color: white !important;
          text-align: center;
          text-decoration: none;
          display: inline-block;
          font-family: 'Lilita One', cursive;
          font-size: 16px;
          font-weight: bold;
          border-radius: 40px !important;
          padding: 8px 16px;
          cursor: pointer;
          margin-left: 10px;
      }}
      .text-container {{
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 70vh;
      }}
      </style>
      """,
      unsafe_allow_html=True,
  )


