import streamlit as st
import pandas as pd




def welcome_page():




  #st.set_page_config(page_title='Welcome', layout='wide')

  st.markdown(
      """
      <style>
          .block-container.css-18e3th9.egzxvld2 {
          padding-top: 0;
          }
          header.css-vg37xl.e8zbici2 {
          background: none;
          }
          span.css-10trblm.e16nr0p30 {
          color: #4477BB;
          }
          .css-1dp5vir.e8zbici1 {
          background-image: linear-gradient(
          90deg, rgb(130 166 192), rgb(74 189 130)
          );
          }
          .css-tw2vp1.e1tzin5v0 {
          gap: 10px;
          }
          h1#topeka-chapter {
          padding: 0;
          }
          h1#liverpool-chapter span.css-10trblm.e16nr0p30 {
          border-bottom: none;
          font-variant: inherit;
          }
          label.css-cgyhhy.effi0qh3, span.css-10trblm.e16nr0p30 {
          font-weight: bold;
          font-variant-caps: small-caps;
          border-bottom: 3px solid #4477BB;
          }
          div[data-testid="stSidebarNav"] li div a {
          margin-left: 1rem;
          padding: 1rem;
          width: 300px;
          border-radius: 0.5rem;
          }
          div[data-testid="stSidebarNav"] li div::focus-visible {
          background-color: rgba(151, 166, 195, 0.15);
          }
          svg.e1fb0mya1.css-fblp2m.ex0cdmw0 {
          width: 2rem;
          height: 2rem;
          }

      </style>
      """, unsafe_allow_html=True
  )

  st.markdown(
  """
  <style>
  body, h1, h2, h3, h4, h5, h6, p, span, label {
  font-family: 'Lilita One', cursive;
  }
  </style>
  """, unsafe_allow_html=True
  )

  col1, col2 = st.columns((1, 5))   
  with col1:
      #st.image("https://raw.githubusercontent.com/explainable-ai/lvef_assessment_app/main/resources/Omdena-Logo.png?raw=true")
      #st.image("https://github.com/explainable-ai/lvef_assessment_app/blob/9ea978de07fd1c27f19d9ebcb1af404f85cde6a6/resources/Favicon.png?raw=true")
      logo_url = "https://github.com/explainable-ai/lvef_assessment_app/blob/9ea978de07fd1c27f19d9ebcb1af404f85cde6a6/resources/Favicon.png?raw=true"

      st.image(logo_url, width=100)
  with col2:
      st.write('# Omdena Topeka Chapter')

  st.markdown('# Project Overview')
  st.markdown(""" The AI model developed in this project aims to automate the measurement of left ventricular ejection fraction (LVEF) during echocardiography. Manual measurement of LVEF is currently time-consuming and prone to variability, which can lead to inaccurate diagnoses and treatment plans. By automating this process, the workload of healthcare professionals can be reduced, and patient outcomes can be improved by enabling more precise diagnoses and personalized treatment plans.

  The resulting solution is an example of how AI can be integrated into clinical practice, which can lead to more widespread adoption of echocardiography for precision medicine. This can have significant benefits for patients, as echocardiography is a non-invasive and relatively low-cost imaging technique that can provide valuable information about the heart's structure and function. By improving the accuracy and efficiency of echocardiography, the AI model can help to make this technique more accessible and widely used, ultimately improving patient outcomes.""")


  st.markdown('## Video: What is Ejection Fraction?')
  col1, col2 = st.columns((1, 1))
  with col1:
      st.video("https://www.youtube.com/watch?v=lGfNjFFbrco")
  with col2:
      st.markdown('### Project Focus')
      st.markdown("""The project’s primary purpose is to accurately predict LVEF measurement.""")
      st.markdown("""* Develop and train an accurate AI model using the EchoNet-Dynamic dataset to predict LVEF measurement during echocardiography.""")
      st.markdown("""* Optimize the AI model to minimize error and variability in LVEF measurement predictions, using supervised and unsupervised learning techniques.""")
      st.markdown("""* Develop a user-friendly interface to integrate the AI model""")
      st.markdown('### Two-Stream Model')
      st.markdown("""A two-stream model is a deep learning model used to analyze echocardiogram videos by processing spatial and temporal information in parallel. It consists of two parallel neural networks that analyze each frame and motion between consecutive frames to capture cardiac dynamics. The model can automate diagnosis of various cardiac conditions with high accuracy.""")
      st.markdown('### Volume Tracing Detection Model')
      st.markdown("""The Volume Tracing Detection Model is a deep learning model that uses keypoint detection techniques to identify the volume within the left ventricle (LV) of the heart as a set of grid lines. The model is trained on the EchoNet-Dynamic dataset, which already contains tracings for the end-systolic (ES) and end-diastolic (ED) frames. The LV volume calculated at both phases is used to calculate the ejection fraction """)

