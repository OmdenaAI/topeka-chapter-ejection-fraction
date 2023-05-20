import streamlit as st

def home_page():
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
          padding: 10px 20px;
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

  st.write("""
          <div class="text-container">
          <h1 style="font-size: 46px; font-weight: bold; margin-bottom: 30px;">Welcome to LVEF Assessment App</h1>
          <p style="font-size: 20px; line-height: 1.6; text-align:center;">This app is designed to help medical professionals assess left ventricular ejection fraction (LVEF), which is a measure of how well the heart is functioning. The app takes in data on a patient's echocardiogram and uses it to calculate LVEF. This can help doctors make more accurate diagnoses and better treatment plans.</p>
          </div>
          """,
          unsafe_allow_html=True)


  st.markdown(
      """
      <div class="icons">
          <a href="https://www.linkedin.com" target="_blank"><button class="iconButton">LinkedIn</button></a>
          <a href="https://omdena.com" target="_blank"><button class="iconButton">Omdena</button></a>
      </div>
      """,
      unsafe_allow_html=True,
  )
