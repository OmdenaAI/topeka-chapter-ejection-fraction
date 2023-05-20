import streamlit as st
import hydralit_components as hc
from instructions_page import instructions
from src.home_page import home_page
from src.about import about
from src.upload_page import upload
from src.welcome import welcome_page
from src.volume_trace import run_trace




# Set page title and favicon
st.set_page_config(layout='wide', page_title="Welcome to LVEF Assessment App - Assessment", page_icon=":heart:")

st.markdown(
    """
    <style>
      .stApp {
        background-image: url('https://raw.githubusercontent.com/explainable-ai/lvef_assessment_app/main/resources/blank%20background.jpg');
        background-repeat: no-repeat;
        background-size: cover;
        background-position: center center;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
      .nav-wrapper {
        background-color: #4477BB;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


menu_data = [
    {'id': 'Home', 'icon': "🏠", 'label': "Home"},
    {'id': 'Welcome', 'icon': '📕', 'label': "Project Overview"},
    #{'id': 'Instructions', 'icon': 'info', 'label': "Instructions"},
    {'id': 'Upload Page', 'icon': '📹', 'label': "Video Assessment"},
    {'id': 'Volume Trace', 'icon': "📷", 'label': "Image Assessment"},
    #{'id': 'Welcome', 'icon': '👋', 'label': "Welcome"}
]


#override_theme={'nav': {'margin-top': '20px','menu_background': '#4477BB'}}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    home_name=None,
    override_theme= {'txc_inactive': '#FFFFFF', 'menu_background' : '#4477BB'} ,
    login_name=None,
    hide_streamlit_markers=True,
    sticky_nav=True,
    sticky_mode='jumpy',
)


if menu_id == "Home":
    home_page()
elif menu_id == "Welcome":
    welcome_page()
elif menu_id == "Upload Page":
    upload()
elif menu_id == "Volume Trace":
    run_trace()   
#elif menu_id =="Welcome":
    #welcome_page()
