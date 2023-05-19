import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import PIL
import cv2
import os
import shutil
import tempfile
from pathlib import Path
import pprint
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


INFO_FILE_NAME = 'FileList.csv'
VOL_TRACE_FILE_NAME = 'VolumeTracings.csv'
VIDEO_DIR_NAME = 'Videos'
OUTPUT_DIR_PATH = '/kaggle/working/Output'
ECHONET_DATA_DIR = 'heartdatabase/EchoNet-Dynamic'

# Define the platform data directory paths
data_directories = {
    'kaggle': '/kaggle/input',
    'google_colab': '/content/drive/MyDrive/LVEF',
    'local': '/opt/Data',
}

# Get the current platform from the environment variable
platform = os.getenv('PLATFORM')

# Set the data directory based on the current platform
data_directory = data_directories.get(platform)

# Mount Google Drive if we're on Google Colab
if platform == 'google_colab':
    from google.colab import drive
    drive.mount('/content/drive')

# Set the DATA_DIRECTORY environment variable to the determined data directory
os.environ['DATA_DIRECTORY'] = data_directory

# Load the data
directory = os.environ.get("DATA_DIRECTORY")
ROOT_DIR = Path(tempfile.mkdtemp()) if directory is None else Path(directory)
print(ROOT_DIR)

## Code to clean kaggle output folders
def remove_folder_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                continue
        except Exception as e:
            print(e)

def checkPathExists(path):
  if not os.path.exists(path):
    print(f"Cannot access path: {path}")
  else:
    print (f"Path {path} accessible")
    
def checkPathExists(path):
  if not os.path.exists(path):
    logging.warning(f"Cannot access path: {path}")
  else:
    logging.info(f"Path {path} accessible")

ECHONET_DATA_DIR = ECHONET_DATA_DIR

pp = pprint.PrettyPrinter()
DATA_DIR = ROOT_DIR.joinpath(ECHONET_DATA_DIR)
checkPathExists(DATA_DIR)

INFO_FILE = DATA_DIR.joinpath(INFO_FILE_NAME)
VOL_TRACE_FILE = DATA_DIR.joinpath(VOL_TRACE_FILE_NAME)
checkPathExists(INFO_FILE)
checkPathExists(VOL_TRACE_FILE)

INFO_DF = pd.read_csv(INFO_FILE)
VOL_TRACE_DF = pd.read_csv(VOL_TRACE_FILE)

logging.info(INFO_DF.Split.value_counts())

## To extract the ED and ES frame from the video file
def extractEDandESframes(image_file, ED_frame_number, ES_frame_number):
    video = cv2.VideoCapture(str(image_file))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ## Retrieve the ED frame
    for i in range(ED_frame_number-1):
        ret, frame = video.read()
    res, ED_frame = video.read()
    ## Retrieve the ES frame
    diff = ES_frame_number - ED_frame_number
    for i in range(diff):
        ret, frame = video.read()
    res1, ES_frame = video.read()
    if not res:
        logging.warning("issue ED")
    if not res1:
        logging.warning("issue ES")
    if res&res1:
        return ED_frame, ES_frame
    else:
        return None, None

## Save the ED and ES frame as a png which is prefixed with the original avi file name
## Save a csv file name is the same as orginal avi file name, that has the volume 
## tracings for both the ES and ED file
## the csv file also has a column with Image file name, and the Split value
## from the original file list.
def saveEDandESimages(data_dir, output_dir, info_df, trace_df):
    patient_list = [x for x in data_dir.iterdir()]
    
    for i, file in enumerate(patient_list):
        patient_id = file.name.split('.')[0]
        frame_df = trace_df.query(f"FileName == '{file.name}'")
        try:
            ed_number, es_number = frame_df.Frame.unique()
        except:
            logging.error(f"This {file} generated an error")
            continue
        split_value = info_df.query(f"FileName == '{patient_id}'").Split
        #print(ed_number, es_number)
        ED_frame, ES_frame = extractEDandESframes(file, ed_number, es_number)
        if ED_frame is not None or ES_frame is not None:
            ## Write the ED and ES frames as images
            iED_path = output_dir.joinpath(f"{patient_id}_ED.png")
            iES_path = output_dir.joinpath(f"{patient_id}_ES.png")
            cv2.imwrite(str(iED_path), ED_frame)
            cv2.imwrite(str(iES_path), ES_frame)
            ## Write the trac points into a csv file
            ED_info = frame_df.query(f'FileName =="{file.name}" and Frame == {ed_number}').reset_index(drop=True)
            ES_info = frame_df.query(f'FileName =="{file.name}" and Frame == {es_number}').reset_index(drop=True)
            ES_info = frame_df.query(f'FileName =="{file.name}" and Frame == {es_number}').reset_index(drop=True)
            ED_stack = np.hstack(ED_info[['X1', 'Y1', 'X2', 'Y2']].values).tolist()
            ES_stack = np.hstack(ES_info[['X1', 'Y1', 'X2', 'Y2']].values).tolist()
            keypoint_df = pd.DataFrame([ED_stack, ES_stack])
            keypoint_df['Image'] = [f"{patient_id}_ED.png", f"{patient_id}_ES.png"]
            keypoint_df['Split'] = [split_value.iloc[0], split_value.iloc[0]]
            keypoint_df.to_csv(output_dir.joinpath(f"{patient_id}.csv"), index=False)
        else:
            logging.warning(f"There was an issue with processing {file}")

VIDEO_DIR = DATA_DIR.joinpath(VIDEO_DIR_NAME)
OUTPUT_DIR = Path(OUTPUT_DIR_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkPathExists(OUTPUT_DIR)

saveEDandESimages(VIDEO_DIR, OUTPUT_DIR, INFO_DF, VOL_TRACE_DF)

