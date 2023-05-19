
<h2> Echocardiogram Video Frame Extraction and Tracing Analysis</h2>

This Python script is designed to process a dataset of echocardiogram videos and accompanying tracing information. The main objectives are to extract End-Diastolic (ED) and End-Systolic (ES) frames from each video and save them as individual image files, and to analyze tracing data corresponding to these frames.

<h2>Dependencies</h2>
The script relies on several third-party Python libraries, including pandas, numpy, matplotlib, SimpleITK, PIL, cv2, os, shutil, tempfile, pathlib, and pprint.

<h2>Code Overview</h2>
The script first sets up various path constants and data directories based on the current platform, either local, Google Colab, or Kaggle. If running on Google Colab, it will mount the Google Drive.

After defining some utility functions for cleaning folders and checking paths, the script then specifies the location of the data directory and checks that it exists.

Two key CSV files are expected in the data directory: 'FileList.csv' and 'VolumeTracings.csv'. These files contain essential information about the echocardiogram videos and the associated volume tracing data.

The script loads the data from these CSV files into pandas DataFrames. 'FileList.csv' contains split values that indicate the partitioning of the data into training, validation, and test sets.

A function named extractEDandESframes is defined to extract the ED and ES frames from a given echocardiogram video.

The main work of the script is done in the function saveEDandESimages, which processes each video in the specified directory. For each video, it extracts the ED and ES frames, saves them as separate image files, and writes the tracing data associated with these frames to a CSV file. The CSV file also includes the name of the associated image file and the split value from the original file list.

All output images and CSV files are saved in a specified output directory, which is created if it does not already exist.

<h2>Usage</h2>
To use the script, place your echocardiogram video files in the 'Videos' directory within your data directory, and make sure the 'FileList.csv' and 'VolumeTracings.csv' files are located in the root of the data directory. Update the paths in the script to reflect the locations of your files, and run the script.

Please make sure to adjust the file paths and names according to your setup, and ensure that the necessary Python libraries are installed in your environment.

You may want to provide additional instructions on how to install the required Python libraries, if necessary, as well as any other relevant usage or troubleshooting tips.

<h1>Sample Images</h1>

![image](https://github.com/OmdenaAI/topeka-chapter-ejection-fraction/assets/87038829/15f7c8f4-86b3-4a87-8bf8-a5ab363f36e3)
