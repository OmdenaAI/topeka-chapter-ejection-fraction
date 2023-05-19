# Import necessary modules
import logging
import pprint
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Import the created scripts
import src.plot_script as ps
import src.calculation_script as cs
import src.visualization_script as vs

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Add console handler
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

# Define constants
OUTPUT_DIR = '/kaggle/working'
MODEL_NAME = 'MyModel'
IMAGE_SIZE = 112
NUM_KEYPOINTS = 12

# Load data
test_images, test_keypoints, test_ids = LoadData(OUTPUT_DIR, type='TEST')
logging.info(test_images.shape)
logging.info(test_keypoints.shape)
test_keypoints_conv = test_keypoints.astype('float32')

# Load model
inputs, outputs = ... # Define inputs and outputs of your model
model = ps.load_latest_model_checkpoint(OUTPUT_DIR, MODEL_NAME, inputs, outputs)

# Evaluate model
logging.info(f"Loss for testing images : {ps.evaluate_model(model, test_images, test_keypoints_conv)}")

# Get predicted keypoints
predicted_test_kps = cs.get_predicted_points(test_images, model)

# Build dataframe
test_output_df = cs.build_dataframe_EFs(test_keypoints_conv, predicted_test_kps)

# Visualize error
vs.visualize_error(test_images, test_keypoints_conv, predicted_test_kps, test_ids, test_output_df)

# Print confusion matrix
logging.info('Confusion Matrix for Testing Data')
vs.Accuracy_ConfusionMatrix(test_output_df.Actual_HFClass, 
                            test_output_df.Pred_HFClass,
                            test_output_df.Actual_HFClass.cat.categories)

# Visualize single data
vs.VisualizeSingleData(test_images, test_keypoints_conv, predicted_test_kps, test_ids, 531)

