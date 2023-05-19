from pathlib import Path

NUM_KEYPOINTS = 84
IMAGE_SIZE = 112
OUTPUT_DIR = Path('/kaggle/working/Output')
EPOCHS = 50
MODEL_NAME = 'LV_Cavity_Volume_Trace'
WEIGHT_DIR = Path('/kaggle/working/Weights')
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
