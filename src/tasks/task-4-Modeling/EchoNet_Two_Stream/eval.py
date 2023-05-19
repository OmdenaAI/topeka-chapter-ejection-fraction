import pandas as pd
import numpy as np
import tensorflow as tf

from inference.data_gen import generate_data, two_stream_batch_generator
from inference.utils import specificity, sensitivity, r2_score
from inference.model import load_model
from inference.test_metrics import calculate_metrics
from inference.plot_results import plot_results

BASE_PATH = '/kaggle/input/heartdatabase'

df = pd.read_csv(f'{BASE_PATH}/EchoNet-Dynamic/FileList.csv')

test_df = df[df["Split"] == "TEST"]
test_files = test_df.FileName + '.avi'
test_df = test_df.reset_index(drop=True)
test_files = list(test_files)

img_path = f'{BASE_PATH}/EchoNet-Dynamic/Videos'

model_path = '/kaggle/input/two-stream-baseline/best_two_stream.h5'
two_stream_model = load_model(model_path)

batch_size = 16
test_labels = test_df['EF'].values

test_gen = generate_data(test_files, img_path, test_df)
test_data = two_stream_batch_generator(batch_size, test_gen)

y_pred = two_stream_model.predict(test_data, steps=len(test_files) // batch_size)
y_true = np.array(test_labels)

mse, mae, r2 = calculate_metrics(y_true, y_pred)

plot_results(y_true, y_pred)
