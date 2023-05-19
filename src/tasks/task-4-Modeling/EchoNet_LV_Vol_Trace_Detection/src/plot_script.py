# plot_script.py

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def plot_learning_curve(history):
    lc = pd.DataFrame(history.history)
    lc.plot(figsize=(10,8))
    plt.title("Learning Curve", fontsize=25)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()

def load_latest_model_checkpoint(directory, MODEL_NAME, inputs, outputs, learning_rate=1e-4):
    latest = latest_checkpoint(directory)
    model = Model(inputs, outputs, name=MODEL_NAME)
    model.compile(loss='mae', optimizer=Adam(learning_rate))
    model.load_weights(latest)
    return model

def evaluate_model(model, data_images, data_keypoints):
    loss = model.evaluate(data_images, data_keypoints, verbose=2)
    return loss
