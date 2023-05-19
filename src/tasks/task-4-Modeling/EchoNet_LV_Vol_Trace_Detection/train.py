from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from constants import EPOCHS, MODEL_NAME, WEIGHT_DIR
from model import create_model
from src.data_loader import LoadData
from src.logger import get_logger
logger = get_logger()

# Load training data
train_images, train_keypoints, train_ids = LoadData(OUTPUT_DIR)

# Load validation data
val_images, val_keypoints, val_ids = LoadData(OUTPUT_DIR, type='VAL')
val_keypoints_conv = val_keypoints.astype('float32')

# Callbacks definition
class ShowProgress(Callback):
    # Class body omitted for brevity...

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ModelCheckpoint(f'{WEIGHT_DIR}/{MODEL_NAME}-{{epoch:04d}}.ckpt', save_best_only=True, save_weights_only=True),
    ShowProgress()
]

# Model creation and training
model = create_model()
history = model.fit(train_images, train_keypoints, 
                    validation_data=(val_images, val_keypoints_conv), 
                    epochs=EPOCHS, 
                    callbacks=callbacks)
logger.info("Model training completed.")
