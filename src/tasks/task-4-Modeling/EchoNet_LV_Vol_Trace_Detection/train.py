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
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 20 == 0:
            plt.subplots(1, 4, figsize=(10, 10))
            for i, k in enumerate(np.random.randint(num_total, size=2)):
                img = train_images[k]
                img = img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
                pred_kps = self.model.predict(img)
                pred_kps = pred_kps.reshape(-1,NUM_KEYPOINTS) * IMAGE_SIZE
                kps = train_keypoints_conv[k].reshape(-1,NUM_KEYPOINTS) * IMAGE_SIZE
                plt.subplot(1, 4, 2*i+1)
                plt.gca().set_yticklabels([])
                plt.gca().set_xticklabels([])
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                VisualizeSampleImages(img[0], pred_kps, col='#16a085')
                plt.xlabel(f"Predicted")
                plt.subplot(1, 4, 2*i+2)
                plt.gca().set_yticklabels([])
                plt.gca().set_xticklabels([])
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                VisualizeSampleImages(img[0], kps)
                plt.xlabel(f"GT:{train_ids[k]}")
            plt.show()..

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
