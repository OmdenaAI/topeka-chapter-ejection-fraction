import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from utils import batch_generator, generate_data
from lrfinder import LRFinder
from model import tsn_resnet50

def main():

    df = pd.read_csv("/kaggle/input/heartdatabase/EchoNet-Dynamic/FileList.csv")
    print(f"Total videos for training: {len(df)}")

    train_df = df[df["Split"] == "TRAIN"]
    train_files = df.FileName[df["Split"] == "TRAIN"]+'.avi'

    train_df = train_df.reset_index(drop=True)
    train_files = list(train_files)

    val_df = df[df["Split"] == "VAL"]
    val_files = df.FileName[df["Split"] == "VAL"]+'.avi'
    val_df = val_df.reset_index(drop=True)
    val_files = list(val_files)

    val_data = batch_generator(16, generate_data(val_files, path, val_df))

    input_shape = (28, 112, 112, 1)  # (frames, height, width, channels)
    num_classes = 10
    dropout_rate = 0.5

    # Create and compile the model
    num_classes = 1
    num_segments = 32
    input_shape = (112, 112, 1)
    tsn_model = tsn_resnet50(num_classes, num_segments, input_shape, base_model)

    batch_size = 64
    num_epoch = 5
    steps = len(train_files[0:500])//batch_size
    print(steps)

    lr_finder = LRFinder()
    history = model.fit(x= batch_generator(batch_size, generate_data(train_files[0:500], path, train_df)),
                        validation_data =(next(val_data)[0], next(val_data)[1]) ,
                        epochs=num_epoch, steps_per_epoch=steps, callbacks=[lr_finder],
                        verbose=1)

    model.save("best_model.h5")

if __name__ == "__main__":
    main()
