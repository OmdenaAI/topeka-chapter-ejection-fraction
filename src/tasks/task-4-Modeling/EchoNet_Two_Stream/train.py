from src.utilities import *
from src.data import *
from src.models import *

def train():
    
    batch_size = 8

    # Load your data
    df = pd.read_csv(f'{BASE_PATH}/EchoNet-Dynamic/FileList.csv')

    train_df = df[df["Split"] == "TRAIN"]
    train_files = train_df.FileName + '.avi'
    train_df = train_df.reset_index(drop=True)
    train_files = list(train_files)

    val_df = df[df["Split"] == "VAL"]
    val_files = val_df.FileName + '.avi'
    val_df = val_df.reset_index(drop=True)
    val_files = list(val_files)

    test_df = df[df["Split"] == "TEST"]
    test_files = test_df.FileName + '.avi'
    test_df = test_df.reset_index(drop=True)
    test_files = list(test_files)

    img_path = f'{BASE_PATH}/EchoNet-Dynamic/Videos'

    tpu = 'local'
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
    strategy = tf.distribute.TPUStrategy(tpu)

    batch_size = batch_size * strategy.num_replicas_in_sync
    train_steps_per_epoch = int(np.ceil(len(train_files) / batch_size))
    val_steps_per_epoch = int(np.ceil(len(val_files) / batch_size))
    test_steps_per_epoch = int(np.ceil(len(test_files) / batch_size))


    with strategy.scope():
        
       
        # Create data generators
        train_gen = generate_data(train_files, img_path, train_df)
        val_gen = generate_data(val_files, img_path, val_df)
        #batch_size = 16 * strategy.num_replicas_in_sync # adjust batch size for multiple GPUs

        # Initialize the spatial and temporal models
        spatial_model = create_spatial_model()
        temporal_model = create_temporal_model()

        # Create data generators
        train_data = two_stream_batch_generator(batch_size, train_gen)
        val_data = two_stream_batch_generator(batch_size, val_gen)

        # Create the two-stream model
        two_stream_model = create_two_stream_model(spatial_model, temporal_model)
  


        # Compile the model
        two_stream_model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            metrics=[tf.keras.metrics.RootMeanSquaredError(),specificity, sensitivity, r2_score]
        )
        
        #Load model
        #two_stream_model = load_model('/kaggle/input/omdena-lvef-two-stream/best_two_stream.h5')
        # Load weights for layer 1 and layer 2 only
        #two_stream_model.load_weights('/kaggle/input/two-stream-baseline/best_two_stream.h5', by_name=True, skip_mismatch=True)
              
        # Load the saved model
        loaded_model = tf.keras.models.load_model('/kaggle/input/omdena-lvef-two-stream/best_two_stream.h5',
                                           custom_objects={'tf': tf,
                                                           'specificity': specificity,
                                                           'sensitivity': sensitivity,
                                                           'r2_score': r2_score
                                                          })

        
        # Set up the model checkpoint
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_two_stream.h5',
                monitor="val_root_mean_squared_error",
                verbose =1,
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=20,
            ),
            tf.keras.callbacks.LearningRateScheduler(step_decay)
        ]

    # Train the model
    history = two_stream_model.fit(
        train_data,
        steps_per_epoch=train_steps_per_epoch,
        epochs=50,
        verbose=1,
        validation_data=val_data,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks
    )
    
    
    #import pandas as pd
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('history.csv', index=False)

if __name__ == "__main__":
    train()
