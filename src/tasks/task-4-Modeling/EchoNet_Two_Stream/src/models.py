from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

import tensorflow_addons as tfa

def create_spatial_model(input_shape=(28, 112, 56, 1)):
    spatial_input = Input(shape=input_shape, name='spatial_input')

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
                kernel_regularizer=regularizers.l2(0.05))(spatial_input)
    x = tfa.layers.InstanceNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.05))(x)
    x = Dropout(0.05)(x)
    spatial_output = Dense(1, activation='linear')(x)
    spatial_model = Model(inputs=spatial_input, outputs=spatial_output)

    return spatial_model

def create_temporal_model(input_shape=(26, 112, 56, 1)):
    temporal_input = Input(shape=input_shape, name='temporal_input')

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.05))(temporal_input)
    x = tfa.layers.InstanceNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(0.05))(x)
    x = Dropout(0.05)(x)

    temporal_output = Dense(1, activation='linear')(x)
    temporal_model = Model(inputs=temporal_input, outputs=temporal_output)

    return temporal_model

def create_two_stream_model(spatial_model, temporal_model):
    spatial_output = spatial_model.output
    temporal_output = temporal_model.output

    merged = concatenate([spatial_output, temporal_output])
    merged = Dense(1024, activation='relu',
                   kernel_regularizer=regularizers.l2(0.05))(merged)
    merged = Dropout(0.05)(merged)
    merged_output = Dense(1, activation='linear')(merged)

    two_stream_model = Model(inputs=[spatial_model.input, temporal_model.input], outputs=merged_output)

    return two_stream_model
