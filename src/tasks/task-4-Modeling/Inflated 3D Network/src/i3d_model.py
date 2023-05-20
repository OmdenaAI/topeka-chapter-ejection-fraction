from tensorflow import keras
from tensorflow.keras import layers

def conv3d_bn(x, filters, kernel_size, strides=(1, 1, 1), padding='same', use_bias=False, name=None):
    x = layers.Conv3D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def i3d_inception_module(x, filters_1x1x1, filters_3x3x3_reduce, filters_3x3x3, filters_3x3x3_double_reduce, filters_3x3x3_double, filters_pool_proj, name=None):
    # 1x1x1
    branch1x1x1 = conv3d_bn(x, filters_1x1x1, kernel_size=(1, 1, 1))

    # 1x1x1 -> 3x3x3
    branch3x3x3 = conv3d_bn(x, filters_3x3x3_reduce, kernel_size=(1, 1, 1))
    branch3x3x3 = conv3d_bn(branch3x3x3, filters_3x3x3, kernel_size=(3, 3, 3))

    # 1x1x1 -> 3x3x3 -> 3x3x3
    branch3x3x3dbl = conv3d_bn(x, filters_3x3x3_double_reduce, kernel_size=(1, 1, 1))
    branch3x3x3dbl = conv3d_bn(branch3x3x3dbl, filters_3x3x3_double, kernel_size=(3, 3, 3))
    branch3x3x3dbl = conv3d_bn(branch3x3x3dbl, filters_3x3x3_double, kernel_size=(3, 3, 3))

    # 3x3x3 (max pooling) -> 1x1x1
    branch_pool = layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, filters_pool_proj, kernel_size=(1, 1, 1))

    return layers.concatenate([branch1x1x1, branch3x3x3, branch3x3x3dbl, branch_pool], axis=-1, name=name)

def build_i3d(num_classes, input_shape, dropout_rate):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv3D(64, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same', name='pool1')(x)

    x = layers.Conv3D(192, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', name='conv2')(x)
    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same', name='pool2')(x)

    # Inception modules
    x = i3d_inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a')
    x = i3d_inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b')
    x = layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', name='pool3')(x)

    x = i3d_inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    x = i3d_inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b')
    x = i3d_inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c')
    x = i3d_inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d')
    x = i3d_inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e')
    x = layers.MaxPooling3D(pool_size=(2, 3, 3), strides=(2, 2, 2), padding='same', name='pool4')(x)

    x = i3d_inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a')
    x = i3d_inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b')
    x = layers.AveragePooling3D(pool_size=(2, 4, 4), strides=(1, 1, 1), padding='valid', name='pool5')(x)

    # Classifier
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)

    # Build and return the model
    model = keras.Model(inputs, x, name='i3d')
    return model

input_shape = (28, 112, 112, 1)  # (frames, height, width, channels)
num_classes = 10
dropout_rate = 0.5

