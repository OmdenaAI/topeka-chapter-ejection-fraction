from keras.models import Model
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from keras.layers import Input, Dropout, SeparableConv2D
from tensorflow.keras.optimizers import Adam
from constants import NUM_KEYPOINTS, IMAGE_SIZE, MODEL_NAME

def create_model():
    backbone = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', include_top=False)
    backbone.trainable = False
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="InputLayer")
    x = mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    x = Dropout(0.3, name="DropOut")(x)
    x = SeparableConv2D(NUM_KEYPOINTS, kernel_size=3, activation='relu', data_format='channels_last', name="ConvPass")(x)
    outputs = SeparableConv2D(NUM_KEYPOINTS, kernel_size=2, activation='sigmoid', data_format='channels_last', name="OutputLayer")(x)
    model = Model(inputs, outputs, name=MODEL_NAME)
    model.compile(loss='mae', optimizer=Adam(learning_rate=1e-4)) 
    return model
