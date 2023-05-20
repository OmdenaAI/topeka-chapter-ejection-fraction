from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


input_shape = (112, 112, 1)


from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Multiply

def spatial_attention(input_feature):
    attention = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(input_feature)
    return Multiply()([input_feature, attention])

def custom_resnet50(input_shape):
    input_layer = layers.Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_layer)
    return base_model

base_model = custom_resnet50(input_shape)

def tsn_resnet50(num_classes, num_segments, input_shape, base_model):
    # Input layer for segments
    input_segments = layers.Input(shape=(num_segments, *input_shape))

    # Preprocessing
    rescale = Rescaling(1.0/255.0)

    # Feature extraction
    features_list = []
    for i in range(num_segments):
        segment = layers.Lambda(lambda x: x[:, i])(input_segments)
        segment = rescale(segment)
        features = base_model(segment)
        features = spatial_attention(features)
        features = layers.GlobalAveragePooling2D()(features)
        features_list.append(features)

    # Temporal aggregation (Global Average Pooling)
    aggregated_features = layers.Average()(features_list)
    aggregated_features = layers.BatchNormalization()(aggregated_features)

    # Classifier
    output = layers.Dense(num_classes, activation='linear')(aggregated_features)

    # Create the model
    model = Model(inputs=input_segments, outputs=output)
    return model

# Set parameters and build the model
num_classes = 1
num_segments = 32
input_shape = (112, 112, 1)
tsn_model = tsn_resnet50(num_classes, num_segments, input_shape, base_model)

# Compile the model
#tsn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
tsn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-02), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
