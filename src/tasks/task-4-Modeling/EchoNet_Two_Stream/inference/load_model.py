import tensorflow as tf

def load_model(model_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'tf': tf,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'r2_score': r2_score
        })
    return model
