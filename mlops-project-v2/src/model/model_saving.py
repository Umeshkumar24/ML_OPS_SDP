import tensorflow as tf

def save_model(model, model_name='draught_model'):
    model.save(f'{model_name}.h5')

def load_model(model_name='draught_model'):
    return tf.keras.models.load_model(f'{model_name}.h5')