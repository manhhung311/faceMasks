import tensorflowjs as tf
from keras.models import Sequential,load_model
model = load_model(path.join(path.abspath(path.dirname(__file__)), "./model_mask.h5"))
tf.converters.save_keras_model(model, path.join(path.abspath(path.dirname(__file__)), "./modelmask"))