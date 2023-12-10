MODEL_NAME = 'emnist_letters.h5'

import keras
import tf2onnx

# оптимизация модели из формата tensorflow в onnx
model = keras.models.load_model(MODEL_NAME)

onnx, _ = tf2onnx.convert.from_keras(model, output_path='model.onnx', opset=9)