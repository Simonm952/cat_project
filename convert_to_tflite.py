import tensorflow as tf



model = tf.keras.models.load_model('/home/pi/CatPreyAnalyzer/models/Eye_Detector/trainwhole100_Epochs_2020_04_30_18_05_25.h5')


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite_eye","wb").write(tflite_model)