import tensorflow as tf
import os
import numpy as np
import h5py
print(tf.__version__)
PC_models_dir = 'CatPreyAnalyzer/models/Prey_Classifier'
models_dir = PC_models_dir
pc_model_name = '0.86_512_05_VGG16_ownData_FTfrom15_350_Epochs_2020_05_15_11_40_56.h5'
pc_model = tf.keras.models.load_model('/home/pi/CatPreyAnalyzer/models/Prey_Classifier/0.86_512_05_VGG16_ownData_FTfrom15_350_Epochs_2020_05_15_11_40_56.h5')



print(pc_model.summary())

model = tf.keras.Sequential()
for layer in pc_model.layers[:-1]: # go through until last layer
    model.add(layer)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy')

base_model = model

base_model.trainable = False



# Create new model on top
inputs = tf.keras.Input(shape=(224, 224, 3))
#x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
# norm_layer = keras.layers.experimental.preprocessing.Normalization()
# mean = np.array([127.5] * 3)
# var = mean ** 2
# # Scale inputs to [-1, +1]
# x = norm_layer(x)
# norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(inputs, training=False)
x= tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout

outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)

x_train=[]
import cv2
import os


for filename in os.listdir('/home/pi/CatPreyAnalyzer/no_prey/'):
    print(filename)
    img = cv2.imread(os.path.join('/home/pi/CatPreyAnalyzer/no_prey/',filename))
    if img is not None:
        
        x_train.append(img)
import cv2
import os


for filename in os.listdir('/home/pi/CatPreyAnalyzer/prey/'):
    print(filename)
    img = cv2.imread(os.path.join('/home/pi/CatPreyAnalyzer/prey/',filename))
    if img is not None:
        
        x_train.append(img)

y_train= []
for i in range(18):
  if i >= 9:
     y_train.append(1)
  else:
    y_train.append(0)




history = model.fit(np.array(x_train),np.array(y_train),validation_split=0.2,epochs=15)

# heb decode verwijderd in library dus wss kan ik wel trainen oop mac moet nog bezien
tf.keras.models.save_model(model,'/home/pi/CatPreyAnalyzer/my_model_prey_2')
import h5py
print(h5py.__version__)