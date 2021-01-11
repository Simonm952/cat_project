
import os, cv2, time, csv, sys, gc


sys.path.append('/home/pi/CatPreyAnalyzer')
sys.path.append('/home/pi')
#from CatPreyAnalyzer.cascade.py import do_pc_stage
from CatPreyAnalyzer.cascade import Cascade
from CatPreyAnalyzer.model_stages import PC_Stage, FF_Stage, Eye_Stage, Haar_Stage, CC_MobileNet_Stage
from CatPreyAnalyzer.camera_class import Camera
import cv2
import os

img_path= '/home/pi/CatPreyAnalyzer/prey/2021-01-11 12.31.53.jpg'
# load the image
snout_crop= cv2.imread(img_path)
# def resize_img( input_img):
#       # prepare input image to shape (300, 300, 3) as the MobileNetV2 model specifies
#       img = input_img
#       #img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#       img = cv2.resize(img, (224, 224))
#       return img
# # convert to numpy array
# data = tf.keras.preprocessing.image.img_to_array(img)

# #data_prey = img_to_array(img_prey)
# data= resize_img(data)

  
pred_class, pred_val, inference_time=Cascade.do_pc_stage(Cascade(),pc_target_img=snout_crop)
print('Prey Prediction: ' + str(pred_class))
print('Pred_Val: ', str('%.2f' % pred_val))