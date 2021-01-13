
import os, cv2, time, csv, sys, gc
import tensorflow as tf
import numpy as np
import time

sys.path.append('/home/pi/CatPreyAnalyzer')
sys.path.append('/home/pi')
#from CatPreyAnalyzer.cascade.py import do_pc_stage
from CatPreyAnalyzer.cascade import Cascade
from CatPreyAnalyzer.model_stages import PC_Stage, FF_Stage, Eye_Stage, Haar_Stage, CC_MobileNet_Stage
from CatPreyAnalyzer.camera_class import Camera
import cv2
import os

img_path= '/home/pi/CatPreyAnalyzer/no_prey/_0_4026.png'
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
start_time = time.time()

cas=Cascade()           
  
# print('predicting normal') 
# start_time = time.time()
# pred_class, pred_val, inference_time=Cascade.do_pc_stage( cas,pc_target_img=snout_crop)
# print('Prey Prediction: ' + str(pred_class))
# print('Pred_Val: ', str('%.2f' % pred_val))
# print('Total Runtime:', time.time() - start_time)

cc_target_img = cv2.imread('CatPreyAnalyzer/readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg')
#original_copy_img = cc_target_img.copy()

dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, cc_inference_time = Cascade.do_cc_mobile_stage(cas,cc_target_img=cc_target_img)
print('CC_Do Time:', time.time() - start_time)
cc_cat_bool = cat_bool
cc_pred_bb = pred_cc_bb_full
bbs_target_img = bbs_target_img
cc_inference_time = cc_inference_time

if cat_bool and bbs_target_img.size != 0:
    print('Cat Detected!')
rec_img = Cascade.cc_mobile_stage.draw_rectangle(img=cc_target_img, box=pred_cc_bb_full, color=(255, 0, 0), text='CC_Pred')

    # Do EYES
bbs_snout_crop, bbs, eye_inference_time = Cascade.do_eyes_stage(cas,eye_target_img=bbs_target_img,
                                                    cc_pred_bb=pred_cc_bb_full,
                                                                cc_target_img=cc_target_img)
rec_img = Cascade.cc_mobile_stage.draw_rectangle(img=rec_img, box=bbs, color=(255, 0, 255), text='BBS_Pred')
bbs_pred_bb = bbs
bbs_inference_time = eye_inference_time

# Do FF for Haar and EYES
bbs_dk_bool, bbs_face_bool, bbs_ff_conf, bbs_ff_inference_time = Cascade.do_ff_stage(cas,snout_crop=bbs_snout_crop)
ff_bbs_bool = bbs_face_bool
ff_bbs_val = bbs_ff_conf
ff_bbs_inference_time = bbs_ff_inference_time

inf_bb = bbs
face_bool = bbs_face_bool
snout_crop = bbs_snout_crop

face_bool = face_bool
face_box = inf_bb

if face_bool:
    rec_img = Cascade.cc_mobile_stage.draw_rectangle(img=rec_img, box=inf_bb, color=(255, 255, 255), text='INF_Pred')
print('Face Detected!')












# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/pi/CatPreyAnalyzer/models/Prey_Classifier/model.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# input details
print(input_details)
# output details
print(output_details)
def standardize(image_data):
        image_data -= np.mean(image_data, axis=0)
        image_data /= np.std(image_data, axis=0)
        return image_data

# read and resize the image
img = cv2.imread(img_path)
def resize_img( input_img):
      # prepare input image to shape (300, 300, 3) as the MobileNetV2 model specifies
      img = input_img
      #img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
      img= standardize(img)
      img = cv2.resize(img, (224, 224))
      return img
# convert to numpy array
data = tf.keras.preprocessing.image.img_to_array(img)
print('predicting tflite') 
#data_prey = img_to_array(img_prey)
data= resize_img(data)
# input_details[0]['index'] = the index which accepts the input
#new_img = data.astype(np.float32)
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], [data])

# run the inference
interpreter.invoke()

# output_details[0]['index'] = the index which provides the input


tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(tflite_results)
print('Total Runtime:', time.time() - start_time)