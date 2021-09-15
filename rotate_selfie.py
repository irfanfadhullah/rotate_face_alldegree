from mtcnn import MTCNN
import cv2 
from scipy import ndimage
# from google.colab.patches import cv2_imshow
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from PIL import Image
from deepface import DeepFace

def rotate_image(frame,center,scale,angle):
    (h, w) = frame.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    frame = cv2.warpAffine(frame, M, (h, w))
    return frame
    
    # return detect_face(frame)

def alignment_procedure(img, left_eye, right_eye):
  #this function aligns given face in img based on left and right eye coordinates
  
  left_eye_x, left_eye_y = left_eye
  right_eye_x, right_eye_y = right_eye
  
  #find rotation direction
  
  if left_eye_y > right_eye_y:
      point_3rd = (right_eye_x, left_eye_y)
      direction = -1 #rotate same direction to clock
  else:
      point_3rd = (left_eye_x, right_eye_y)
      direction = 1 #rotate inverse direction of clock
  
  #find length of triangle edges
  
  a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
  b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
  c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))
  
  #apply cosine rule
  
  if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
  
      cos_a = (b*b + c*c - a*a)/(2*b*c)
      angle = np.arccos(cos_a) #angle in radian
      angle = (angle * 180) / math.pi #radian to degree
  
      #rotate base image
  
      if direction == -1:
          angle = 90 - angle
  
      img = Image.fromarray(img)
      img = np.array(img.rotate(direction * angle))
   
  return img #return img anyway

def rotate_90 (image_path):
  detector = MTCNN()
  img = cv2.imread(image_path)
  detections = detector.detect_faces(img)
  (h, w) = img.shape[:2]
  # calculate the center of the image
  center = (w / 2, h / 2)
  scale = 1.0

  if detections:
      image_rotated = img
    # cv2_imshow(img)
    # print(detections)

  else:
    image_rotated = ndimage.rotate(img, 90, cval=255, order=1)
    detections = detector.detect_faces(image_rotated)
    if detections:
    #   cv2_imshow(image_rotated)
    else:
      image_rotated = ndimage.rotate(image_rotated, 90, cval=255, order=1)
      detections = detector.detect_faces(image_rotated)
      if detections:
        # cv2_imshow(image_rotated)
      else:
        image_rotated = ndimage.rotate(image_rotated, 90, cval=255, order=1)
        detections = detector.detect_faces(image_rotated)
        if detections:
        #   cv2_imshow(image_rotated)
  
  face = DeepFace.detectFace(image_rotated)
  plt.imshow(face)
  return face