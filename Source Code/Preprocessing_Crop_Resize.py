'''
    This code crops and resizes a face.
    This code was executed on Google Colaboratory.
    Create the OUTPUT_PATH Directory.
    The dataset directory structure should be classwise directories.
  
        /Krishna/Krishna_1.jpg
        /Rahul/Rahul_1.jpg
        /Sonal/Sonal_1.jpg
        /Priya/Priya_1.jpg
'''
import cv2
import sys
import os

#INSERT THE DATASET PATH HERE.
root_path = "DATASET_PATH"
save_path = "OUTPUT_PATH"
imagePath = root_path

for image_class in os.listdir(root_path):

    count = 0
    save_path_file = os.path.join(save_path,image_class)
    try:
      os.mkdir(save_path_file)
    except OSError as error:
      print(error)  
    
    for img in os.listdir(os.path.join(root_path,image_class)):
    
      path_ = os.path.join(root_path,image_class,img)
      image = cv2.imread(path_)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
      faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
      faces = faceCascade.detectMultiScale(
          gray,
          scaleFactor=1.3,
          minNeighbors=3,
          minSize=(30, 30)
      )      
      
      for (x, y, w, h) in faces:          
          roi_color = image[y:y + h, x:x + w]
          roi_color = cv2.resize(roi_color, (64, 64))
          filename = image_class + '_' + str(count) + '.jpg'
          cv2.imwrite(os.path.join(save_path_file,filename), roi_color)
            
      count+=1