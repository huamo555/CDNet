
#加载和调用
import csv
import joblib
import os
import cv2
import numpy as np
clf = joblib.load("D:\svm_model.pkl")
SHAPE = (30, 30)


def extractFeaturesFromImage(image_file):
   img = cv2.imread(image_file)
   img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)
   img = img.flatten()
   img = img / np.mean(img)
   return img



r_1=[]
r_2=[]
r_3=[]

fp=open("C:/Users/Administrator/Desktop/graspness_implementation-main/a/svm_model.pkl","w")
f_csv=csv.writer(fp)
ll=[]
ll.append(["id","label"])
for i in range(0,300): 
          
            img=extractFeaturesFromImage("C:/Users/Administrator/Desktop/graspness_implementation-main/a/test"+"/"+str(i)+".jpg")
            imageFeature = img.reshape(1, -1)
            r_1.append(clf.predict(imageFeature)[0])
            print(str(i)+".jpg" , clf.predict(imageFeature)[0])
            """ a=int(clf.predict(imageFeature)[0])
            #print(str(a))
            ll.append([str(i)+".jpg",str(a)]) """
f_csv.writerows(ll)
fp.close()
os.system("pause")
os.system("pause")
