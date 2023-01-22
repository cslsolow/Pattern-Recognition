# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     try
   Description :
   Author :       Silin
   date：          2023/1/5
-------------------------------------------------
   Change Activity:
                   2023/1/5:
-------------------------------------------------
"""
import os
import cv2

path = './chest_xray/train_DA_Horizontal/NORMAL/'
dirs = os.listdir(path)

for pics in dirs:
    outputPath = "./chest_xray/train_DA_Horizontal/NORMAL/"
    img = cv2.imread(path + pics)

    frame = cv2.flip(img, 1, dst=None)

    filename = outputPath + "DA_" + pics
    cv2.imwrite(filename, frame)
    print("Successfully save :{}".format(filename))
    break

path = './chest_xray/train_DA_Horizontal/VIRUS/'
dirs = os.listdir(path)

for pics in dirs:
    outputPath = "./chest_xray/train_DA_Horizontal/VIRUS/"
    img = cv2.imread(path + pics)

    frame = cv2.flip(img, 1, dst=None)

    filename = outputPath + "DA_" + pics
    cv2.imwrite(filename, frame)
    print("Successfully save :{}".format(filename))
    break
