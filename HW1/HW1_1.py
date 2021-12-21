# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 01:55:40 2021

@author: leo90
"""
import cv2
import numpy as np



def main():
    path = r'C:\Users\leo90\AI_Medical\HW1\EKG\EKG_001-120\EKG_001-120\2.jpg'
    img = cv2.imread(path)
    img_np = np.array(img)
    ltop = (110, 350)
    rtbm = (1350, 810)
    img_cap = img_np[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]
    # print(img_cap.shape)
    cv2.imshow('test',img_cap)
    # cv2.imshow('test2',img_np)
    cv2.waitKey(0) 
    list_img  = list()
    for i in range(0,3):
        for j in range(0,4):
            ltop = ((1240//4)*j, (460//3)*i)
            rtbm = ((1240//4)*(j+1), (460//3)*(i+1))
            img_cut=img_cap[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]
            list_img.append(img_cut)
    print(type(list_img[0]))
    cv2.imshow('I',list_img[0])
    cv2.imshow('aVR',list_img[1])
    cv2.imshow('V1',list_img[2])
    cv2.imshow('V4',list_img[3])
    cv2.imshow('II',list_img[4])
    cv2.imshow('aVL',list_img[5])
    cv2.imshow('V2',list_img[6])
    cv2.imshow('V5',list_img[7])
    cv2.imshow('III',list_img[8])
    cv2.imshow('aVF',list_img[9])
    cv2.imshow('V3',list_img[10])
    cv2.imshow('V6',list_img[11])
    cv2.waitKey(0)
    
    
    
    
if __name__ == '__main__':
    main()