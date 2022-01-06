# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 01:55:40 2021

@author: leo90
"""
import cv2
import numpy as np



def main():
    path = r'1.jpg'
    #change the path if need
    img = cv2.imread(path)
    img_np = np.array(img)
    ltop = (110, 350)
    rtbm = (1350, 810)
    img_cap = img_np[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]

    list_img  = list()  #use list to save the image nparray
    for i in range(0,3):
        for j in range(0,4):
            ltop = ((1240//4)*j, (460//3)*i)
            rtbm = ((1240//4)*(j+1), (460//3)*(i+1))
            img_cut=img_cap[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]
            list_img.append(img_cut)
    print(type(list_img[0]))
    for i in range(0,12):   #show twelve pictures one by one
        cv2.imshow('twelve leads',list_img[i])
        cv2.waitKey(500)        #0.5 second per image
    cv2.destroyAllWindows()
    
    # cv2.imshow('I',list_img[0])
    # cv2.imshow('aVR',list_img[1])
    # cv2.imshow('V1',list_img[2])
    # cv2.imshow('V4',list_img[3])
    # cv2.imshow('II',list_img[4])
    # cv2.imshow('aVL',list_img[5])
    # cv2.imshow('V2',list_img[6])
    # cv2.imshow('V5',list_img[7])
    # cv2.imshow('III',list_img[8])
    # cv2.imshow('aVF',list_img[9])
    # cv2.imshow('V3',list_img[10])
    # cv2.imshow('V6',list_img[11])
    
    
    
    
    
if __name__ == '__main__':
    main()