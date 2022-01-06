# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:50:00 2021

@author: leo90
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import signal


def main():
    path = r'1.jpg'
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img = cv2.GaussianBlur(img, (5, 5), 0)  #use gaussianblur to process image
    img_np = np.array(img)

    ltop = (110, 805)
    rtbm = (1350, 920)
    img_cap = img_np[ltop[1]:rtbm[1], ltop[0]: rtbm[0]] #cut image for Long Lead II
    # cv2.imshow('Long Lead II',img_cap)
    # cv2.waitKey(500) 
    threshold=135;  #set the threshold for turning the image into black and white
    x_list=list()
    y_list=list()
    # print(img_cap.shape)
    
    #image turn into balck and white
    for i in range(img_cap.shape[1]):   
        for j in range(img_cap.shape[0]):
            if img_cap[j][i]>threshold: 
                img_cap[j][i]=255
            else:
                img_cap[j][i]=0 
                
    for i in range(img_cap.shape[1]):
        for j in range(img_cap.shape[0]):
            if img_cap[j][i]==0: 
                img_cap[j][i]=0 
                x_list.append(i)
                y_list.append(img_cap.shape[0]-j)
                break
    
    #following codes are using to get rid of the things besides ECG signals
    ltop = (25, 10)
    rtbm = (50, 30)
    img_cap[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]=255
    # cv2.imshow('Long Lead II',img_cap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    y_nparray = np.array(y_list)

    
    #extract outline of image
    b, a = signal.butter(8, 0.01, 'lowpass')   
    y_nparray_outline = signal.filtfilt(b, a, y_nparray)  
    
    y_nparray_norm_data = y_nparray - y_nparray_outline
    avg = sum(y_nparray_norm_data)/len(y_nparray_norm_data)
    for i in range(len(y_nparray)):
        y_nparray[i] =  y_nparray_norm_data[i] - avg
    
    #filter to get rid of high frequency noise
    b, a = signal.butter(8, 0.18, 'lowpass')
    filtedData = signal.filtfilt(b, a, y_nparray)  
    
    plt.figure().set_size_inches(10, 4)
    peaks, _ = find_peaks(filtedData,distance=50,height=0)  #find peaks
    plt.plot(filtedData)
    # print(filtedData.shape)
    plt.xticks(range(0,1210,121),['0','1','2','3','4','5','6','7','8','9'])
    plt.plot(peaks, filtedData[peaks], "o")
    BPM = str(len(peaks)*6)
    plt.legend(['ECG signal','BPM: '+BPM])
    plt.xlabel('Time(s)')
    plt.show()
    
    
    
if __name__ == '__main__':
    main()