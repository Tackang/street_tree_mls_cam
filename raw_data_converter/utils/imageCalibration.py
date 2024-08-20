from dis import dis
import numpy as np
import cv2 as cv
import os
import natsort
import parmap
import multiprocessing

def imageCalibration(index, images, tkysImageDir,tkysImageDir_Calibrated):
    # Camera Paramters for Carsense System
    intrinsicParameter = np.array([
    [1058.688334, 0, 957.887598],
    [0, 1059.889393, 614.085426],
    [0, 0, 1],
    ])
    distortion = np.array([-0.126184,0.057164,-0.000468,-0.001805])
    
    # images = os.listdir(tkysImageDir)
    # images = natsort.natsorted(images)
    fileName =images[index]
    imagePath = os.path.join(tkysImageDir, fileName)
    image = cv.imread(imagePath)
    image_calibrated= cv.undistort(image, intrinsicParameter, distortion)
    cv.imwrite(os.path.join(tkysImageDir_Calibrated,fileName), image_calibrated)

def main(tkysImageDir, tkysImageDir_Calibrated):
    images = os.listdir(tkysImageDir)
    images = natsort.natsorted(images)
    
    pool =  multiprocessing.Pool(processes = 15)
    result = parmap.map(imageCalibration, range(len(images)), images, tkysImageDir, tkysImageDir_Calibrated, pm_pbar=True, pm_processes = 15)

if __name__ == '__main__':
    main()

    

