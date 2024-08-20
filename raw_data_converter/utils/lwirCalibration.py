import os
import numpy as np
import math
import cv2 as cv
import multiprocessing
from tqdm import tqdm
# import sys
import parmap

class thermalDataset_bin():
    def __init__(self,baseDir):
        self.thermalDir=os.path.join(baseDir,'lwir/')
        self.thermals=os.listdir(self.thermalDir)
    
    def __len__(self):
        return len(self.thermals)

    def raw_to_celsius(self, bin, B_, R_, O_, F_, kT_0):
        return (B_ / math.log(R_ / (bin - O_) + F_) - kT_0)
 
    def __getitem__(self,index):
        thermalPath=os.path.join(self.thermalDir,self.thermals[index])
        thermalName=os.path.splitext(os.path.basename(thermalPath))[0]
        thermalVal=os.path.splitext(os.path.basename(thermalPath))

        with open (thermalPath,'rb') as f:
            rawData= f.read()
            dt = np.dtype(np.int16)
            dt = dt.newbyteorder('little')
            if np.frombuffer(rawData[-655360:],dtype=dt).size == 0:
                thermal = 9999
            else:
                # if np.frombuffer(rawData[-655360:],dtype=dt)
                bin = np.frombuffer(rawData[-655360:],dtype=dt).reshape((512,640)) 
                # Bin 에서 파라미터 읽어오기
                params = np.frombuffer(rawData[28:28+(8*5)], dtype=np.double)
                B_ = params[0]
                F_ = params[1]
                O_ = params[2]
                R_ = params[3]
                kT_0 = params[4]
                raw_to_celsius_vec = np.vectorize(self.raw_to_celsius)
        try:
            thermal = raw_to_celsius_vec(bin,B_,R_,O_,F_,kT_0)
        except:
            thermal = 9999

        thermalDic={thermalName:thermal}
        
        return thermalDic

def lwirCalibration(index, thermalData, tkysLwirImageDir_calibrated):

    # Lwir Camera Paramters for Carsense System
    intrinsicParameter = np.array([
        [434.905155, 0, 324.480806],
        [0, 434.920710, 262.421627],
        [0, 0, 1],
        ])
    distortion = np.array([-0.381497,0.117428,-0.006102,-0.000935])

    filename = list(thermalData[index].keys())[0]
    lwir = list(thermalData[index].values())[0]
    if type(lwir) is int:
        lwir = np.full((512,640),0)
        np.savetxt(os.path.join(tkysLwirImageDir_calibrated,'{}.txt'.format(filename)),lwir)
    else:
        lwirCalibrated = cv.undistort(lwir, intrinsicParameter,distortion)
        np.savetxt(os.path.join(tkysLwirImageDir_calibrated,'{}.txt'.format(filename)),lwirCalibrated)
        
def main(tkysDir, tkysLwirImageDir_Calibrated): 
    thermalData = thermalDataset_bin(tkysDir)

    pool =  multiprocessing.Pool(processes = 15)
    result = parmap.map(lwirCalibration, range(len(thermalData)),thermalData, tkysLwirImageDir_Calibrated, pm_pbar=True, pm_processes = 15)

if __name__ == '__main__':
    main()


