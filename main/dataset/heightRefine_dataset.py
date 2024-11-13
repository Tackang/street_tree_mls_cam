import numpy as np
import os
import natsort

class hrDataset():
    def __init__(self,baseDir):
        self.groundDir=os.path.join(baseDir,'ground_singleFrame/')
        self.grounds=os.listdir(self.groundDir)
        self.grounds = natsort.natsorted(self.grounds)

        self.dsmDir=os.path.join(baseDir,'dsm_singleFrame/')

        self.imuDir=os.path.join(baseDir,'../preprocessed_data/imu/')


    def __len__(self):
        return len(self.grounds)


    def __getitem__(self,index):
        # lidar
        groundPath = os.path.join(self.groundDir,self.grounds[index])
        groundName = os.path.splitext(os.path.basename(groundPath))[0]
        ground = np.load(groundPath)
        # dsm
        dsmPath = os.path.join(self.dsmDir,'{}.npy'.format(groundName))
        dsm = np.load(dsmPath)
        # imu
        imuPath = os.path.join(self.imuDir,'{}.txt'.format(groundName))
        imu = np.loadtxt(imuPath)
    
        groundDsmData = [groundName, imu, ground, dsm]
        return groundDsmData

