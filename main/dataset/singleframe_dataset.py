import numpy as np
import os
from PIL import Image
import natsort
import warnings

warnings.filterwarnings('ignore')

class sfDataset():
    def __init__(self,baseDir):
        # self.lidarDir=os.path.join(baseDir,'pointCloudFrame/')
        self.lidarDir=os.path.join(baseDir,'tree_singleFrame/')
        self.lidars=os.listdir(self.lidarDir)
        self.lidars = natsort.natsorted(self.lidars)
        
        self.imageImuDir=os.path.join(baseDir,'../preprocessed_data/image_imu/')

        self.centerDir = os.path.join(baseDir,'center_singleFrame/')

    def __len__(self):
        return len(self.lidars)

    def __getitem__(self,index):
        # lidar
        lidarPath = os.path.join(self.lidarDir,self.lidars[index])
        lidarName = os.path.splitext(os.path.basename(lidarPath))[0]
        # lidar = np.fromfile(lidarPath,dtype = np.float64).reshape(-1,6)
        lidar = np.load(lidarPath)
    
        # imageImu
        imageImuPath = os.path.join(self.imageImuDir,'{}.txt'.format(lidarName))
        if not os.path.isfile(imageImuPath) :
            imageImu = 9999
        else:
            imageImu = np.loadtxt(imageImuPath)
        
        
        centerPath = os.path.join(self.centerDir,'{}.npy'.format(lidarName))
       
        if not os.path.isfile(centerPath):
            center = 9999
        else:
            center = np.load(centerPath)

        treedata = [lidarName, lidar, imageImu, center]
        return treedata
