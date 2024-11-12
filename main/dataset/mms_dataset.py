import numpy as np
import os
from PIL import Image
import natsort
import warnings
warnings.filterwarnings('ignore')

class mmsDataset():
    def __init__(self,baseDir):
        self.lidarDir=os.path.join(baseDir,'pointCloudFrame/')
        # self.lidarDir=os.path.join(baseDir,'treeTemp_std1/')
        self.lidars=os.listdir(self.lidarDir)
        self.lidars = natsort.natsorted(self.lidars)

        # self.imgDir=os.path.join(baseDir,'image_calibrated/')
        self.imgDir=os.path.join(baseDir,'image_processed/')

        self.thermalDir=os.path.join(baseDir,'lwir/')

        self.imuDir=os.path.join(baseDir,'imu/')
        
        self.imageImuDir=os.path.join(baseDir,'image_imu/')
        self.lwirImuDir=os.path.join(baseDir,'lwir_imu/')

        self.segDir = os.path.join(baseDir,'image_seg/')
        self.bbDir = os.path.join(baseDir,'image_species/labels/')

    def __len__(self):
        return len(self.lidars)

    def raw_to_celsius(self,bin,B_,R_,O_,F_,kT_0):
        return (B_ / np.log(R_ / (bin - O_) + F_) - kT_0)

    def __getitem__(self,index):
        # lidar
        lidarPath = os.path.join(self.lidarDir,self.lidars[index])
        lidarName = os.path.splitext(os.path.basename(lidarPath))[0]
        lidar = np.fromfile(lidarPath,dtype = np.float64).reshape(-1,6)
        # lidar = np.loadtxt(lidarPath)
        
        # imu
        imuPath = os.path.join(self.imuDir,'{}.txt'.format(lidarName))
        imu = np.loadtxt(imuPath)
        
        # image
        imgPath = os.path.join(self.imgDir,'{}.jpg'.format(lidarName))
        if np.array(Image.open(imgPath).convert("RGB")).size ==0:
            image = 9999
        image = np.array(Image.open(imgPath).convert("RGB"))
        
        # thermal
        thermalPath = os.path.join(self.thermalDir,'{}.bin'.format(lidarName))
        with open (thermalPath,'rb') as f:
            rawData= f.read()
            dt = np.dtype(np.int16)
            dt = dt.newbyteorder('little')
            if np.frombuffer(rawData[-655360:],dtype=dt).size == 0:
                thermal = 9999
            else:
                try:
                    bin = np.frombuffer(rawData[-655360:],dtype=dt).reshape((512,640)) 
                    params = np.frombuffer(rawData[28:28+(8*5)], dtype=np.double)
                    B_ = params[0]
                    F_ = params[1]
                    O_ = params[2]
                    R_ = params[3]
                    kT_0 = params[4]
                    raw_to_celsius_vec = np.vectorize(self.raw_to_celsius)
                    thermal = raw_to_celsius_vec(bin,B_,R_,O_,F_,kT_0)
                except:
                    thermal=9999
                
        # imageImu
        imageImuPath = os.path.join(self.imageImuDir,'{}.txt'.format(lidarName))
        if not os.path.isfile(imageImuPath) :
            imageImu = 9999
        else:
            imageImu = np.loadtxt(imageImuPath)
        
        # lwirImu
        lwirImuPath = os.path.join(self.lwirImuDir,'{}.txt'.format(lidarName))
        if not os.path.isfile(lwirImuPath) :
            lwirImu = 9999
        else:
            lwirImu = np.loadtxt(lwirImuPath)
        
        # Segmentation
        segPath = os.path.join(self.segDir,'{}.jpg'.format(lidarName))
        seg = np.array(Image.open(segPath).convert("L"), dtype=np.float32)

        # Yolo Detection
        bbPath = os.path.join(self.bbDir,'{}.txt'.format(lidarName))
        if not os.path.isfile(bbPath) :
            bb = 9999
        else:
            annotation=np.loadtxt(bbPath).reshape(-1,6) 
            # Bounding Box preprocessing. Change Center to Corner value
            # Annotation column : 0)classID; 1)xCenter; 2)yCenter; 3)width; 4)height; 5)confidenceScore 
            '''Multiply pixel size'''
            # xPixelValue
            annotation[:,1]*=1920
            annotation[:,3]*=1920
            # yPixelValue
            annotation[:,2]*=1200
            annotation[:,4]*=1200      
            xMin=annotation[:,1]-annotation[:,3]/2
            xMax=annotation[:,1]+annotation[:,3]/2
            yMin=annotation[:,2]-annotation[:,4]/2
            yMax=annotation[:,2]+annotation[:,4]/2
            boundingBoxFloat=np.vstack((annotation[:,0],np.vstack((np.vstack((np.vstack((xMin,xMax)),yMin)),yMax)))).T
            # BoundingBox column : 0)classID; 1)xMin; 2)xMax; 3)yMin; 4)yMax; 5)confidence score;
            boundingBox=np.around(boundingBoxFloat)
            bb=np.hstack((boundingBox,annotation[:,5].reshape(-1,1)))

        mmsdata = [lidarName, imu, lidar, image, thermal, imageImu, lwirImu, seg, bb]
        return mmsdata

