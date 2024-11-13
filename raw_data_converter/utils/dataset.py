import numpy as np
import os
# from PIL import Image
import math
import natsort
import warnings
warnings.filterwarnings('ignore')

class bbDataset():
    def __init__(self,baseDir):
        self.bbDir=os.path.join(baseDir,'image_00_cal_detected_final/labels/')
        self.bbs=os.listdir(self.bbDir)
        self.bbs = natsort.natsorted(self.bbs)

    def __len__(self):
        return len(self.bbs)

    def __getitem__(self,index):
        bbPath=os.path.join(self.bbDir,self.bbs[index])
        bbNameForCheck=os.path.basename(bbPath)
        if "NaN" in bbNameForCheck:
            bbDic={"NaN":"NaN"}
        
        else:
            annotation=np.loadtxt(bbPath).reshape(-1,6)
            bbName=os.path.splitext(os.path.basename(bbPath))[0]

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
            bbDic={bbName:bb}
        
        return bbDic

class lidarDataset():
    def __init__(self,baseDir):
        self.lidarDir=os.path.join(baseDir,'pointCloudPackets/')
        self.lidars=os.listdir(self.lidarDir)
        self.lidars = natsort.natsorted(self.lidars)

    def __len__(self):
        return len(self.lidars)

    def __getitem__(self,index):
        lidarPath=os.path.join(self.lidarDir,self.lidars[index])
        lidarName=os.path.splitext(os.path.basename(lidarPath))[0]

        lidar=np.fromfile(lidarPath,dtype = np.float64).reshape(-1,6)

        # lidar=np.fromfile(lidarPath, dtype=np.float32).reshape((-1,4))
        # lidarX=lidar[:,0].reshape(-1,1)
        # lidarY=lidar[:,1].reshape(-1,1)
        # lidarZ=lidar[:,2].reshape(-1,1)
        # lidarI=lidar[:,3].reshape(-1,1)
        # lidar=np.hstack((np.hstack((np.hstack((-lidarY,lidarX)),lidarZ)),lidarI))
        lidarDic={lidarName:lidar}

        return lidarDic

# thermal Part 나중에 작업하기
class thermalDataset():
    def __init__(self,baseDir):
        self.thermalDir=os.path.join(baseDir,'image_02_cal/')
        self.thermals=os.listdir(self.thermalDir)
        self.thermals = natsort.natsorted(self.thermals)
    
    def __len__(self):
        return len(self.thermals)
 
    def __getitem__(self,index):
        thermalPath=os.path.join(self.thermalDir,self.thermals[index])
        thermalName=os.path.splitext(os.path.basename(thermalPath))[0]
        thermal = np.loadtxt(thermalPath)
        if thermal.shape != (512,640):
            thermal = -9999
        thermalDic={thermalName:thermal}
        
        return thermalDic

class imuDataset():
    def __init__(self,baseDir):
        self.imuDir=os.path.join(baseDir,'imu/')
        self.imus=os.listdir(self.imuDir)
        self.imus = natsort.natsorted(self.imus)
    
    def __len__(self):
        return len(self.imus)

    def __getitem__(self,index):
        imuPath=os.path.join(self.imuDir,self.imus[index])
        imuName=os.path.splitext(os.path.basename(imuPath))[0]
        imu=np.loadtxt(imuPath)

        imuDic={imuName:imu}

        return imuDic

class segDataset():
    def __init__(self,baseDir):
        self.segDir=os.path.join(baseDir,'image_00_cal_seg/')
        self.segs=os.listdir(self.segDir)
        self.segs = natsort.natsorted(self.segs)
        
    def __len__(self):
        return len(self.segs)

    def __getitem__(self,index):
        segPath=os.path.join(self.segDir,self.segs[index])
        segName=os.path.splitext(os.path.basename(segPath))[0]
        seg = np.array(Image.open(segPath).convert("L"), dtype=np.float32)

        segDic={segName:seg}
        
        return segDic

# 추가 데이터 필요할시 여기서 작업
class totalDataset(
    # imageDataset,
    # bbDataset,
    lidarDataset,
    imuDataset,
    # thermalDataset,
    # segDataset,
):
    def __init__(self,baseDir):
        self.imu = imuDataset(baseDir)
        # self.bb = bbDataset(baseDir)
        self.lidar = lidarDataset(baseDir)
        # self.image = imageDataset(baseDir)
        # self.thermal = thermalDataset(baseDir)
        # self.seg = segDataset(baseDir)
#         print("GNSS: {} Frames\nClassification: {} Frames\n\
# LiDAR: {} Frames\nImage: {} Frames\n\
# Thermal: {} Frames\nSegmentation: {} Frames".format(
#                     len(self.imu),
#                     len(self.bb),
#                     len(self.lidar),
#                     len(self.image),
#                     len(self.thermal),
#                     len(self.seg),
#                     ))
        print("IMU: {} Frames\n\
PointCloud: {} Frames\n".format(
                    len(self.imu),
                    # len(self.bb),
                    len(self.lidar),
                    # len(self.image),
                    # len(self.thermal),
                    # len(self.seg),
                    ))
        
        checkList = [
                    len(self.imu),
                    # len(self.bb),
                    len(self.lidar),
                    # len(self.image),
                    # len(self.thermal),
                    # len(self.seg),
                    ]
        item = np.unique(np.array(checkList))


        if item.reshape(-1,1).shape[0]>1:
            raise Exception("Data Number Does Not Match")

        print("=======================================================")
        print("*********************DATASET LOADED********************")
        print("=======================================================")



    def __len__(self):
        return len(self.imu)

    def __getitem__(self,index):


        # newData =  0) frameNumber; 1)imu; 2)image; 3)lidar; 4)bb; 5)thermal; 6)segmentation
        frameNumber=list(self.imu[index].keys())[0]
        
        newData=[]
        # 0) UPLOAD Frame Number
        newData.append(frameNumber)
        # print(
        #     self.imu[index],
        #     self.image[index],
        #     self.lidar[index],
        #     self.bb[index],
        #     self.thermal[index],
        #     self.seg[index],
        # )
        # 1) UPLOAD imu
        newData.append(self.imu[index][frameNumber])
        
        # # 2) UPLOAD Image
        # if frameNumber in self.image[index]:
        #     newData.append(self.image[index][frameNumber])
        # # else :
        # #     for x in self.image:
        # #         if frameNumber in x:
        # #             newData.append(x[frameNumber])
        # if len(newData)!=3:
        #     newData.append(-9999)

        # for x in self.image:
        #     if x[1]==frameNumber:
        #         newData.append(x[0])
        # # if there is no matching frame, append -9999
        # if len(newData)!=3:
        #     newData.append(-9999)
        
        # 3) UPLOAD Lidar
        if frameNumber in self.lidar[index]:
            newData.append(self.lidar[index][frameNumber])
        # else :
        #     for x in self.lidar:
        #         if frameNumber in x:
        #             newData.append(x[frameNumber])
        # if len(newData)!=4:
        #     newData.append(-9999)  

        # for x in self.lidar:
        #     if x[1]==frameNumber:
        #         newData.append(x[0])
        # # if there is no matching frame, append -9999
        # if len(newData)!=4:
        #     newData.append(-9999)        
        
        # 4) UPLOAD Bounding Box
        # if frameNumber in self.bb[index]:
        #     newData.append(self.bb[index][frameNumber])
        # # else :
        # #     for x in self.bb:
        # #         if frameNumber in x:
        # #             newData.append(x[frameNumber])
        # if len(newData)!=5:
        #     newData.append(-9999)

        # # for x in self.bb:
        # #     if x[1]==frameNumber:
        # #         newData.append(x[0])
        # # # if there is no matching frame, append -9999
        # # if len(newData)!=5:
        # #     newData.append(-9999)

        # # 5) UPLOAD Thermal
        # if frameNumber in self.thermal[index]:
        #     newData.append(self.thermal[index][frameNumber])
        # # else :
        #     # for x in self.thermal:
        #     #     if frameNumber in x:
        #     #         newData.append(x[frameNumber])
        # if len(newData)!=6:
        #     newData.append(-9999)    

        # # for x in self.thermal:
        # #     if x[1]==frameNumber:
        # #         newData.append(x[0])
        # # # if there is no matching frame, append -9999
        # # if len(newData)!=6:
        # #     newData.append(-9999)    

        # # 6) UPLOAD Segmentation
        # if frameNumber in self.seg[index]:
        #     newData.append(self.seg[index][frameNumber])
        # # else :
        # #     for x in self.seg:
        # #         if frameNumber in x:
        # #             newData.append(x[frameNumber])
        # if len(newData)!=7:
        #     newData.append(-9999)   
        
        # # for x in self.seg:
        # #     if x[1]==frameNumber:
        # #         newData.append(x[0])
        # # # if there is no matching frame, append -9999
        # # if len(newData)!=7:
        # #     newData.append(-9999)     
                            
        return newData

# Calibration Parameters Data
class calibrationData():
    def __init__(self,calibrationDir):
        with open(calibrationDir,'r') as file:
            b=file.readlines()

        '''Translation Parameters'''
        translationMatrix=np.array(
            [[float(b[6].split(",")[0])],
            [float(b[6].split(",")[1])],
            [float(b[6].split(",")[2])]],
            )

        '''Intrinsic Camera Parameters'''
        fx=float(b[8].split(",")[0])
        fy=float(b[8].split(",")[1])
        cx=float(b[8].split(",")[2])
        cy=float(b[8].split(",")[3])
        k1=float(b[8].split(",")[4])
        k2=float(b[8].split(",")[5])
        p1=float(b[8].split(",")[6])
        p2=float(b[8].split(",")[7])
        intrinsicParameters= np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        kpParameters=np.array([k1,k2,p1,p2])

        '''Rotation Matrix'''
        LR= float(b[4].split(",")[0])
        UD= float(b[4].split(",")[1])
        Rot=float(b[4].split(",")[2])
        
        unknownRotation=np.array(
            [[0,1,0],
            [0,0,-1],
            [-1,0,0]],
            )
        rollRotation=np.array(
            [[1,0,0],
            [0,math.cos(Rot),-math.sin(Rot)],
            [0,math.sin(Rot),math.cos(Rot)]],
            )
        pitchRotation=np.array(
            [[math.cos(UD),0,math.sin(UD)],
            [0,1,0],
            [-math.sin(UD),0,math.cos(UD)]],
            )
        yawRotation=np.array(
            [[math.cos(LR),-math.sin(LR),0],
            [math.sin(LR),math.cos(LR),0],
            [0,0,1]],
            )
        rotationMatrix= np.dot(np.dot(np.dot(unknownRotation,rollRotation),pitchRotation),yawRotation)
        
        calibrationParameters=[
            translationMatrix, 
            intrinsicParameters, 
            rotationMatrix, 
            kpParameters,
            ]
        
        self.calibrationParameters=calibrationParameters

    def getData(self):
        return self.calibrationParameters



'''Thermal Dataset for Bin File'''
class thermalDatasetBinVersion():
    def __init__(self,baseDir):
        self.thermalDir=os.path.join(baseDir,'image_02_cal/')
        self.thermals=os.listdir(self.thermalDir)
    
    def __len__(self):
        return len(self.thermals)

    def raw_to_celsius(self,bin,B_,R_,O_,F_,kT_0):
        return (B_ / math.log(R_ / (bin - O_) + F_) - kT_0)
 
    def __getitem__(self,index):
        thermalPath=os.path.join(self.thermalDir,self.thermals[index])
        thermalName=os.path.splitext(os.path.basename(thermalPath))[0]
        thermalVal=os.path.splitext(os.path.basename(thermalPath))
        if thermalVal[1]!='.bin':
            thermal = -9999
        else :  
            with open (thermalPath,'rb') as f:
                rawData= f.read()
                dt = np.dtype(np.int16)
                dt = dt.newbyteorder('little')
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
            thermal = raw_to_celsius_vec(bin,B_,R_,O_,F_,kT_0)

        thermalDic={thermalName:thermal}
        
        return thermalDic


