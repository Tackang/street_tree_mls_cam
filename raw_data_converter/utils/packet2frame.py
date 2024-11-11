from utils.dataset import totalDataset
import numpy as np
import os
import multiprocessing
import parmap
import math
from scipy.spatial.transform import Rotation as R

def getTransform(imu):
    x=float(imu[1])
    y=float(imu[2])
    z=float(imu[3])
    roll=math.radians(float(imu[4]))
    pitch=math.radians(float(imu[5]))
    yaw=math.radians(float(imu[6]))
    buffer = np.array([0,0,0,1])

    axisChangeWorld=np.array([[0,1,-0],[1,-0,0],[0,-0,-1]])
    # axisChangeWorld_Inverse =axisChangeWorld.T

    rotationCore =  np.dot(
        axisChangeWorld,
        R.from_euler('xyz', [roll, pitch, yaw], degrees=0).as_matrix()
    )
    rotationCore_Inverse =  rotationCore.T
    translation = np.array([x, y, z]).reshape(-1,1)
    translation_Inverse = -translation

    imu2world = np.vstack(
            (np.hstack((rotationCore,translation)),buffer)
        )
    
    world2imu = np.vstack(
        (np.hstack((
            rotationCore_Inverse, 
            np.dot(rotationCore_Inverse,translation_Inverse)
            )),buffer)
        )

    lidar2imu=np.vstack(
        (np.array([[-0.999555,-0.0296537,0.0032518,0.00762325],
        [-0.00313653,-0.00393283,-0.999987,-0.000283261],
        [0.0296662,-0.999552,0.00383807,-0.0802608],]),
        buffer
        ))

    imu2lidar = np.vstack(
        (
        np.hstack((lidar2imu[:3,:3].T,-np.dot(lidar2imu[:3,:3].T,lidar2imu[:3,3]).reshape(-1,1))),
        buffer
        )
        )

    world2lidar = np.dot(imu2lidar, world2imu)    
    lidar2world = np.dot(imu2world, lidar2imu)   

    return [lidar2world, world2lidar]

def packet2frame(index, totalData, tkysPointCloudPacketDir):
    # 20개씩 인덱싱 : input 0 :0~19
    coreValue =10
    startFrame = index*20
    for coreValue in range(10,20):
        pointCloudCore = totalData[startFrame + coreValue][2]
        if pointCloudCore.size == 0:
            continue
        else:
            nameCore = totalData[startFrame + coreValue][0]
            imuCore = totalData[startFrame + coreValue][1]
            TransformationListCore = getTransform(imuCore)
            world2lidarCore = TransformationListCore[1]
            break      

    if pointCloudCore.size == 0:
        for coreValue in range(0,10): 
            pointCloudCore = totalData[startFrame + coreValue][2]
            if pointCloudCore.size == 0:
                continue
            else:
                nameCore = totalData[startFrame + coreValue][0]
                imuCore = totalData[startFrame + coreValue][1]
                TransformationListCore = getTransform(imuCore)
                world2lidarCore = TransformationListCore[1]
                break      
  
    if pointCloudCore.size == 0:
        print("No Core", startFrame)
        return

    for i in range(startFrame, startFrame+20):
        if i == startFrame + coreValue:
            continue

        pointCloud = totalData[i][2]
        if pointCloud.size == 0:
            continue
        if pointCloud.ndim == 1:
            pointCloud = pointCloud.reshape(1,-1)

        imu = totalData[i][1]
        transformationList = getTransform(imu)
        lidar2world = transformationList[0]

        pcBuff = np.ones((pointCloud.shape[0],4)).reshape(-1,4)

        pcBuff[:,:3]=pointCloud[:,:3]
        pc2core = np.dot(
            world2lidarCore,
            np.dot(lidar2world,pcBuff.T)
        ).T

        pc2core = np.hstack((
            pc2core[:,:3],
            pointCloud[:,3:]
        ))

        pointCloudCore = np.vstack((pointCloudCore,pc2core))
    
    binary = pointCloudCore.tobytes()
    with open(os.path.join(tkysPointCloudPacketDir,"{}.bin".format(nameCore)),'wb') as f:
        f.write(binary)

    # np.savetxt(os.path.join(tkysPointCloudPacketDir,"{}.txt".format(nameCore)), pointCloudCore)

def main(baseDir, tkysPointCloudPacketDir):

    # totalData: 0)frameNumber; 1)imu; 2)lidar
    totalData = totalDataset(baseDir)
    dataLength = len(totalData)//20

    pool =  multiprocessing.Pool(processes = 50)
    result = parmap.map(packet2frame, range(dataLength), totalData, tkysPointCloudPacketDir, pm_pbar=True, pm_processes = 15)
