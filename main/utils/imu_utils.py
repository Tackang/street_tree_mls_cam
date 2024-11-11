import numpy as np
from scipy.spatial.transform import Rotation as R

def getTransform(imu):
    '''
    ### Notice
    get transformation matrix from the IMU data(include transformation from lidar to IMU). index0=toWorld, index1=inverse
    '''
    x=float(imu[1])
    y=float(imu[2])
    z=float(imu[3])
    roll=np.deg2rad(float(imu[4]))
    pitch=np.deg2rad(float(imu[5]))
    yaw=np.deg2rad(float(imu[6]))
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

def getTransformedPointCloud(pointCloud,imu,imageImu):
    '''
    ### Notice
    get point cloud transformed from point cloud taken time to image taken time
    '''
    pc_transform = getTransform(imu)
    image_transform = getTransform(imageImu)

    pcBuff = np.ones((pointCloud.shape[0],4)).reshape(-1,4)

    pcBuff[:,:3]=pointCloud[:,:3]
    pc2core = np.dot(
        image_transform[1],
        np.dot(pc_transform[0],pcBuff.T)
    ).T

    pc2core = np.hstack((
        pc2core[:,:3],
        pointCloud[:,3:]
    ))

    return pc2core

def world2PointCloud(pointCloud,imu,imageImu):
    '''
    ### Notice
    Detected tree position to the pointcloud of image
    '''
    pc_transform = getTransform(imu)
    image_transform = getTransform(imageImu)

    pcBuff = np.ones((pointCloud.shape[0],4)).reshape(-1,4)

    pcBuff[:,:3]=pointCloud[:,:3]
    pc2core = np.dot(
        image_transform[1],pcBuff.T
    ).T

    return pc2core[:,:3]

def getWorldPointCloud(pointCloud,imu):
    '''
    ### Notice
    Last three coordinates are world coordinates
    '''
    
    transformMatrix =getTransform(imu)
    pcBuff = np.ones((pointCloud.shape[0],4)).reshape(-1,4)
    pcBuff[:,:3]=pointCloud[:,:3]
    pc2world = np.dot(transformMatrix[0],pcBuff.T).T
    pc2world = np.hstack((
        pointCloud,
        pc2world[:,:3],
    ))

    return pc2world
