from dataset.mms_dataset import mmsDataset
import utils.pointCloud_utils as pointCloud_utils
import utils.lidar_cam_utils as lidar_cam_utils
import utils.lidar_lwircam_utils as lidar_lwircam_utils
import utils.imu_utils as imu_utils
import os
import numpy as np
import parmap
import traceback
import sys
import time
import yaml
import datetime
import pytz

def find_error_function(tb, function_names):
    for frame in reversed(tb):
        if frame.name in function_names:
            return frame.name
    return "Unknown"


def main(index,mmsData):
    # index = 1000

    lidarName = mmsData[index][0]
    imu = mmsData[index][1]
    pointCloud = mmsData[index][2]
    image = mmsData[index][3]
    lwirImage = mmsData[index][4]
    imageImu = mmsData[index][5]
    # lwirImageImu = mmsData[index][6]
    seg = mmsData[index][7]
    bb = mmsData[index][8]
    
    try:
        # pointCloud preprocess
        # pointcloud shape : (n,6) at the beginning
        pointCloud, ground, dsmPointCloud = pointCloud_utils.groundFilter(pointCloud_utils.distanceFilter(pointCloud),return_ground=True)
        # pointcloud shape : (n,5) after here
        np.save(os.path.join(singleFrameGroundFolder, '{}.npy'.format(lidarName)),ground)
        np.save(os.path.join(singleFrameDsmFolder, '{}.npy'.format(lidarName)),dsmPointCloud)
        del ground,dsmPointCloud
        pointCloud = imu_utils.getTransformedPointCloud(pointCloud,imu,imageImu)
        # pointcloud shape : (n,5) after here(x,y,x transformed to image time coordinate)
        # get image info
        pointCloud = lidar_cam_utils.lidar2cam(pointCloud)
        # pointcloud shape : (n,7) after here(pixel u,v updated)

        pointCloud = lidar_cam_utils.getSegInfo(pointCloud, seg)
        pointCloud = lidar_cam_utils.getRGBInfo(pointCloud, image)
        # pointcloud shape : (n,10) after here(RGB updated)
        pointCloud = lidar_lwircam_utils.lidar2lwircam(pointCloud)
        # pointcloud shape : (n,12) after here(lwir pixel u,v updated)
        pointCloud = lidar_lwircam_utils.getThermalInfo(pointCloud, lwirImage)
        # pointcloud shape : (n,13) after here(temperature updated)
        pointCloud = np.delete(pointCloud,(-3,-2),axis=1)
        # pointcloud shape : (n,11) after here(lwir pixel u,v removed)
        pointCloud = pointCloud_utils.dbscan(pointCloud)
        # pointcloud shape : (n,12) after here(cluster ID updated)
        pointCloud = pointCloud_utils.removeShortCluster(pointCloud)
        if pointCloud.size != 0 :
            pointCloud,clusterCenterDict = pointCloud_utils.divideCluster(pointCloud)
            pointCloud = pointCloud_utils.removeShortCluster(pointCloud)
        if pointCloud.size != 0 :
            pointCloud = lidar_cam_utils.getBBinfo(pointCloud,bb)
            # pointcloud shape : (n,13) after here(Species information updated)
            pointCloud = np.delete(pointCloud,(-8,-7),axis=1)
            # pointcloud shape : (n,11) after here(pixel u,v removed)

            np.save(os.path.join(singleFrameTreeFolder, '{}.npy'.format(lidarName)),pointCloud)
            clusterCenterArr = np.hstack((
                np.array(list(clusterCenterDict.keys())).reshape(-1,1),
                np.array(list(clusterCenterDict.values()))
            ))
            np.save(os.path.join(singleFrameCenterFolder,'{}.npy'.format(lidarName)),clusterCenterArr)
            del pointCloud,clusterCenterArr

    except Exception as e:
        error_msg = str(e)
        tb = traceback.extract_tb(sys.exc_info()[2])
        function_names = [
            'groundFilter',
            'distanceFilter',
            'getDhm',
            'getTransformedPointCloud',
            'lidar2cam',
            'getSegInfo',
            'getRGBInfo',
            'lidar2lwircam',
            'getThermalInfo',
            'dbscan',
            'removeShortCluster',
            'divideCluster',
            'getBBinfo',
            'savetxt',
            'hstack',
        ]  # Add other function names if needed
        error_function = find_error_function(tb, function_names)

        with open(logFileName, 'a') as log:
            log.write(f'Error in function: {error_function}, frame number: {lidarName}, image size: {type(image)}, lwir size: {type(lwirImage)}\nError message: {error_msg}\n')
    # progress_bar.update(1)

def convert_seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return int(hours), int(minutes), int(seconds)
   
if __name__ == '__main__':

    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for single frame processing")
    print(folderList)

    # log = []
    for idx in range(len(folderList)):
        print(f"Single frame processing for {folderList[idx]}")
        
        start_time = time.time()
        resultPath = os.path.join(rootPath,folderList[idx],"result")
        os.makedirs(resultPath, exist_ok = True)
        dataPath= os.path.join(rootPath,folderList[idx],"preprocessed_data")

        mmsData = mmsDataset(dataPath) # mmsdata = [lidarName, imu, lidar, image, thermal, imageImu, lwirImu, seg, bb]


        singleFrameTreeFolder = os.path.join(resultPath,'tree_singleFrame')
        singleFrameCenterFolder = os.path.join(resultPath,'center_singleFrame')
        singleFrameGroundFolder = os.path.join(resultPath,'ground_singleFrame')
        singleFrameDsmFolder = os.path.join(resultPath,'dsm_singleFrame')
        logFolder = os.path.join(resultPath,'log')
        os.makedirs(logFolder, exist_ok = True)
        logFileName = os.path.join(logFolder,'launch_sf_log.txt')

        os.makedirs(singleFrameTreeFolder, exist_ok = True)
        os.makedirs(singleFrameCenterFolder, exist_ok = True)
        os.makedirs(singleFrameGroundFolder, exist_ok = True)
        os.makedirs(singleFrameDsmFolder, exist_ok = True)
        
        # progress_bar = tqdm(total=len(tkysData), desc="Processing", unit="Frame")
        NUM_WORKERS= 50
        parmap.map(
            main,
            range(len(mmsData)), mmsData, pm_pbar=True,
            pm_processes = NUM_WORKERS,
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, minutes, seconds = convert_seconds_to_hms(elapsed_time)
        with open(logFileName, 'a') as log:
            log.write(f'Finished. Processing time: {hours}h {minutes}m {seconds}s\n')
    # log.append(result)
    # print(log)

