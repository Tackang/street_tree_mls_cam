import sys
from dataset.singleframe_dataset import sfDataset
import utils.pointCloud_utils as pointCloud_utils
import utils.lidar_cam_utils as lidar_cam_utils
import utils.imu_utils as imu_utils
import os
import numpy as np
import parmap
import traceback
import time
import yaml

def find_error_function(tb, function_names):
    for frame in reversed(tb):
        if frame.name in function_names:
            return frame.name
    return "Unknown"



def main(pointCloud, imageImu, center,
         centerArr, count, pointList,
         THRESHOLD=1):
    if pointCloud.size==0: # empty point cloud skip
        del pointCloud,center,imageImu
        return centerArr,count,pointList
    if pointCloud.ndim == 1: # 1 dimensional point cloud skip(point number= 1)
        del pointCloud,center,imageImu
        return centerArr,count,pointList
        
    if center.ndim == 1 : # when only one cluster exists
        center = center.reshape(1,-1) 

    # give centers world coordinates and add the number of count column   
    center[:,1:] = imu_utils.getWorldPointCloud(center[:,1:],imageImu)[:,-3:]
    clusterCount = np.full((center.shape[0]),1)
    center = np.hstack((center,clusterCount.reshape(-1,1))) 

    # make initial pointList and centerArr at the first iteration
    if count == 0 :
        count +=1 
        pointCloud = imu_utils.getWorldPointCloud(pointCloud,imageImu)
        pointList.append(pointCloud)
        centerArr=center
        del pointCloud,center,imageImu
        return centerArr,count,pointList
    
    # process after the init
    if count >= 1 :
        # existing cluster IDs 
        currentClusterID = np.unique(centerArr[:,0])
        # new cluster ID to give
        newClusterID = np.max(currentClusterID)+1

        # change input center and pointclusters' cluster ID to negative value
        center[:,0] = -center[:,0]-1
        pointCloud[:,-2]= -pointCloud[:,-2]-1

        # calculate distance between incoming cluster center and existing cluster center
        distance = np.sqrt(
            (center[:,1][:,np.newaxis]-centerArr[:,1][np.newaxis,:])**2+
            (center[:,2][:,np.newaxis]-centerArr[:,2][np.newaxis,:])**2
            )

        # Choose the cluster groups within the Threshold
        roi = np.where(distance<=THRESHOLD)
        # changeArr 0) incoming cluster 1) existing cluster 2) distance between income&current
        changeArr = np.vstack((center[roi[0],0], centerArr[roi[1],0])).T
        changeArr = np.hstack((changeArr,distance[roi[0],roi[1]].reshape(-1,1)))

        # Incoming cluster가 두곳에 가서 붙으려는 현상 필터
        currentCenter, countCenter = np.unique(changeArr[:,0], return_counts=True)
        if np.any(countCenter>=2):
            roi2 = np.where(countCenter>=2)
            # center_roi2 is ID of input cluster close to more than two existing centers
            center_roi2 = currentCenter[roi2] 

            for g in center_roi2:
                inputCenter = g
                # changeArr of target center 
                targetCenter = changeArr[changeArr[:,0]==inputCenter,:] 
                closestTarget = targetCenter[targetCenter[:,-1]==targetCenter[:,-1].min(),1][0]
                changeArr[changeArr[:,0]==inputCenter,1] = closestTarget
        
        # Input Cluster --> Target 변경
        for i in range(changeArr.shape[0]):
            source, target = changeArr[i,:2]
            # target = changeArr[i,1]
            pointCloud[pointCloud[:,-2]==source,-2] = target # 기존 포인트 클라우드에서 source2target
            center[center[:,0]==source,0] = target # 기존 input center에서 source2target
            centerArr[centerArr[:,0]==target,-1] += 1 # changeArr에서 타겟클러스터 카운팅
         
            # centerArr에 center 업데이트부
            # if more than two input clusters go to one existing cluster, there will be the same cluster ID in center variable
            # update centerArr if input clusters' center is lower than existing center. 
            if center[center[:,0]==target,-2].min() <= centerArr[centerArr[:,0]==target,-2]:
                minIndex = np.argmin(center[center[:,0]==target,-2])
                centerArr[centerArr[:,0]==target,:-1] = center[center[:,0]==target,:-1][minIndex] 

        # 남는 cluster 유지
        for i in range(center.shape[0]):
            if center[i,0]<0:
                pointCloud[pointCloud[:,-2]==center[i,0],-2]=newClusterID
                center[i,0]=newClusterID
                centerArr = np.vstack((centerArr,center[i,:]))
                newClusterID += 1

        pointCloud = imu_utils.getWorldPointCloud(pointCloud,imageImu)
        pointList.append(pointCloud)
        count ==1
        del pointCloud,center,imageImu
        return centerArr, count, pointList

def convert_seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return int(hours), int(minutes), int(seconds)

def main_par(idx, folderList,rootPath):
    dataPath= os.path.join(rootPath,folderList[idx],"preprocessed_data")
    resultPath = os.path.join(rootPath,folderList[idx],"result")
    sfData = sfDataset(resultPath)
    logFolder = os.path.join(resultPath,'log')
    os.makedirs(logFolder, exist_ok = True)
    logFileName = os.path.join(logFolder,'launch_mf_log.txt')
   
    centerArr = np.zeros(4)
    count =0
    pointList = []
    
    start_time = time.time()
    for index in range(len(sfData)):
            
        lidarName = sfData[index][0]
        try:
            pointCloud = sfData[index][1]
            imageImu = sfData[index][2]
            center = sfData[index][3]
            centerArr, count,pointList = main(pointCloud,imageImu,center,centerArr,count,pointList)

        except Exception as e:
            error_msg = str(e)
            tb = traceback.extract_tb(sys.exc_info()[2])
            function_names = ['sfData', 'main']  # Add other function names if needed
            error_function = find_error_function(tb, function_names)
            with open(logFileName, 'a') as log:
                log.write(f'Error in function: {error_function}, frame number: {lidarName}\nError message: {error_msg}\n')
            continue

        
    # invalidCluster = centerArr[centerArr[:,-1]<=3,0]
    total = np.concatenate(pointList,axis = 0)

    # for id_invalid in invalidCluster:
    #     total = total[total[:,-5]!=id_invalid,:]
    #     centerArr = np.delete(centerArr,np.where(centerArr[:,0]==id_invalid)[0],axis=0)

    np.savetxt(os.path.join(resultPath,"tree_multiframe.txt"),total)
    np.savetxt(os.path.join(resultPath,"center_multiframe.txt"),centerArr)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = convert_seconds_to_hms(elapsed_time)
    with open(logFileName, 'a') as log:
        log.write(f'Finished. Processing time: {hours}h {minutes}m {seconds}s\n')


if __name__ == '__main__':
    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for single frame to multi frame")
    print(folderList)
        
    NUM_WORKERS= 1
    parmap.map(
        main_par,
        range(len(folderList)),folderList,rootPath, 
        pm_pbar=True, pm_processes = NUM_WORKERS,
        )

