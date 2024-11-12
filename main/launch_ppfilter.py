import numpy as np
from scipy.stats import mode
# import open3d as o3d
from tqdm import tqdm
import os
import parmap
from sklearn.cluster import DBSCAN
import sys
import yaml 

def dbscan(pointCloud,EPS=1,MIN_POINTS=10):
  
    pointCloud2=pointCloud[:,-3:]

    db = DBSCAN(eps=EPS, min_samples=MIN_POINTS, n_jobs=1).fit(pointCloud2)    #########################DBSCAN parameter
    labels = db.labels_
    pointCloud = np.hstack((pointCloud,labels.reshape(-1,1)))
    modeVal = mode(labels,keepdims=True)
    pointCloud_output = pointCloud[pointCloud[:,-1]==modeVal[0],:-1]
    del pointCloud, pointCloud2, labels, modeVal
    return pointCloud_output

def treeRefinement(pointCloud):
    clusterID = np.unique(pointCloud[:,-5])
    # print(clusterID.min(),clusterID.max(),clusterID.shape)
    # pointNumber = []
    newPointCloud = []
    centerList = []
    for ID in tqdm(clusterID) :
        # currentCenter = center[center[:,0]==ID,:]
        roi = np.where(pointCloud[:,-5]==ID)[0]
        pc_roi = pointCloud[roi,:]
        try:
            pc_roi= dbscan(pc_roi)
        except:
            print(ID)
        pc_roi, centerDict = divideCluster(pc_roi)
        if (pc_roi is None) or (centerDict is None):
            continue
        else:

            newPointCloud.append(pc_roi)
            clusterCenterArr = np.hstack((
                np.array(list(centerDict.values())),
                np.array(list(centerDict.keys())).reshape(-1,1)
            ))

            centerList.append(clusterCenterArr.reshape(4))
            del pc_roi, centerDict, clusterCenterArr


        # center[center[:,0]==ID,-1] = pc_roi.shape[0]

    total = np.concatenate(newPointCloud,axis = 0)
    center = np.array(centerList)
    # center_cols_change=np.hstack((center[:,(1,2,3)],center[:,0].reshape(-1,1),center[:,-1].reshape(-1,1)))
    # center_cols_change=np.hstack((center[:,(1,2,3)],center[:,0].reshape(-1,1)))
    # del pointCloud, center
    return total,center
    # return pointCloud[pointCloud[:,-5]!=-9999,:],center_cols_change

def divideCluster(pointCloud, VOXEL_SIZE = 0.3):

    clusterCenterDict = {}
    # pc_roi = world coord of point cloud
    pc_roi = pointCloud[:,-3:]
    # currentID = clusterID of point cloud
    currentID = np.unique(pointCloud[:,-5])
    # skip if cluster is empty
    if pc_roi.size == 0:
        return None,None
    # make grid and add grid Id to the last column
    # x, y, z, intensity, laserID, pixelu, pixelv, R, G, B, temperature, clusterID, gridID
    x= np.floor((pc_roi[:,0]-np.min(pc_roi[:,0]))/VOXEL_SIZE)
    y= np.floor((pc_roi[:,1]-np.min(pc_roi[:,1]))/VOXEL_SIZE)
    # z = pc_roi[:,2]-np.min(pc_roi[:,2])
    THRESHOLD = np.min(pc_roi[:,2]) + 1

    gridID = 100000*x + y
    # pc_roi has grid ID (n,4) from here
    pc_roi = np.hstack((pc_roi,gridID.reshape(-1,1)))
    cellID=np.unique(gridID)
    minGridZ = np.zeros((pc_roi.shape[0],1))
    # Calcalute min z value of grid and add it to the last column
    # x, y, z, intensity, laserID, pixelu, pixelv, R, G, B, temperature, clusterID, gridID, mixZ
    for g in cellID:
        roi2 = np.where(pc_roi[:,-1]==g)[0]
        minGridZ[roi2] = np.min(pc_roi[roi2,2])
    # pc_roi has min z value of grid (n,5) from here
    pc_roi = np.hstack((pc_roi,minGridZ))
    
    #make meshgrid (n+2,n+2)
    x_,y_ = np.meshgrid(np.arange(x.max()+1),np.arange(y.max()+1))
    # (n,n)array having 1~n
    x_window = x_[1:-1,1:-1]
    y_window = y_[1:-1,1:-1]
    # make zPlane having same size with meshgrid without first and last row or column.(n,n)
    zPlane = np.full(x_.shape,9999).astype(np.float64)
    rows, cols = zPlane.shape
    # skip if zPlane is smaller than 3,3
    if rows < 3 and cols < 3:
        return None,None

    #Put min z values to zPlane if min z value is smaller than threshold
    for g in range(0,pc_roi.shape[0]):
        if pc_roi[g,-1]<=THRESHOLD:
            zPlane[y[g].astype(np.int16),x[g].astype(np.int16)] = pc_roi[g,-1]
    
    # index list window to visit
    windowArr = np.vstack((y_window.ravel(),x_window.ravel())).T
    count = 0
    clusterCenterList = []
    # visit every window center
    for g in range(0,len(x_window.ravel())):
        windowCenter = windowArr[g].astype(np.int16)
        window = zPlane[(windowCenter[0]-1):(windowCenter[0]+2),(windowCenter[1]-1):(windowCenter[1]+2)]
        if np.argmin(window)==4:
            count+=1
            # grid ID which is defined as center 
            gridIDforSearch  = (100000*windowCenter[1]+windowCenter[0])
            # in cluster, in target grid, choose point that has minimum z value, and append to clusterCenterList
            pc_targetGrid = pc_roi[pc_roi[:,-2]==gridIDforSearch,:]
            try:
                clusterCenterList.append(pc_targetGrid[np.argmin(pc_targetGrid[:,2]),:3])
            except:
                print(x.max(),y.max(), x.min(),y.min())
                print(x_window.max(), x_window.min())
                print(y_window.max(), y_window.min())
                print(windowArr[g])
                print(windowCenter)
                print(windowCenter[1])
                print(windowCenter[0])
                print(windowArr)
                print(gridIDforSearch)
                print(pc_targetGrid)
                print(window)
      
    # if center is not detected, change clusterID to 9999
    if count == 0:
        return None,None
    
    # if a single center is detected, use original clusterID again 
    if count ==1 :
        # clusterCenterDict is clusterID: clusterCenterx,y,z
        clusterCenterDict[currentID[0]]= clusterCenterList[0]
        return pointCloud, clusterCenterDict
    
    if count >=2:
        # distanceList is (the number of point in cluster, the number of count)
        # calculate distance between center and all other points
        distanceList = np.zeros((pc_roi.shape[0],count))
        for h in range(len(clusterCenterList)):
            distance = np.sqrt(np.sum((pointCloud[:,-3:]-clusterCenterList[h].reshape(1,-1))**2,axis=1))
            distanceList[:,h] = distance
        # new cluserID is (the number of points in cluster,1) where the value is newClusterID
        newClusterID = np.argmin(distanceList,axis=1)+currentID
        pointCloud[:,-5] = newClusterID
        unique_newclusterID, counts = np.unique(newClusterID,return_counts=True)
        checker = np.hstack((unique_newclusterID.reshape(-1,1),counts.reshape(-1,1)))
        validID = checker[np.argmax(checker[:,1]),0]
        pointCloud = pointCloud[pointCloud[:,-5]==validID,:]
        pointCloud[:,-5] = currentID
        clusterCenterDict[currentID[0]]= clusterCenterList[np.argmax(checker[:,1])]
        
        return pointCloud, clusterCenterDict
        
  

def main_par(idx, folderList):

    resultPath = os.path.join(rootPath,folderList[idx],"result")

    lidarPath = os.path.join(resultPath,"tree_multiframe.txt")
    # centerPath = os.path.join(mapDir,"center_RGBT_230428.txt")

    pointCloud = np.loadtxt(lidarPath)
    # center = np.loadtxt(centerPath)

    pointCloud, center = treeRefinement(pointCloud)
    np.savetxt(os.path.join(resultPath, 'tree_multiframe_ppfilter.txt'),pointCloud)
    np.savetxt(os.path.join(resultPath, 'center_multiframe_ppfilter.txt'),center)
    del pointCloud, center
    print(folderList[idx], 'Done')
    # distance = calculateDistance(center)
    # removeCenterList = getRemoveCenterList(center,distance)
    # removeNoiseCluster(center,pointCloud,removeCenterList,mapDir)


if __name__ =="__main__":
    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for Pseudo-plane filter")
    print(folderList)

    for idx in range(len(folderList)):      
        main_par(idx,folderList) 