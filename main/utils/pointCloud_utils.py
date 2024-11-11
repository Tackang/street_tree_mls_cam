import numpy as np
# import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull

def distanceFilter(pointCloud, MIN_DISTANCE=3.5, MAX_DISTANCE=80):
    xyDistance=np.sqrt(pointCloud[:,0]**2+pointCloud[:,1]**2)
    mask=np.where((xyDistance<=MAX_DISTANCE)&(xyDistance>=MIN_DISTANCE))
    
    return pointCloud[mask]

def groundFilter(pointCloud, GRID_SIZE=0.5, GROUND_THICKNESS=0.25, THRESHOLD = 0.25, return_ground = False):
    '''
    ### Notice
    Return ground removed point cloud. If return_groud=True, there is the second output and it is a array of ground points.
    '''
    pointCloud2Process=pointCloud[:,0:5]
    zMinMaxAverage=np.zeros((pointCloud2Process.shape[0],2))

    '''2D gridID one-hot encoding'''
    gridID = (np.floor(pointCloud2Process[:,0]/GRID_SIZE)+(80/GRID_SIZE))*100000\
            +(np.floor(pointCloud2Process[:,1]/GRID_SIZE)+(80/GRID_SIZE))

    '''pointCloudInBoundingBox columns: 
    0)x; 1)y; 2)z; 3)intensity; 4)gridID(cellID); 5)elevation; 6)zAverage'''
    pointCloudGrid=np.hstack((pointCloud2Process,gridID.reshape(-1,1)))
    pointCloudGrid=np.hstack((pointCloudGrid,zMinMaxAverage))

    cellInfo=np.unique(gridID)
    highestPoints = []

    '''z values calculation'''
    for cellID in cellInfo:
 
        arg = np.where(pointCloudGrid[:,-3]==cellID)
        topPointArg =np.argmax(pointCloudGrid[arg,2])
        topPoint = pointCloudGrid[arg][topPointArg]
        if topPoint[2] > 0:
            highestPoints.append(topPoint)
        
        pointCloudGrid[arg[0],-2:]=\
            [
            np.max(pointCloudGrid[arg[0],2])-np.min(pointCloudGrid[arg[0],2]),
            np.mean(pointCloudGrid[arg[0],2]),
            ]
        '''Filter ground cell'''
        # cell의 모든 component에서 같은 cell Elevation이 검출되므로  unique 필요
        cellElevation=np.unique(pointCloudGrid[arg[0],-2])
        
        '''Remove Empty Cell'''           
        if cellElevation.size==0:
            continue

        elif cellElevation<=GROUND_THICKNESS and \
            np.mean(pointCloudGrid[arg[0],-1])<=-1:
            pointCloudGrid[arg[0],-3]=0
            
        else: 
            continue
    groundAverage = np.mean(pointCloudGrid[pointCloudGrid[:,-3]==0,-1])
    dhm_point_cloud = np.array(highestPoints)
    
    # ground point = 0, non-ground points = 1
    if return_ground == True:
        pointCloudGrid[pointCloudGrid[:,2]<=groundAverage+THRESHOLD,-3] = 0
        pointCloudGrid[pointCloudGrid[:,-3]!=0,-3] = 1
        '''Remove Carpet-Like Ground'''
        pointCloud = pointCloudGrid[:,:-2]
        return pointCloud[pointCloud[:,-1]==1,:-1], pointCloud[pointCloud[:,-1]==0,:-1], dhm_point_cloud

    if return_ground == False:
        pointCloudGroundFiltered=pointCloudGrid[pointCloudGrid[:,-3]!=0,:-3]
        '''Remove Carpet-Like Ground'''
        pointCloud = pointCloudGroundFiltered[
            pointCloudGroundFiltered[:,2]>groundAverage+THRESHOLD,:
            ]
        return pointCloud,dhm_point_cloud
    
def getDhm(pointCloud, GRID_SIZE = 0.3):
    '''
    ### Notice
    Return DHM point cloud. 
    '''
    # zMinMaxAverage=np.zeros((pointCloud.shape[0],1))

    '''2D gridID one-hot encoding'''
    gridID = (np.floor(pointCloud[:,0]/GRID_SIZE)+(80/GRID_SIZE))*100000\
            +(np.floor(pointCloud[:,1]/GRID_SIZE)+(80/GRID_SIZE))

    '''pointCloudInBoundingBox columns: 
    0)x; 1)y; 2)z; 3)intensity; 4)gridID(cellID); 5)elevation; 6)zAverage'''
    pointCloudGrid=np.hstack((pointCloud,gridID.reshape(-1,1)))
    # pointCloudGrid=np.hstack((pointCloudGrid,zMinMaxAverage))
    
    cellInfo=np.unique(gridID)
    highestPoints = []
    for cellID in cellInfo:
        arg = np.where(pointCloudGrid[:,-1]==cellID)
        topPointArg =np.argmax(pointCloudGrid[arg,2])[0]
        topPoint = pointCloudGrid[arg][topPointArg]
        if topPoint[2] > 0:
            highestPoints.append(topPoint)

    # Convert the list of highest points to a NumPy array
    dhm_point_cloud = np.array(highestPoints)

    return dhm_point_cloud 

 

def dbscan(pointCloud,EPS=1.9,MIN_POINTS=10):
    '''
    ### Notice
    Add cluster ID at the last columns
    '''
    pointCloudGroundFiltered=pointCloud[:,:3]
    pointNumber = pointCloudGroundFiltered.shape[0]

    if pointNumber <= MIN_POINTS:
        pointCloudCluster = np.hstack((pointCloudGroundFiltered,np.full((pointNumber,1),-1)))        
    #     return pointCloudCluster
    else: 
        if np.any(np.isnan(pointCloudGroundFiltered)):
            return pointCloud    

        db = DBSCAN(eps=EPS, min_samples=MIN_POINTS, n_jobs=1).fit(pointCloudGroundFiltered)    #########################DBSCAN parameter
        labels = db.labels_
        pointCloudCluster=np.hstack((pointCloud,labels.reshape(-1,1)))
        pointCloudCluster=pointCloudCluster[pointCloudCluster[:,-1]!=-1,:]
        # Number of clusters in labels, ignoring noise if present.
        
    # pointCloudCluster columns: 0)x; 1)y; 2)z; 3)intensity; 4)label; 5)pixelU; 6)pixelV; 7)clusterID
    
    return pointCloudCluster


def removeShortCluster(pointCloud, CLUSTER_HEIGHT= 2.5):
    '''
    ## Notice
    remove clusters that has shorter height than threshold
    '''
    pointCloudCluster=pointCloud
    # filtering too short cluster
    clusterID=np.unique(pointCloudCluster[:,-1])
    for i in range(0,clusterID.shape[0]):
        singleCluster=pointCloudCluster[pointCloudCluster[:,-1]==clusterID[i],:]
        clusterHeight=np.max(singleCluster[:,2])-np.min(singleCluster[:,2])
        if clusterHeight<=CLUSTER_HEIGHT:
            pointCloudCluster[pointCloudCluster[:,-1]==clusterID[i],-1]=-1
        if np.median(singleCluster[:,2])<0:
            pointCloudCluster[pointCloudCluster[:,-1]==clusterID[i],-1]=-1

    # pointCloudCluster columns: 0)x; 1)y; 2)z; 3)intensity; 4)label; 5)pixelU; 6)pixelV; 7)clusterID
    pointCloudClusterFiltered=pointCloudCluster[pointCloudCluster[:,-1]!=-1,:]

    return pointCloudClusterFiltered

# def removeNoise(pointCloud,VOXEL_SIZE=0.2,K_NEIGHBORS=3,STD_RATIO=0.05):
#     clusterID=np.unique(pointCloud[:,-1])
#     # print(pointCloud[pointCloud[:,-1]==0,:].shape)
#     for i in range(0,clusterID.shape[0]):
#         pointCloud[pointCloud[:,-1]==clusterID[i],-1] = -1
#         roi = np.where(pointCloud[:,-1]==-1)[0]
        
#         pcd=o3d.geometry.PointCloud()
#         pcd.points=o3d.utility.Vector3dVector(pointCloud[roi,0:3])
#         pcdDownSampled=pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
#         cl, ind= pcdDownSampled.remove_statistical_outlier(nb_neighbors=K_NEIGHBORS,std_ratio=STD_RATIO)  
#         filteredPoints=np.asarray(cl.points)
#         # print(pointCloud[roi,0:3])
#         # print(filteredPoints)
#         inlierIndex = np.argmin(
#             (pointCloud[:,0][:,np.newaxis]-filteredPoints[:,0][np.newaxis,:])**2+
#             (pointCloud[:,1][:,np.newaxis]-filteredPoints[:,1][np.newaxis,:])**2+
#             (pointCloud[:,2][:,np.newaxis]-filteredPoints[:,2][np.newaxis,:])**2,
#             axis=0
#         )
#         # print(inlierIndex)
#         pointCloud[inlierIndex,-1]= i
#         pointCloud = pointCloud[pointCloud[:,-1]!=-1,:]
#         return pointCloud

#     return pointCloud

def divideCluster(pointCloud, VOXEL_SIZE = 0.3, THRESHOLD = 0):
    '''
    ### Notice
    Divide cluster if there are more than two trees
    '''
    # pointcloud shape : (n,12) after here(cluster ID updated)
    # x, y, z, intensity, laserID, pixelu, pixelv, R, G, B, temperature, clusterID
    clusterID = np.unique(pointCloud[:,-1])
    clusterCenterDict = {}
    # set new cluster ID
    newCluster = np.max(clusterID)+1
    
    for i in clusterID:
        # Temporally change working clusterId to -1
        pointCloud[pointCloud[:,-1]==i,-1] = -1
        roi = np.where(pointCloud[:,-1]==-1)[0]
        pc_roi = pointCloud[roi,:3]
        # skip if cluster is empty
        if pc_roi.size == 0:
            continue
        # make grid and add grid Id to the last column
        # x, y, z, intensity, laserID, pixelu, pixelv, R, G, B, temperature, clusterID, gridID
        x= np.floor((pc_roi[:,0]-np.min(pc_roi[:,0]))/VOXEL_SIZE)
        y= np.floor((pc_roi[:,1]-np.min(pc_roi[:,1]))/VOXEL_SIZE)
        gridID = 100000*x + y
        pc_roi = np.hstack((pc_roi,gridID.reshape(-1,1)))
        cellID=np.unique(gridID)
        minGridZ = np.zeros((pc_roi.shape[0],1))
        # Calcalute min z value of grid and add it to the last column
        # x, y, z, intensity, laserID, pixelu, pixelv, R, G, B, temperature, clusterID, gridID, mixZ
        for g in cellID:
            roi2 = np.where(pc_roi[:,-1]==g)[0]
            minGridZ[roi2] = np.min(pc_roi[roi2,2])
        pc_roi = np.hstack((pc_roi,minGridZ))
        
        #make meshgrid (n+2,n+2)
        x_,y_ = np.meshgrid(np.arange(x.max()+1),np.arange(y.max()+1))
        # (n,n)array having 1~n
        x_window = x_[1:-1,1:-1]
        y_window = y_[1:-1,1:-1]
        # make zPlane having same size with meshgrid without first and last row or column.(n,n)
        zPlane = np.full(x_.shape,0).astype(np.float64)
        rows, cols = zPlane.shape
        # skip if zPlane is smaller than 3,3
        if rows < 3 and cols < 3:
            continue

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
                clusterCenterList.append(pc_targetGrid[np.argmin(pc_targetGrid[:,2]),:3])
        
        # if center is not detected, change clusterID to 9999
        if count == 0:
            pointCloud[roi,-1]=9999
        
        # if a single center is detected, use original clusterID again 
        if count ==1 :
            # clusterCenterDict is clusterID: clusterCenterx,y,z
            clusterCenterDict[i]= clusterCenterList[0]
            pointCloud[roi,-1]=i
        
        if count >=2:
            # distanceList is (the number of point in cluster, the number of count)
            # calculate distance between center and all other points
            distanceList = np.zeros((pc_roi.shape[0],count))
            for h in range(len(clusterCenterList)):
                distance = np.sqrt(np.sum((pointCloud[roi,:3]-clusterCenterList[h].reshape(1,-1))**2,axis=1))
                distanceList[:,h] = distance
            # new cluserID is (the number of points in cluster,1) where the value is newClusterID
            newClusterID = np.argmin(distanceList,axis=1)+newCluster
            pointCloud[roi,-1] = newClusterID
            unique_newclusterID = np.unique(newClusterID)
            
            for h in range(len(unique_newclusterID)):
                clusterCenterDict[int(unique_newclusterID[h])]= clusterCenterList[h]
            newCluster = np.max(newClusterID)+1
            pointCloud[roi,-1]=newClusterID
    
    return pointCloud[pointCloud[:,-1]!=9999,:],clusterCenterDict

def getTreeCluster(pointCloud,clusterCenterDict,THRESHOLD = 0.5):
    '''Threshold means the distance from the center to the farthest point of each zCell'''
    clusterID = np.unique(pointCloud[:,-1])

    for i in clusterID:
        pointCloud[pointCloud[:,-1]==i,-1] = -1
        roi = np.where(pointCloud[:,-1]==-1)[0]
        pc_roi = pointCloud[roi,:3]
        zGrid = np.arange(np.min(pc_roi[:,2]),np.max(pc_roi[:,2]),step = 0.3)
        center = clusterCenterDict[int(i)][:2]
        distanceStd = []
        for g in range(len(zGrid)-1):            
            zCell = pc_roi[(pc_roi[:,2]>=zGrid[g]) & (pc_roi[:,2]<zGrid[g+1]), :]
            if (zCell.size ==0) or (zCell.shape[0]==1):
                continue
            distance = np.sqrt(np.sum((zCell[:,:2]-center)**2,axis=1))
            # distanceStd.append(np.std(distance))
            distanceStd.append(np.max(distance))
    
        if (len(distanceStd)>=4) and ((distanceStd[0]<=THRESHOLD) or (distanceStd[1]<=THRESHOLD)) and (np.any(np.array(distanceStd)[2:]>THRESHOLD*2)):
        # if (len(distanceStd)>=2) and ((distanceStd[0]<=0.1) or (distanceStd[1]<=0.1)) and any(distanceStd)>0.1:
            # if distanceStd[0]>0.1:
            #     pointCloud[]
            pointCloud[roi,-1]=i
        else:
            del clusterCenterDict[int(i)]
            pointCloud[roi,-1]=9999    


    return pointCloud[pointCloud[:,-1]!=9999,:], clusterCenterDict




