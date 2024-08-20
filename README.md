# 😊❌❌❌ THIS REPOSITORY is still in its early days 

# street_tree_mls_cam
Street Tree Sensing Using Vehicle-based Mobile Laser Scanning and Camera  

### Dataset structure  
```
{YOUR_DATASET_PATH}
⏬ RAW DATA ⏬
├─ 📂 1_camera  
│  ├─ {one_digit}_{six_digit_raw_frame_number}.jpg  
│  └─ ...    
│ 
├─ 📂 2_lidar  
│  └─ {year}-{month}-{date}-{hour}-{minute}-{second}_VLP_32C.pcap    
│ 
├─ 📂 3_gps  
│  ├─ {unique_code_for_file}.T04  
│  └─ ...  
│ 
├─ 📂 4_lwir  
│  ├─ {one_digit}_{six_digit_raw_frame_number}.bin  
│  └─ ...
│ 
⏬ POSPAC PROCESSED ⏬
├─ 📂 pospac    
│  ├─ pospac default project files  
│  └─ ...
├─ event1.txt 
├─ event1_Mission 1.dat
├─ event2.txt
├─ event2_Mission 2.dat
├─ output.txt
│ 
⏬ PREPROCESSED DATA ⏬
└─ 📂 preprocessed_data
   ├─ 📂 imu
   │  ├─ {ten_digit_matched_frame_number}.txt
   │  └─ ... 
   │ 
   ├─ 📂 pointCloudPackets
   │  ├─ {ten_digit_matched_frame_number}.bin
   │  └─ ... 
   │ 
   ├─ 📂 pointCloudFrame
   │  ├─ {ten_digit_matched_frame_number}.bin
   │  └─ ... 
   │ 
   ├─ 📂 image
   │  ├─ {ten_digit_matched_frame_number}.jpg
   │  └─ ... 
   │ 
   ├─ 📂 image_imu
   │  ├─ {ten_digit_matched_frame_number}.txt
   │  └─ ... 
   │ 
   ├─ 📂 lwir
   │  ├─ {ten_digit_matched_frame_number}.bin
   │  └─ ... 
   │ 
   └─ 📂 lwir_imu
   │  ├─ {ten_digit_matched_frame_number}.txt
   │  └─ ... 
   │
⏬ IMAGE PREPROCESSED DATA ⏬ 
   └─ 📂 image_processed2
      ├─ {ten_digit_matched_frame_number}.jpg
      └─ ... 
```
### Raw data 
Raw data was collected using off-the-shelf Mobile Mapping System (MMS) unit (DL-Replica; Mobiltech, Seoul, Republic of Korea). The sensor suite consisted of a LiDAR sensor (VLP-32C; Velodyne, San Jose, California, USA), an RGB camera (FLIR Blackfly; Teledyne FLIR, Wilsonville, Oregon, USA), a thermal camera (FLIR A65; Teledyne FLIR, Wilsonville, Oregon, USA), and a positioning sensor (APX-15 UAV; Trimble Applanix, Richmond Hill, Ontario, Canada) 

### POSPAC preprocess
we used commercial software POSPac-UAV 8.4 (Trimble Applanix, Richmond Hill, Ontario, Canada) to preprocess position data of MMS.
for detail, contact: tackangYang@gmail.com

### Dataset preprocess
Preprocessing was done for sensor alignment.
Modify config file to choose folders to preprocess.  
```
cd raw_data_converter
python raw_data_converter.py
```

### Image preprocess (Optional)
Image are preprocessed for the better depiction of street trees. We used a matlab code.
```
cd image_preprocess
```

# 😊❌❌❌ below is temp

### map_RGBT column info  
0)x 1)y 2)z 3)intensity 4)laserID 5)R(float0-1) 6)G(float0-1) 7)B(float0-1) 8)Thermal 9)clusterID 10)species 11)x_world 12)y_world 13)z_world
### Result outline  
launch --> launch_tree --> treeRefinement --> rf_featureExtract --> rf_module --> launch_groundDhm --> treeFeatureRetrieval --> tocsv --> calculateCarbon_ver2 --> MergeAllInventory  
+ treeRefinement  
`IN: map_RGBT_{Date}.txt --> OUT: map_RGBT_{Date}_rf.txt, center_RGBT_{Date}_rf.txt`  
+ rf_featureExtract  
`IN: map_RGBT_{Date}_rf.txt --> OUT: map_RGBT_{Date}_rf_Features.txt`  
+ rf_module  
`IN: map_RGBT_{Date}_rf.txt, map_RGBT_{Date}_rf_Features.txt, center_RGBT_{Date}_rf.txt --> OUT: map_RGBT_{Date}_rf_final.txt, center_RGBT_{Date}_rf_final.txt`  
+ launch_groundDhm  
`IN: center_RGBT_{Date}_rf_final.txt --> OUT: center_RGBT_{Date}_with_height.txt`  
+ treeFeatureRetrieval  
`IN: map_RGBT_{Date}.txt, center_RGBT_{Date}_with_height.txt --> OUT: inventory_{Date}_rf_final.txt`  
+ tocsv  
`IN: inventory_{Date}_rf_final.txt --> OUT: {folderName}_inventory_{Date}_rf_final.txt`  
+ calculateCarbon_ver2  
`IN: {folderName}_inventory_{Date}_rf_final.txt --> OUT: {folderName}_inventory_{Date}_rf_final_ver2.csv`  
+ MergeAllInventory  
`IN: {folderName}_inventory_{Date}_rf_final_ver2.csv --> OUT: total_inventory{Date}_ver2.csv`  


## Contents
+ [launch.py](#launch)

+ [launch_tree.py](#launch_tree)

+ [treeRefinement.py](#treerefinement)

+ [pointCloud_utils.py](#pointcloud_utils)
  + [distanceFilter](#distancefilter)
  + [groundFilter](#groundfilter) `update needed - make it work regardless of input point cloud column numbers`
  + [dbscan](#dbscan) `update needed - changing it to open3d dbscan`
  + [removeShortCluster](#removeshortcluster)
  + [divideCluster](#dividecluster)
  + [getTreeCluster](#gettreecluster)
  
+ [lidar_cam_utils.py](#lidar_cam_utils) `update needed - This part need a bit more generalization. e.g. get rotation matrix as a input instead of calculating rotation matrix from extrinsic parameters.`
  + [getCameraRotation](#getcamerarotation)
  + [lidar2cam](#lidar2cam)
  + [getSegInfo](#getseginfo)
  + [getBBinfo](#getbbinfo) 
  
+ [imu_utils.py](#imu_utils)
  + [getTransform](#gettransform)
  + [getTransformedPointCloud](#gettransformedpointcloud)
  + [getWorldPointCloud](#getworldpointcloud)
  
## launch
`issue at launch stage is that it does not work for the dataset that has an empty image(0kb)  - check the image imu path and car imu path`
  + ### point cloud preprocess
    `distanceFilter` + `groundFilter`
  + ### get segmentation information from the image `update needed - RGB information gathering should be added here`
    `getTransfromedPointCloud` Transform point cloud coordinates to the position of a car at the time image is taken.  
    `lidar2cam` + `getSegInfo`
    
  + ### point cloud processing
    `dbscan` clustering to get the objects from the point cloud  
    `removeShortCluster` remove too short clusters  
    `divideCluster` to divide clusters that have multiple trees  
    `getTreeCluster` to get tree clusters from the clusters 
  + ### tree information retrieval
    `getBBinfo` to get the species information from the image 

## launch_tree
`issue : center number > cluster number`
Merge each frame into the whole map. 

## treeRefinement
`Better performances`
Refine tree clusters by dbscan.

## pointCloud_utils
point cloud data is `n x k` form(`n` is the number of points and `k` is the number of point cloud attributes, e.g. intensity). First three columns have to be local(sensor) x,y,z coordinates of the point cloud
+ ### distanceFilter
  Return the point cloud within the `MIN_DISTANCE` and `MAX_DISTANCE`.  
  >*__*Requisite: First two columns have to be x, y coordinates of point cloud.__* 

  ``` python
  def distanceFilter(pointCloud, MIN_DISTANCE = 3.5, MAX_DISTANCE = 80)
  ```
+ ### groundFilter
  Return ground removed point cloud. If `return_ground = True`, ground label(ground = 0, non-ground = 1) is added to the last column. 
  >*__*Requisite: To be written later__(about input data format)* 
  
  ``` python
  def groundFilter(pointCloud, GRID_SIZE=0.5, GROUND_THICKNESS=0.25, THRESHOLD = 0.25, return_ground = False)
  ```
+ ### dbscan

  ``` python
  def dbscan(pointCloud,EPS=1.9,MIN_POINTS=10)
  ```
+ ### removeShortCluster

  ``` python
  def removeShortCluster(pointCloud, CLUSTER_HEIGHT= 2.5)
  ```
+ ### divideCluster
  Points below `THRESHOLD` is used to make the surface
  
  ``` python
  def divideCluster(pointCloud, VOXEL_SIZE = 0.3, THRESHOLD = 0)
  ```
+ ### getTreeCluster
  `THRESHOLD` is the maximum distance between the cluster center and the point in trunk.
  
  ``` python
  def getTreeCluster(pointCloud,clusterCenterDict, THRESHOLD = 0.5)
  ```
  
## lidar_cam_utils
+ ### getCameraRotation
  Return 3 x 3 rotation matrix from extrinsic parameters between LiDAR and camera.  
  `LR` = left-right(Yaw), `UD` = up-down(Pitch), `Rot` = Rotate(Roll)
  
  ``` python
  def getCameraRotation(LR,UD,Rot)
  ```
+ ### lidar2cam
  Return point cloud with pixel coordinates when it is projected on the camera image plane. Pixel coordinates are added to the last two columns. 
  
  ``` python
  def lidar2cam(pointCloud, PIXEL_WIDTH=1920, PIXEL_HEIGHT=1200)
  ```
+ ### getSegInfo
  Return point cloud within the binary segmentation result when it is projected on the binary segmented image. Assume that segmented pixel value is 255.  
  >*__*Requisite: point cloud should have pixel coordinates at the last two columns. Segmented image have to be in a numpy array format.__* 
  
  ``` python
  def getSegInfo(pointCloud, seg)
  ```
+ ### getBBinfo
  Return point cloud within the bounding box when it is projected on bounding box classification result on image. Bounding box is `n x 6` data, `n` is the bounding box number and 6 column means: 0)classID; 1)xMin; 2)xMax; 3)yMin; 4)yMax; 5)confidence score.
  
  ``` python
  def getBBinfo(pointCloud,boundingBox,PIXEL_WIDTH=1920, PIXEL_HEIGHT=1200)
  ```
  
 ## imu_utils
 + ### getTransform
 
  ```python
  def getTransform(imu)
  ```
 + ### getTransformedPointCloud
  
  ```python
  def getTransformedPointCloud(pointCloud,imu,imageImu)
  ```
 + ### getWorldPointCloud

  ```python
  def getWorldPointCloud(pointCloud,imu)
  ```
