# street_tree_mls_cam
Street Tree Sensing Using Vehicle-based Mobile Laser Scanning and Camera 

## Contents
+ [launch.py](#launch)
+ [pointCloud_utils.py](#pointcloud_utils)
  + [distanceFilter](#distancefilter)
  + [groundFilter](#groundfilter) `update needed - make it work regardless of input point cloud column numbers`
  + dbscan
  + removeShortCluster
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
point cloud preprocess : `distanceFilter` + `groundFilter`


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
