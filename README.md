# street_tree_mls_cam
Street Tree Sensing Using Vehicle-based Mobile Laser Scanning and Camera 

## Contents
+ [pointCloud_utils.py](#pointcloud_utils)
  + [distanceFilter](#distancefilter)
  + [groundFilter](#groundfilter)
  + dbscan
  + removeShortCluster
+ [lidar_cam_utils.py](#lidar_cam_utils)
+ lidar_lwircam_utils
+ imu_utils

## pointCloud_utils
point cloud data is `n x k` form(`n` is the number of points and `k` is the number of point cloud attributes, e.g. intensity). First three columns have to be local(sensor) x,y,z coordinates of the point cloud
+ ### distanceFilter
  Return the point cloud within the `MIN_DISTANCE` and `MAX_DISTANCE`.  
  *__Requisite: First two columns have to be x, y coordinates of point cloud.__* 

  ``` python
  def distanceFilter(pointCloud, MIN_DISTANCE = 3.5, MAX_DISTANCE = 80)
  ```
+ ### groundFilter
  Return ground removed point cloud. If `return_ground = True`, ground label(ground = 0, non-ground = 1) is added to the last column. 
  *__Requisite: To be written later__* 
  ``` python
  def groundFilter(pointCloud, GRID_SIZE=0.5, GROUND_THICKNESS=0.25, THRESHOLD = 0.25, return_ground = False)
  ```
  
## lidar_cam_utils
+ ### getCameraRotation
  Return 3 x 3 rotation matrix from extrinsic parameters between LiDAR and camera.  
  LR = left-right(yaw), UD = up-down(pitch), Rot = Rotate(roll)
  ``` python
  def getCameraRotation(LR,UD,Rot)
  ```
+ ### lidar2cam
  ``` python
  def lidar2cam(pointCloud, PIXEL_WIDTH=1920, PIXEL_HEIGHT=1200)
  ```
