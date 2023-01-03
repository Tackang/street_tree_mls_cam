# street_tree_mls_cam
Street Tree Sensing Using Vehicle-based Mobile Laser Scanning and Camera 

## Contents
+ [pointCloud_utils](##pointcloud_utils)
  + [distanceFilter](####distancefilter)
  + groundFilter
  + dbscan
  + removeShortCluster
+ lidar_cam_utils
+ lidar_lwircam_utils
+ imu_utils

## pointCloud_utils
point cloud data is ```n x k``` form(```n``` is the number of points and ```k``` is the number of attributes). First three columns have to be local(sensor) x,y,z coordinates of the point cloud
+ #### distanceFilter
  Return the point cloud within the ```MIN_DISTANCE``` and ```MAX_DISTANCE```. 
First two columns have to be x, y coordinates of point cloud. 

  ``` python
  def distanceFilter(pointCloud, MIN_DISTANCE = 3.5, MAX_DISTANCE = 80)
  ```
+ #### groundFilter
  Return ground removed point cloud. If ```return_ground = True```, ground label(ground = 0, non-ground = 1) is added to the last column. 
  
  ``` python
  def groundFilter(pointCloud, GRID_SIZE=0.5, GROUND_THICKNESS=0.25, THRESHOLD = 0.25, return_ground = False)
  ```
