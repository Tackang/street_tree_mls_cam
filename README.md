# street_tree_mls_cam
Street Tree Sensing Using Vehicle-based Mobile Laser Scanning and Camera 

## Contents
+ pointCloud_utils
  + distanceFilter
  + groundFilter
  + dbscan
  + removeShortCluster
+ lidar_cam_utils
+ lidar_lwircam_utils
+ imu_utils

## pointCloud_utils
+ #### distanceFilter
``` python
def distanceFilter(pointCloud, MIN_DISTANCE=3.5, MAX_DISTANCE=80):
    xyDistance=np.sqrt(pointCloud[:,0]**2+pointCloud[:,1]**2)
    mask=np.where((xyDistance<=MAX_DISTANCE) & (xyDistance>=MIN_DISTANCE))
    return pointCloud[mask]
```
