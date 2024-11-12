from dataset.heightRefine_dataset import hrDataset
import utils.imu_utils as imu_utils
import os
import numpy as np
import parmap
import traceback
import sys
import yaml
import time
import numpy as np
from rtree import index
# import pyproj

def find_error_function(tb, function_names):
    for frame in reversed(tb):
        if frame.name in function_names:
            return frame.name
    return "Unknown"

def convert_seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return int(hours), int(minutes), int(seconds)

def create_spatial_index(points_array):
    idx = index.Index(((i, (x, y, x, y), None) for i, (x, y, z) in enumerate(points_array)))
    return idx

def find_points_within_boundary(idx, center, radius):
    x_min, y_min = center[0] - radius, center[1] - radius
    x_max, y_max = center[0] + radius, center[1] + radius
    return list(idx.intersection((x_min, y_min, x_max, y_max)))

def tree_heights(dhm_points, dem_points, tree_locations, boundary_radius):
    os.chdir('Temp/heightRefine')
    dhm_idx = create_spatial_index(dhm_points)
    dem_idx = create_spatial_index(dem_points)

    heights = []
    for tree_location in tree_locations:
        dhm_points_within_boundary = find_points_within_boundary(dhm_idx, tree_location, boundary_radius)
        dem_points_within_boundary = find_points_within_boundary(dem_idx, tree_location, boundary_radius)
        try:
            max_dhm_z = max(dhm_points[dhm_points_within_boundary, 2])
        except:
            max_dhm_z = 9999
        try:
            min_dem_z = min(dem_points[dem_points_within_boundary, 2])
        except:
            min_dem_z = 9999
            
        
        heights.append([max_dhm_z,min_dem_z])

    os.chdir('../../')
    return heights

def main(index,groundDhmData):

    groundName = groundDhmData[index][0]
    imu = groundDhmData[index][1]
    ground = groundDhmData[index][2]
    dhm = groundDhmData[index][3]
    
    try:
        # pointcloud shape : (n,5) at the beginning
        groundWorld = imu_utils.getWorldPointCloud(ground, imu)
        dhmWorld = imu_utils.getWorldPointCloud(dhm, imu)
        # pointcloud shape : (n,8) from here(world x,y,z updated)
        return groundWorld, dhmWorld
    
    
    except Exception as e:
        error_msg = str(e)
        tb = traceback.extract_tb(sys.exc_info()[2])
        function_names = [
            'getWorldPointCloud',
        ]  # Add other function names if needed
        error_function = find_error_function(tb, function_names)

        with open(logFileName, 'a') as log:
            log.write(f'Error in function: {error_function}, frame number: {groundName}\nError message: {error_msg}\n')
        return tuple()
    
if __name__ == '__main__':
    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for feature extract")
    print(folderList)

    for idx in range(len(folderList)):
        print(folderList[idx])
        dataPath= os.path.join(rootPath,folderList[idx],"preprocessed_data")
        resultPath = os.path.join(rootPath,folderList[idx],"result")
        hrData = hrDataset(resultPath)
        treeLocation = np.loadtxt(os.path.join(resultPath,'center_multiframe_ppfilter_rf.txt'))
        logFileName = os.path.join(resultPath,'launch_heightRefine_log.txt')

    
        # Create DHM and DEM
        start_time = time.time()
        NUM_WORKERS= 50
        results = parmap.map(
            main,
            range(len(hrData)),hrData, pm_pbar=True,
            pm_processes = NUM_WORKERS,
            )
        # Separate the results into groundWorld and dhmWorld lists
        groundWorlds, dhmWorlds = zip(*results)
        # Merge the point clouds
        dem_points = np.vstack(groundWorlds)[:,-3:]
        dhm_points = np.vstack(dhmWorlds)[:,-3:]



        # Street tree locations
        street_trees = treeLocation[:,:2]  # Replace with your street tree coordinates
        boundary_radius = 1.1  # Set the boundary radius

        tree_heights_array = tree_heights(dhm_points, dem_points, street_trees, boundary_radius)
        streetTreesWithHeight= np.hstack((treeLocation,np.array(tree_heights_array)))
        # Save the merged point clouds to files
        # groundWorld_merged_path = os.path.join(tkysDir, outputFolder, "groundMap.txt")
        dhmWorld_merged_path = os.path.join(resultPath,f"center_multiframe_ppfilter_rf_hr.txt")
        # np.savetxt(groundWorld_merged_path, groundWorld_merged)
        np.savetxt(dhmWorld_merged_path, streetTreesWithHeight)

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, minutes, seconds = convert_seconds_to_hms(elapsed_time)
        with open(logFileName, 'a') as log:
            log.write(f'Finished. Processing time: {hours}h {minutes}m {seconds}s\n')
    # log.append(result)
    # print(log)