import numpy as np
import circle_fit                       # pip install circle-fit
import parmap
import os
from tqdm import tqdm
from scipy.stats import mode
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
import yaml

def heightCal(center):
    height = center[:,-2]-center[:,-1]
    center_output = np.hstack((center[:,:4],height.reshape(-1,1)))
    center_output = np.hstack((center_output,center[:,-1].reshape(-1,1)))
    return center_output

def heightDbhAllometry(height, a=  0.9173, b=  1.0608):
    dbh_log = a+b*np.log(height)
    dbh = np.exp(dbh_log)
    return dbh
def HallometryError_upper(H):
    return (10.6259 + 4.3476*np.log(H))/100
def HallometryError_lower(H):
    return (10.8502 + 2.7928*np.log(H))/100

def hcal_raw(pc_roi):
    return pc_roi[:,-1].max() - pc_roi[:,-1].min()

def dbhCal_noAllometry(pc_roi,center_roi):
    heightList = np.arange(1.25,1.45,0.02)
    breastHeight = pc_roi[:,-1].min()+heightList
    pointCloud_section = []
    c_fit_result = []

    for i in range(len(breastHeight)):
        cellIndex= np.where(
                (pc_roi[:,-1]>= breastHeight[i]-0.01)&(pc_roi[:,-1]<breastHeight[i]+0.01)
                )[0]
        pc_roi_cell = pc_roi[cellIndex,:]
        point_coordinates = pc_roi_cell[:,(-3,-2)]
        pointCloud_section.append(point_coordinates)
        try:
            xc, yc, r, sigma = circle_fit.riemannSWFLa(point_coordinates)
            
            c_fit_result.append(round(r*2*100,2))
        except:
            continue
    dbh = np.mean(np.array(c_fit_result))
    
    return dbh

def dbhCal(pc_roi,center_roi):
    height = center_roi[:,-6]
    error_upper = HallometryError_upper(height)
    error_lower = HallometryError_lower(height)
    dbh_estimate = heightDbhAllometry(height)
    dbh_lower = dbh_estimate*(1-error_lower)
    dbh_upper = dbh_estimate*(1+error_upper)

    heightList = np.arange(1.25,1.45,0.02)
    breastHeight = center_roi[:,-5]+heightList
    pointCloud_section = []
    c_fit_result = []
    for i in range(len(breastHeight)):
        cellIndex= np.where(
                (pc_roi[:,-1]>= breastHeight[i]-0.01)&(pc_roi[:,-1]<breastHeight[i]+0.01)
                )[0]
        pc_roi_cell = pc_roi[cellIndex,:]
        point_coordinates = pc_roi_cell[:,(-3,-2)]
        pointCloud_section.append(point_coordinates)
        try:
            xc, yc, r, sigma = circle_fit.riemannSWFLa(point_coordinates)
            
            c_fit_result.append(round(r*2*100,2))
        except:
            continue
    dbh_inlier = list(filter( lambda x: np.log(x) <= np.log(dbh_upper) and np.log(x) >= np.log(dbh_lower), c_fit_result))
    if dbh_inlier == []:
        dbh = dbh_estimate
    else:
        dbh = np.mean(np.array(dbh_inlier))
    
    return dbh

def getSpecies(pc_roi):
    modeVal = mode(pc_roi[:,-4],keepdims=True)[0]
    if modeVal == 9999:
        try:
            modeVal = mode(pc_roi[pc_roi[:,-4]!=9999,-4],keepdims=True)[0]
        except:
            modeVal = 9999
    return modeVal


def infoRetrieval(pointCloud,center):
    center = heightCal(center)
    #center x,y,z,ID,Height,groundlevel
    clusterID = np.unique(center[:,3])
    dummy = np.full((center.shape[0],4),-9999)
    center = np.hstack((center,dummy))
    #center x,y,z,ID,Height,groundlevel, height_raw, dbh_raw,dbh,species 

    for ID in tqdm(clusterID):
        roi = np.where(pointCloud[:,-5]==ID)[0]
        roi2 = np.where(center[:,3]==ID)[0]
        
        pc_roi = pointCloud[roi,:]
        center_roi = center[roi2,:]
        # height
        center[roi2,-4] = hcal_raw(pc_roi)
        # dbh
        center[roi2,-3] = dbhCal_noAllometry(pc_roi,center_roi)
        center[roi2,-2] = dbhCal(pc_roi,center_roi)
        # species
        center[roi2,-1] = getSpecies(pc_roi)

    return center

def removeInvalidData(inventory, groups):
    removeIDList = []
    for group in groups:
        inventory_roi = inventory[group,:]
        minZvalue = inventory_roi[:,2].min()

        inventory_roi2 = inventory_roi[inventory_roi[:,2]<(minZvalue+1),:]
        meanHeight = inventory_roi2[:,-6].mean()
        meanGroundLevel = inventory_roi2[:,-5].mean()
        meanH_raw = inventory_roi2[:,-4].mean()
        meanDbh_raw = inventory_roi2[:,-3].mean()
        meanDbh = inventory_roi2[:,-2].mean()
        speciesArr = inventory_roi[inventory_roi[:,-1]!=9999,-1]
        if speciesArr.size == 0:
            species = 9999  
        else:
            species = mode(speciesArr,keepdims=True)[0][0]

        
        # print(inventory_roi)
        validID = inventory_roi[np.argmin(inventory_roi[:,2]),3]
        # print(validID)
        # print(inventory_roi[inventory_roi[:,3]!=validID,3])
        removeIDList.append(inventory_roi[inventory_roi[:,3]!=validID,3])
        inventory[inventory[:,3]==validID,-6] = meanHeight
        inventory[inventory[:,3]==validID,-5] = meanGroundLevel
        inventory[inventory[:,3]==validID,-4] = meanH_raw
        inventory[inventory[:,3]==validID,-3] = meanDbh_raw
        inventory[inventory[:,3]==validID,-2] = meanDbh
        inventory[inventory[:,3]==validID,-1] = species
        # break
    # print(np.array(removeIDList).reshape(-1))
    # inventory_final = np.delete(inventory,np.concatenate(removeIDList),axis=0)
    # print(np.concatenate(removeIDList))
    mask = np.isin(inventory[:,3],np.concatenate(removeIDList),invert=True) 
    # print(inventory.shape)
    # print(mask.shape)
    inventory_final = inventory[mask]
    # print(inventory_final.shape)
    return inventory_final
   
def getNeighborGroups(inventory, THRESHOLD = 3):
    inventory_X = inventory[:,0]
    inventory_Y = inventory[:,1]
    distance_X = inventory_X[:,np.newaxis]-inventory_X[np.newaxis,:]
    distance_Y = inventory_Y[:,np.newaxis]-inventory_Y[np.newaxis,:]
    distance = np.sqrt(distance_X**2+distance_Y**2)
    
    idx = np.where(distance<=THRESHOLD)
    pair = np.hstack((idx[0][:,np.newaxis],idx[1][:,np.newaxis]))
    # Sort each pair
    sorted_arr = np.sort(pair, axis=1)

    # Remove duplicates
    unique_pairs = np.unique(sorted_arr, axis=0)

    # Remove pairs where both elements are the same
    unique_pairs = unique_pairs[unique_pairs[:, 0] != unique_pairs[:, 1]]

    # Create a graph using a dictionary
    graph = defaultdict(list)

    for x, y in unique_pairs:
        graph[x].append(y)
        graph[y].append(x)

    # Now we perform a depth-first search (DFS) on the graph
    def dfs(node, group):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                group.append(neighbor)
                dfs(neighbor, group)

    visited = set()
    groups = []

    for node in graph:
        if node not in visited:
            group = [node]
            dfs(node, group)
            groups.append(group)
    return groups


def main_par(idx, folderList):
    print(folderList[idx])
    resultPath = os.path.join(rootPath,folderList[idx],"result")
    lidarPath = os.path.join(resultPath,"tree_multiframe_ppfilter_rf.txt")
    centerPath = os.path.join(resultPath,"center_multiframe_ppfilter_rf_hr.txt")
    # centerPath = os.path.join(mapDir,"center_RGBT_rf_refine- Cloud.txt")

    pointCloud = np.loadtxt(lidarPath)
    center = np.loadtxt(centerPath)

    inventory = infoRetrieval(pointCloud,center)
    np.savetxt(os.path.join(resultPath,"inventory.txt"),inventory)
    # np.savetxt(os.path.join(mapDir,"inventory_refine.txt"),inventory)


def main_par2(idx, folderList):
    print(folderList[idx])

    resultPath = os.path.join(rootPath,folderList[idx],"result")
    inventoryPath = os.path.join(resultPath,"inventory.txt")
    inventory = np.loadtxt(inventoryPath)
    # fix species nan value
    inventory[:,-1] = np.nan_to_num(inventory[:,-1], nan = 9999)
    groups = getNeighborGroups(inventory)
    inventory_final = removeInvalidData(inventory, groups)
    # print(inventory_final.shape)
    np.savetxt(os.path.join(resultPath,'inventory_refined.txt'),inventory_final)
    # np.savetxt(os.path.join('/bess25/didxorkd/04resultCarsense/treeInventory',f'{folderList[idx]}_inventory{dataKey}.txt'),inventory_final)
    # inventory_csv = utmToWgs(inventory_final)
    # print(inventory_csv.shape)
    # inventory_csv.to_csv(os.path.join(mapDir,f'{folderList[idx]}_inventory{dataKey}.csv'),sep=',')
    # inventory_csv.to_csv(os.path.join('/bess25/didxorkd/04resultCarsense/treeInventory','inventory{}_{}.csv'.format(dataKey,folderList[idx])),sep=',')


if __name__ =="__main__":
    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for feature extract")
    print(folderList)
        

    for idx in range(len(folderList)):
        main_par(idx,folderList)   
        main_par2(idx,folderList)