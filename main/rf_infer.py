import joblib
import numpy as np
import os
import parmap
import sys
import yaml

def randomForest(newData, loadedClf, loadedScaler):
    # Split the new data into instance IDs, class labels, and features
    newInstanceIds = newData[:, 0]
    newClassLabels = newData[:, 1]
    newFeatures = newData[:, 2:]

    # Normalize the features using the loaded scaler
    normalizedNewFeatures = loadedScaler.transform(newFeatures)

    # Make predictions on the new dataset
    newPredictions = loadedClf.predict(normalizedNewFeatures)

    # Find the indices where the true class labels and predictions match
    correctPredictionIndices = np.where(newClassLabels == newPredictions)

    # Extract the instance IDs of the correct predictions
    correctPredictionIds = newInstanceIds[correctPredictionIndices]

    return correctPredictionIds

def main_par(idx, folderList,loadedClf, loadedScaler):
    resultPath = os.path.join(rootPath,folderList[idx],"result")
    lidarPath = os.path.join(resultPath, 'tree_multiframe_ppfilter.txt')
    centerPath = os.path.join(resultPath, 'center_multiframe_ppfilter.txt')

    basename = os.path.splitext(os.path.basename(lidarPath))[0]
    centerbasename = os.path.splitext(os.path.basename(centerPath))[0]
    rfDataPath = os.path.join(resultPath,f'{basename}_Features.txt')
    
    rfData = np.loadtxt(rfDataPath)
    pointCloud = np.loadtxt(lidarPath)
    center = np.loadtxt(centerPath)
    
    correctPredictionIds = list(randomForest(rfData, loadedClf, loadedScaler))
    clusterID = np.unique(pointCloud[:,-5])

    for ID in clusterID:
        if not ID in correctPredictionIds:
            pointCloud = np.delete(pointCloud,np.where(pointCloud[:,-5]==ID)[0],axis=0)
            center = np.delete(center,np.where(center[:,-1]==ID)[0],axis=0)
    
    np.savetxt(os.path.join(resultPath,f"{basename}_rf.txt"), pointCloud)
    np.savetxt(os.path.join(resultPath,f"{centerbasename}_rf.txt"), center)

    print(folderList[idx], 'is done')

if __name__ == "__main__" :
    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for random forest application")
    print(folderList)
        
   
    # Load the saved model and scaler
    loadedClf = joblib.load('../ckpts/random_forest_classifier_model.pkl')
    loadedScaler = joblib.load('../ckpts/min_max_scaler.pkl')
    
    
    NUM_WORKERS = 1
    parmap.map(
        main_par,
        range(len(folderList)),folderList, loadedClf, loadedScaler,
        pm_pbar=True, pm_processes = NUM_WORKERS,
        )