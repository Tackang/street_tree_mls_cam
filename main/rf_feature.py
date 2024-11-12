import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import parmap
import yaml
import sys 

def computeHeight(pointCloud):
    return pointCloud[:,2].max()-pointCloud[:,2].min()

def computeHeightStd(pointCloud):
    return np.std(pointCloud[:,2])

def computeHeightMean(pointCloud):
    return np.mean(pointCloud[:,2])

def computeHeightMedian(pointCloud):
    return np.median(pointCloud[:,2])

def computeCovarianceFeatures(pointCloud):
    centroid = np.mean(pointCloud, axis=0)
    centeredPointCloud = pointCloud - centroid
    covarianceMatrix = np.cov(centeredPointCloud.T)
    return covarianceMatrix

def computeEigenvaluesAndEigenvectors(covarianceMatrix):
    eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
    sortedIndices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[sortedIndices]
    eigenvectors = eigenvectors[:, sortedIndices]
    return eigenvalues, eigenvectors

def computeGeometricFeatures(eigenvalues):
    lambda1, lambda2, lambda3 = eigenvalues
    sum_eigenvalues = np.sum(eigenvalues)

    # Linearity
    linearity = (lambda1 - lambda2) / sum_eigenvalues

    # Planarity
    planarity = (lambda2 - lambda3) / sum_eigenvalues

    # Sphericity
    sphericity = lambda3 / sum_eigenvalues

    return linearity, planarity, sphericity

def angleBetweenVectors(vector1, vector2):
    dotProduct = np.dot(vector1, vector2)
    magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle = np.arccos(dotProduct / magnitudes)
    return angle * 180 / np.pi  # Convert to degrees

def getFeatures(pointCloud):

    pointCloudNumber = pointCloud.shape[0]
    pointCloudCoords = pointCloud[:,-3:]
    pointCloudIntensity = pointCloud[:,3]
    pointCloudRgb = pointCloud[:,5:8]
    pointCloudThermal = pointCloud[:,8]

    height = computeHeight(pointCloudCoords)
    heightStd = computeHeightStd(pointCloudCoords)
    heightMean = computeHeightMean(pointCloudCoords)
    heightMedian = computeHeightMedian(pointCloudCoords)

    covarianceMatrix = computeCovarianceFeatures(pointCloudCoords)
    eigenvalues, eigenvectors = computeEigenvaluesAndEigenvectors(covarianceMatrix) 
    linearity, planarity, sphericity = computeGeometricFeatures(eigenvalues)

    intensityStd = np.std(pointCloudIntensity)
    intensityMean = np.mean(pointCloudIntensity)
    intensityMedian = np.median(pointCloudIntensity)

    rStd = np.std(pointCloudRgb[:,0])
    rMean = np.mean(pointCloudRgb[:,0])
    rMedian = np.median(pointCloudRgb[:,0])

    gStd = np.std(pointCloudRgb[:,1])
    gMean = np.mean(pointCloudRgb[:,1])
    gMedian = np.median(pointCloudRgb[:,1])

    bStd = np.std(pointCloudRgb[:,2])
    bMean = np.mean(pointCloudRgb[:,2])
    bMedian = np.median(pointCloudRgb[:,2])
    
    thermalStd = np.std(pointCloudThermal)
    thermalMean = np.mean(pointCloudThermal)
    thermalMedian = np.median(pointCloudThermal)
    
    pc1 = eigenvectors[:, 0]
    zAxis = np.array([0, 0, 1])
    angle = angleBetweenVectors(pc1, zAxis)

    return [pointCloudNumber, 
            height, 
            heightStd, 
            heightMean, heightMedian,
            linearity, 
            planarity, 
            sphericity, 
            angle, 
            intensityStd, 
            intensityMean, 
            intensityMedian, 
            rStd, 
            rMean, 
            rMedian,  
            gStd, 
            gMean, 
            gMedian,
            bStd,
            bMean, 
            bMedian,
            thermalStd, thermalMean, thermalMedian
            ]

def main(idx, folderList):
    resultPath = os.path.join(rootPath,folderList[idx],"result")
    inputPath = os.path.join(resultPath,'tree_multiframe_ppfilter.txt')
    basename = os.path.splitext(os.path.basename(inputPath))[0]
    outputPath = os.path.join(resultPath,f'{basename}_Features.txt')
    
    inputData = np.loadtxt(inputPath)
    clusterID = np.unique(inputData[:,-5])
    annotation = np.full((clusterID.shape[0],1),3)
    outputDataList = []
    
    for ID in tqdm(clusterID):
        pointCloud = inputData[inputData[:,-5]==ID,:]
        if pointCloud.shape[0]<= 5:
            continue
        features = getFeatures(pointCloud)
        features.insert(0,ID)
        if ID in annotation:
            features.insert(1,0)
        else :
            features.insert(1,1)
        features = np.array(features)[np.newaxis,:]
        outputDataList.append(features)

    outputData = np.concatenate(outputDataList, axis= 0)
    np.savetxt(outputPath,outputData)
    print(folderList[idx], "is done")

if __name__ == '__main__':
    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for feature extract")
    print(folderList)
        

    NUM_WORKERS = 1
    parmap.map(
        main,
        range(len(folderList)),folderList, 
        pm_pbar=True, pm_processes = NUM_WORKERS,
        )
    
    
    