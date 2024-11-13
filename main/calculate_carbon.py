import os
import numpy as np
import pandas as pd
import yaml

#Jo et al., 2012
def zelkovaAllometry(dbh):
    leafRatio = 0.055
    carbonstock = np.exp(-2.7512) * dbh**2.4952
    return carbonstock*(1-leafRatio)

def ginkgoAllometry(dbh):
    leafRatio = 0.06
    carbonstock = np.exp(-2.8428) * dbh**2.3787
    return carbonstock*(1-leafRatio)

def prunusAllometry(dbh):
    leafRatio = 0.055
    carbonstock = np.exp(-2.8265) * dbh**2.4181
    return carbonstock*(1-leafRatio)

def acerPalAllometry(dbh):
    leafRatio = 0.055
    carbonstock = -23.2064 + 4.8538 * dbh
    return carbonstock*(1-leafRatio)

# Matsue et al., 2009
def platanusAllometry(dbh): 
    carbonstock = 0.5*0.0434 * dbh ** 2.7773
    return carbonstock

#jo et al., 2013
def pinusDensifloraAllometry(dbh):
    leafRatio = 0.126
    carbonstock = np.exp(-3.1140) * dbh**2.4430
    return carbonstock*(1-leafRatio)

# jo et al., 2014
def chionanthusAllometry(dbh):
    leafRatio = 0.075
    carbonstock = np.exp(-2.7512) * dbh**2.4952
    return carbonstock*(1-leafRatio)

def acerAllometry(dbh):
    volume = 0.0000709 * dbh ** 2.511
    mass = volume * 620
    return mass*0.5*1.26

def metasequoiaAllometry(dbh):
    volume = 0.000527 * dbh ** 2.0285
    mass = volume * 284
    return mass*0.5*1.26

def generalAllometry(dbh):
    rootRatio = 0.26
    mass = 0.080 * dbh ** 2.299*(1+rootRatio)
    return mass*0.5

allometry_functions = {
    'Zelkova serrata' : zelkovaAllometry,
    'Ginkgo biloba' : ginkgoAllometry,
    'Prunus yedoensis' : prunusAllometry,
    'Chionanthus  retusus' : chionanthusAllometry,
    'Pinus densiflora' : pinusDensifloraAllometry,
    'Metasequoia glyptostroboides' : metasequoiaAllometry,
    'Styphnolobium japonicum' : generalAllometry,
    'Platanus occidentalis' : platanusAllometry,
    'Acer palmatum' : acerPalAllometry,
    'Quercus palustris' : generalAllometry,
    'Aesculus turbinata' : generalAllometry,
    'Others' : generalAllometry,
    'Acer buergerianum' : acerAllometry,
    'Not Detected' : generalAllometry
}

def applyAllometry(row):
    species = row['species']
    dbh = row['dbh']
    # Select the correct allometry function based on species
    allometry_func = allometry_functions[species]
    # Apply the allometry function
    return allometry_func(dbh)

def calculateCarbon(pointCloud):
    # pointCloudWithWorldCoordinates = pointCloud
    
    # projection = Proj("+proj=utm +zone=52 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    # lon,lat=projection(pointCloudWithWorldCoordinates[:,0],pointCloudWithWorldCoordinates[:,1],inverse=True)
    # treeLocation=np.vstack((lat,lon)).T
    # treeLocation=np.hstack((treeLocation,pointCloudWithWorldCoordinates))
    # treeLocationPandas=pd.DataFrame(treeLocation,columns=['lat','long','utm_x','utm_y','altitude','ID','height','dbh','species'])
    treeLocationPandas=pd.DataFrame(pointCloud,columns=['utm_x','utm_y','altitude','ID','height','groundLevel','height_raw','dbh_raw','dbh','species'])
    treeLocationPandas.replace({'species':0},'Zelkova serrata',inplace=True)
    treeLocationPandas.replace({'species':1},'Ginkgo biloba',inplace=True)
    treeLocationPandas.replace({'species':2},'Prunus yedoensis',inplace=True)
    treeLocationPandas.replace({'species':3},'Chionanthus  retusus',inplace=True)
    treeLocationPandas.replace({'species':4},'Pinus densiflora',inplace=True)
    treeLocationPandas.replace({'species':5},'Metasequoia glyptostroboides',inplace=True)
    treeLocationPandas.replace({'species':6},'Styphnolobium japonicum',inplace=True)
    treeLocationPandas.replace({'species':7},'Platanus occidentalis',inplace=True)
    treeLocationPandas.replace({'species':8},'Acer palmatum',inplace=True)
    treeLocationPandas.replace({'species':9},'Quercus palustris',inplace=True)
    treeLocationPandas.replace({'species':10},'Aesculus turbinata',inplace=True)
    treeLocationPandas.replace({'species':11},'Others',inplace=True)
    treeLocationPandas.replace({'species':12},'Acer buergerianum',inplace=True)
    treeLocationPandas.replace({'species':9999 },'Not Detected',inplace=True)
    treeLocationPandas['carbonStock'] = treeLocationPandas.apply(applyAllometry, axis=1)
    # absPath=os.getcwd()

    return treeLocationPandas

def main_par(idx, folderList):
    print(folderList[idx])
    resultPath = os.path.join(rootPath,folderList[idx],"result")
    inventoryPath = os.path.join(resultPath,f"inventory_refined.txt")
    inventory = np.loadtxt(inventoryPath)
    # fix species nan value
    inventory = inventory[~np.isnan(inventory).any(axis=1)]
    inventory_final = calculateCarbon(inventory)
    # print(inventory_final.shape)
    inventory_final.to_csv(os.path.join(resultPath,f'inventory_carbon.csv'))

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