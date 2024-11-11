import numpy as np
import os
import yaml
from utils.packet_destroyer import startt,start_velo16
import utils.packet2frame as packet2frame
from utils.pickImages import finish
import utils.lwirCalibration as lwirCalibration
import utils.imageCalibration as imageCalibration

'''함수 선언'''
# save preprocessedImu and preprocessedPointCloudPacket
def pcapDestroy(preprocessedDir, output, pcap, event1):
    startt(preprocessedDir,output,pcap,event1)

# save preprocessedImu and preprocessedPointCloudPacket
def pcapDestroy_velo16(preprocessedDir, output, pcap):
    start_velo16(preprocessedDir,output,pcap)

# save preprocessedPointCloudFrame
def pointCloudPacket2frame(preprocessedDir,preprocessedPointCloudFrameDir):
    packet2frame.main(preprocessedDir,preprocessedPointCloudFrameDir)

# save preprocessedImage and tysLwirImage
def pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,preprocessedDir,preprocessedImageDir,preprocessedLwirImageDir):
    finish(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,preprocessedDir,preprocessedImageDir,preprocessedLwirImageDir)

def checkDataNumber():
    print("IMU data number is:",len(os.listdir(preprocessedImuDir)))
    print("Point Cloud Frame data number is:",len(os.listdir(preprocessedPointCloudFrameDir)))
    print("Image data number is:",len(os.listdir(preprocessedImageDir)))
    print("Lwir Image data number is:",len(os.listdir(preprocessedLwirImageDir)))
    print("Image Imu data number is:",len(os.listdir(preprocessedImageImuDir)))
    print("Lwir Image Imu data number is:",len(os.listdir(preprocessedLwirImageImuDir)))

def checkDataNumber_last():
    print("IMU data number is:",len(os.listdir(preprocessedImuDir)))
    print("Point Cloud Frame data number is:",len(os.listdir(preprocessedPointCloudFrameDir)))
    print("Image data number is:",len(os.listdir(preprocessedImageDir)))
    print("Lwir Image data number is:",len(os.listdir(preprocessedLwirImageDir)))

def imageAndLwirCalibration(preprocessedImageDir, preprocessedImageDir_Calibrated, preprocessedDir, preprocessedLwirImageDir_Calibrated):
    imageCalibration.main(preprocessedImageDir, preprocessedImageDir_Calibrated)    
    lwirCalibration.main(preprocessedDir, preprocessedLwirImageDir_Calibrated)

def main():
    if OPTION == 0:
        pcapDestroy(preprocessedDir, output, pcap, event1)
        pointCloudPacket2frame(preprocessedDir,preprocessedPointCloudFrameDir)
        pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,preprocessedDir,preprocessedImageDir,preprocessedLwirImageDir)
        checkDataNumber()

    if OPTION == 1:
        pointCloudPacket2frame(preprocessedDir,preprocessedPointCloudFrameDir)
        pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,preprocessedDir,preprocessedImageDir,preprocessedLwirImageDir)
        checkDataNumber()

    if OPTION == 2:
        pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,preprocessedDir,preprocessedImageDir,preprocessedLwirImageDir)
        checkDataNumber()

    if OPTION == 3:
        checkDataNumber()
        
if __name__ == "__main__":
    '''SELECT OPTION'''
    OPTION = int(input("***********SELECLT OPTION***********\n\
    OPTION 0: start from pcapDestroy\n\
    OPTION 1: start from pointCloudPacket2frame\n\
    OPTION 2: start from pickImages\n\
    OPTION 3: only check the data numbers\n"))

    while not (OPTION==0 or OPTION == 1 or OPTION == 2 or OPTION == 3):
        print("Wrong Option")
        OPTION = int(input("***********SELECLT OPTION***********\n\
        OPTION 0: start from pcapDestroy\n\
        OPTION 1: start from pointCloudPacket2frame\n\
        OPTION 2: start from pickImages\n\
        OPTION 3: only check the data numbers\n"))

    removeOption = int(input("***********SELECLT REMOVE OPTION (SELECT 1 WHEN REPROCESSING)***********\n\
    REMOVE OPTION 0: DO NOT REMOVE EXISTING FILE\n\
    REMOVE OPTION 1: REMOVE EXISTING FILE\n"))
    while not (removeOption==0 or removeOption == 1):
        print("Wrong Option")
        removeOption = int(input("***********SELECLT REMOVE OPTION (SELECT 1 WHEN REPROCESSING)***********\n\
        REMOVE OPTION 0: DO NOT REMOVE EXISTING FILE\n\
        REMOVE OPTION 1: REMOVE EXISTING FILE\n"))


    # Load the folderList from the YAML file
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
    print("Folder list for preprocessing")
    print(folderList)
        
    for idx in range(len(folderList)):
        BASEDIR = os.path.join(rootPath, folderList[idx])
        print(f"preprocessing {folderList[idx]}")
        
        '''INPUTS'''
        baseDir = BASEDIR
        print(baseDir)
        if OPTION != 4:
            cameraDir = os.path.join(baseDir,"1_camera") 
            
            pcapDir = os.path.join(baseDir,"2_lidar")
            pcapFileName = os.listdir(pcapDir)[0]
            pcap = os.path.join(pcapDir,pcapFileName)
            
            lwirDir = os.path.join(baseDir,"4_lwir") 
            
            event1 = os.path.join(baseDir,"event1.txt")
            event1Mission = os.path.join(baseDir,"event1_Mission 1.dat")
            
            event2 = os.path.join(baseDir,"event2.txt")
            event2Mission = os.path.join(baseDir,"event2_Mission 1.dat")
            
            output = os.path.join(baseDir,"output.txt") 

            '''OUTPUTS DIRECTORIES'''
            preprocessedDir = os.path.join(baseDir,"preprocessed_data")
            if not os.path.isdir(preprocessedDir) :
                os.mkdir(preprocessedDir)

            preprocessedImuDir = os.path.join(preprocessedDir,"imu")
            if not os.path.isdir(preprocessedImuDir) :
                os.mkdir(preprocessedImuDir)
            
            preprocessedPointCloudPacketsDir = os.path.join(preprocessedDir,"pointCloudPackets")
            if not os.path.isdir(preprocessedPointCloudPacketsDir) :
                os.mkdir(preprocessedPointCloudPacketsDir)
            
            preprocessedPointCloudFrameDir = os.path.join(preprocessedDir,"pointCloudFrame")
            if not os.path.isdir(preprocessedPointCloudFrameDir) :
                os.mkdir(preprocessedPointCloudFrameDir)
            
            preprocessedImageDir = os.path.join(preprocessedDir,"image")
            if not os.path.isdir(preprocessedImageDir) :
                os.mkdir(preprocessedImageDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(preprocessedImageDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {preprocessedImageDir}')

            preprocessedImageImuDir = os.path.join(preprocessedDir,"image_imu")
            if not os.path.isdir(preprocessedImageImuDir) :
                os.mkdir(preprocessedImageImuDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(preprocessedImageImuDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {preprocessedImageImuDir}')

            preprocessedLwirImageImuDir = os.path.join(preprocessedDir,"lwir_imu")
            if not os.path.isdir(preprocessedLwirImageImuDir) :
                os.mkdir(preprocessedLwirImageImuDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(preprocessedLwirImageImuDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {preprocessedLwirImageImuDir}')
            
            preprocessedLwirImageDir = os.path.join(preprocessedDir,"lwir")
            if not os.path.isdir(preprocessedLwirImageDir) :
                os.mkdir(preprocessedLwirImageDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(preprocessedLwirImageDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {preprocessedLwirImageDir}')

        if OPTION == 4:     
            pcapDir = os.path.join(baseDir,"2_lidar")
            pcapFileName = os.listdir(pcapDir)[0]
            pcap = os.path.join(pcapDir,pcapFileName)
            
            output = os.path.join(baseDir,"output_withAccel.txt") 

            '''OUTPUTS DIRECTORIES'''
            preprocessedDir = os.path.join(baseDir,"preprocessed_data")
            if not os.path.isdir(preprocessedDir) :
                os.mkdir(preprocessedDir)

            preprocessedImuDir = os.path.join(preprocessedDir,"imu")
            if not os.path.isdir(preprocessedImuDir) :
                os.mkdir(preprocessedImuDir)
            
            preprocessedPointCloudPacketsDir = os.path.join(preprocessedDir,"pointCloudPackets")
            if not os.path.isdir(preprocessedPointCloudPacketsDir) :
                os.mkdir(preprocessedPointCloudPacketsDir)
            
            preprocessedPointCloudFrameDir = os.path.join(preprocessedDir,"pointCloudFrame")
            if not os.path.isdir(preprocessedPointCloudFrameDir) :
                os.mkdir(preprocessedPointCloudFrameDir)
        

        '''Functions'''
        main()

    
    

    
    

