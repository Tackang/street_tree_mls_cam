import numpy as np
import os
import yaml
from utils.packet_destroyer import startt,start_velo16
import utils.packet2frame as packet2frame
from utils.dataset import totalDataset
from utils.pickImages import finish
import utils.lwirCalibration as lwirCalibration
import utils.imageCalibration as imageCalibration

'''함수 선언'''
# save tkysImu and tkysPointCloudPacket
def pcapDestroy(tkysDir, output, pcap, event1):
    startt(tkysDir,output,pcap,event1)

# save tkysImu and tkysPointCloudPacket
def pcapDestroy_velo16(tkysDir, output, pcap):
    start_velo16(tkysDir,output,pcap)

# save tkysPointCloudFrame
def pointCloudPacket2frame(tkysDir,tkysPointCloudFrameDir):
    packet2frame.main(tkysDir,tkysPointCloudFrameDir)

# save tkysImage and tysLwirImage
def pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,tkysDir,tkysImageDir,tkysLwirImageDir):
    finish(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,tkysDir,tkysImageDir,tkysLwirImageDir)

def checkDataNumber():
    print("IMU data number is:",len(os.listdir(tkysImuDir)))
    print("Point Cloud Frame data number is:",len(os.listdir(tkysPointCloudFrameDir)))
    print("Image data number is:",len(os.listdir(tkysImageDir)))
    print("Lwir Image data number is:",len(os.listdir(tkysLwirImageDir)))
    print("Image Imu data number is:",len(os.listdir(tkysImageImuDir)))
    print("Lwir Image Imu data number is:",len(os.listdir(tkysLwirImageImuDir)))

def checkDataNumber_last():
    print("IMU data number is:",len(os.listdir(tkysImuDir)))
    print("Point Cloud Frame data number is:",len(os.listdir(tkysPointCloudFrameDir)))
    print("Image data number is:",len(os.listdir(tkysImageDir)))
    print("Lwir Image data number is:",len(os.listdir(tkysLwirImageDir)))
    # print("Image Calibrated data number is:",len(os.listdir(tkysImageDir_Calibrated)))
    # print("Lwir Image Calibrated data number is:",len(os.listdir(tkysLwirImageDir_Calibrated)))

def imageAndLwirCalibration(tkysImageDir, tkysImageDir_Calibrated, tkysDir, tkysLwirImageDir_Calibrated):
    imageCalibration.main(tkysImageDir, tkysImageDir_Calibrated)    
    lwirCalibration.main(tkysDir, tkysLwirImageDir_Calibrated)

def main():
    if OPTION == 0:
        pcapDestroy(tkysDir, output, pcap, event1)
        pointCloudPacket2frame(tkysDir,tkysPointCloudFrameDir)
        pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,tkysDir,tkysImageDir,tkysLwirImageDir)
        checkDataNumber()
        # imageAndLwirCalibration(tkysImageDir, tkysImageDir_Calibrated,tkysDir, tkysLwirImageDir_Calibrated)

    if OPTION == 1:
        pointCloudPacket2frame(tkysDir,tkysPointCloudFrameDir)
        pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,tkysDir,tkysImageDir,tkysLwirImageDir)
        checkDataNumber()
        # imageAndLwirCalibration(tkysImageDir, tkysImageDir_Calibrated,tkysDir, tkysLwirImageDir_Calibrated)

    if OPTION == 2:
        pickImages(baseDir,event1Mission,event2Mission,cameraDir,lwirDir,tkysDir,tkysImageDir,tkysLwirImageDir)
        checkDataNumber()
        # imageAndLwirCalibration(tkysImageDir, tkysImageDir_Calibrated,tkysDir, tkysLwirImageDir_Calibrated)

    if OPTION == 3:
        checkDataNumber()
        # imageAndLwirCalibration(tkysImageDir, tkysImageDir_Calibrated,tkysDir, tkysLwirImageDir_Calibrated)

    # if OPTION == 4:
    #     # imageAndLwirCalibration(tkysImageDir, tkysImageDir_Calibrated,tkysDir, tkysLwirImageDir_Calibrated)
    #     checkDataNumber_last()

    if OPTION ==4:
        pcapDestroy_velo16(tkysDir, output, pcap)
        pointCloudPacket2frame(tkysDir,tkysPointCloudFrameDir)
        
if __name__ == "__main__":
    '''SELECT OPTION'''
    OPTION = int(input("***********SELECLT OPTION***********\n\
    OPTION 0: start from pcapDestroy\n\
    OPTION 1: start from pointCloudPacket2frame\n\
    OPTION 2: start from pickImages\n\
    OPTION 3: only check the data numbers\n\
    OPTION 4: for velo-16\n"))

    while not (OPTION==0 or OPTION == 1 or OPTION == 2 or OPTION == 3 or OPTION == 4):
        print("Wrong Option")
        OPTION = int(input("***********SELECLT OPTION***********\n\
        OPTION 0: start from pcapDestroy\n\
        OPTION 1: start from pointCloudPacket2frame\n\
        OPTION 2: start from pickImages\n\
        OPTION 3: only check the data numbers\n\
        OPTION 4: for velo-16\n"))

    removeOption = int(input("***********SELECLT REMOVE OPTION***********\n\
    REMOVE OPTION 0: DO NOT REMOVE EXISTING FILE\n\
    REMOVE OPTION 1: REMOVE EXISTING FILE\n"))
    while not (removeOption==0 or removeOption == 1):
        print("Wrong Option")
        removeOption = int(input("***********SELECLT REMOVE OPTION***********\n\
        REMOVE OPTION 0: DO NOT REMOVE EXISTING FILE\n\
        REMOVE OPTION 1: REMOVE EXISTING FILE\n"))

        
    # Load the folderList from the YAML file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    folderList = config['folderList']
    rootPath = config['rootPath']
        
    for idx in range(len(folderList)):
        BASEDIR = os.path.join(rootPath, folderList[idx])
        print(folderList[idx])
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
            tkysDir = os.path.join(baseDir,"preprocessed_data")
            if not os.path.isdir(tkysDir) :
                os.mkdir(tkysDir)

            tkysImuDir = os.path.join(tkysDir,"imu")
            if not os.path.isdir(tkysImuDir) :
                os.mkdir(tkysImuDir)
            
            tkysPointCloudPacketsDir = os.path.join(tkysDir,"pointCloudPackets")
            if not os.path.isdir(tkysPointCloudPacketsDir) :
                os.mkdir(tkysPointCloudPacketsDir)
            
            tkysPointCloudFrameDir = os.path.join(tkysDir,"pointCloudFrame")
            if not os.path.isdir(tkysPointCloudFrameDir) :
                os.mkdir(tkysPointCloudFrameDir)
            
            tkysImageDir = os.path.join(tkysDir,"image")
            if not os.path.isdir(tkysImageDir) :
                os.mkdir(tkysImageDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(tkysImageDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {tkysImageDir}')
            # 이미 경로에 파일이 있으면 다 지움. 경로안에 다른 하위 디렉토리 있으면 실패
            tkysImageImuDir = os.path.join(tkysDir,"image_imu")
            if not os.path.isdir(tkysImageImuDir) :
                os.mkdir(tkysImageImuDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(tkysImageImuDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {tkysImageImuDir}')

            tkysLwirImageImuDir = os.path.join(tkysDir,"lwir_imu")
            if not os.path.isdir(tkysLwirImageImuDir) :
                os.mkdir(tkysLwirImageImuDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(tkysLwirImageImuDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {tkysLwirImageImuDir}')
            
            tkysLwirImageDir = os.path.join(tkysDir,"lwir")
            if not os.path.isdir(tkysLwirImageDir) :
                os.mkdir(tkysLwirImageDir)
            else:
                if removeOption == 1:
                    for trash_file in os.scandir(tkysLwirImageDir):
                        try:
                            os.remove(trash_file.path)
                        except:
                            print(f'failed to clean directory: {tkysLwirImageDir}')
            # 이미 경로에 파일이 있으면 다 지움. 경로안에 다른 하위 디렉토리 있으면 실패
            
            # tkysImageDir_Calibrated = os.path.join(tkysDir,"image_calibrated")
            # if not os.path.isdir(tkysImageDir_Calibrated) :
            #     os.mkdir(tkysImageDir_Calibrated)
            
            # tkysLwirImageDir_Calibrated = os.path.join(tkysDir,"lwir_calibrated")
            # if not os.path.isdir(tkysLwirImageDir_Calibrated) :
            #     os.mkdir(tkysLwirImageDir_Calibrated)

        if OPTION == 4:     
            pcapDir = os.path.join(baseDir,"2_lidar")
            pcapFileName = os.listdir(pcapDir)[0]
            pcap = os.path.join(pcapDir,pcapFileName)
            
            output = os.path.join(baseDir,"output_withAccel.txt") 

            '''OUTPUTS DIRECTORIES'''
            tkysDir = os.path.join(baseDir,"tkys_data")
            if not os.path.isdir(tkysDir) :
                os.mkdir(tkysDir)

            tkysImuDir = os.path.join(tkysDir,"imu")
            if not os.path.isdir(tkysImuDir) :
                os.mkdir(tkysImuDir)
            
            tkysPointCloudPacketsDir = os.path.join(tkysDir,"pointCloudPackets")
            if not os.path.isdir(tkysPointCloudPacketsDir) :
                os.mkdir(tkysPointCloudPacketsDir)
            
            tkysPointCloudFrameDir = os.path.join(tkysDir,"pointCloudFrame")
            if not os.path.isdir(tkysPointCloudFrameDir) :
                os.mkdir(tkysPointCloudFrameDir)
        

        '''Functions'''
        main()

    
    

    
    

