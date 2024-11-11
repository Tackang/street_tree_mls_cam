import os
from pickle import NONE
import natsort
import numpy as np
import shutil
from tqdm import tqdm
import cv2 as cv

def get_file_mission(numdict_reverse,cameraDir):
    file_mission_dict ={}
    file_name = os.listdir(cameraDir)
    for name in file_name:
        try:
            file_mission_dict[numdict_reverse[int(name.split('.')[0].split('_')[1])+100000*(int(name.split('_')[0])-1)]]=name
        except KeyError:
            with open(os.path.join(cameraDir, '../preprocessed_data/image_log.txt'), 'a') as log:
                log.write(f'KeyError: {name} not in event1_Mission file\n')
        except Exception as e:
            print('Error in get_file_mission/',e)
    return file_mission_dict # {mission file time : file name }   ex)  183634.123412: 1_000123.jpg     /   183636.123123 :0_000231.bin 


def get_file_mission_lwir(numdict_reverse,lwirDir):
    file_mission_dict ={}
    file_name = os.listdir(lwirDir)
    for name in file_name:
        try:
            file_mission_dict[numdict_reverse[int(name.split('.')[0].split('_')[1])+100000*int(name.split('_')[0])]]=name
        except KeyError:
            with open(os.path.join(lwirDir, '../preprocessed_data/lwir_log.txt'), 'a') as log:
                log.write(f'KeyError: {name} not in event2_Mission file\n')
        except Exception as e:
            print('Error in get_file_mission/',e)
    return file_mission_dict # {mission file time : file name }   ex)  183634.123412: 1_000123.jpg     /   183636.123123 :0_000231.bin 



def get_name_dict(file_mission_dict, time_name_dict,event1array,tkysDir):
    mission_time_array = np.array(list(file_mission_dict.keys()))
    imu_name_list = list(time_name_dict.keys())
    namename_dict ={}
    for i in range(len(imu_name_list)):
        imu_time = time_name_dict[imu_name_list[i]] 
        differ_array = abs(mission_time_array-imu_time)
        if np.min(differ_array) <0.05:
            namename_dict[imu_name_list[i]] = file_mission_dict[mission_time_array[np.argmin(differ_array)]]
            event1saver(mission_time_array[np.argmin(differ_array)],imu_name_list[i],event1array,tkysDir)
        else:
            namename_dict[imu_name_list[i]] = NONE

    return namename_dict #lidar_frame_name : camera_file_name

def get_name_dict_lwir(file_mission_dict, time_name_dict,event2array,tkysDir):
    mission_time_array = np.array(list(file_mission_dict.keys()))
    imu_name_list = list(time_name_dict.keys())
    namename_dict ={}
    for i in range(len(imu_name_list)):
        imu_time = time_name_dict[imu_name_list[i]] 
        differ_array = abs(mission_time_array-imu_time)
        if np.min(differ_array) <0.017:
            namename_dict[imu_name_list[i]] = file_mission_dict[mission_time_array[np.argmin(differ_array)]]
            event2saver(mission_time_array[np.argmin(differ_array)],imu_name_list[i],event2array,tkysDir)
        else:
            namename_dict[imu_name_list[i]] = NONE

    return namename_dict #lidar_frame_name : camera_file_name


def copying(namename_dict, cameraDir,tkysImageDir):
    ysys = cameraDir
    for lidar, camera in tqdm(namename_dict.items()):
        if os.path.isfile(os.path.join(tkysImageDir,'{}.jpg'.format(lidar))):
            os.remove(os.path.join(tkysImageDir,'{}.jpg'.format(lidar)))
        else:
            pass
        if camera == NONE:
            black_numpy = np.full((1200,1920,3), 0)
            cv.imwrite(os.path.join(tkysImageDir,'{}.jpg'.format(lidar)),black_numpy)
        else:
            shutil.copy(os.path.join(ysys,camera),tkysImageDir)
            os.rename(os.path.join(tkysImageDir,camera),os.path.join(tkysImageDir,'{}.jpg'.format(lidar)))

def copying_lwir(namename_dict, cameraDir,tkysImageDir):
    ysys = cameraDir
    for lidar, camera in tqdm(namename_dict.items()):
        if os.path.isfile(os.path.join(tkysImageDir,'{}.bin'.format(lidar))):
            os.remove(os.path.join(tkysImageDir,'{}.bin'.format(lidar)))
        else:
            pass
        if camera == NONE:
            
            empty_numpy = np.empty(shape=(0,1))
            saving = empty_numpy.tobytes()
            with open(os.path.join(tkysImageDir,'{}.bin'.format(lidar)),'wb') as f:
                f.write(saving)
        else:
            shutil.copy(os.path.join(ysys,camera),tkysImageDir)
            os.rename(os.path.join(tkysImageDir,camera),os.path.join(tkysImageDir,'{}.bin'.format(lidar)))
            

def event1saver(time_string,file_name,event1array,tkysDir):
    event_time_array = event1array[:,0]
    save_index = np.where(event_time_array == time_string)
    np.savetxt(os.path.join(tkysDir,'image_imu',f'{file_name}.txt'),event1array[save_index], delimiter='\t' )


def event2saver(time_string,file_name,event2array,tkysDir):
    event_time_array = event2array[:,0]
    save_index = np.where(event_time_array == time_string)
    np.savetxt(os.path.join(tkysDir,'lwir_imu',f'{file_name}.txt'),event2array[save_index], delimiter='\t' )


def Mission_completer(mission):
    first_number = 0
    numdict_reverse = {}
    for miss_ in mission:
        if miss_ == '\n':
            continue
        number = int(miss_.split()[1])+100000*first_number
        time = float(miss_.split()[0])
        numdict_reverse[number]=time
        if int(miss_.split()[1]) == 65535:
            first_number +=1
    return numdict_reverse


def finish(baseDir, event1Mission,event2Mission,cameraDir,lwirDir,tkysDir,tkysImageDir,tkysLwirImageDir):
    lidar_filename = os.listdir(os.path.join(tkysDir,'pointCloudFrame'))
    lidar_filename = natsort.natsorted(lidar_filename)
    time_name_dict = {} # 파일이름 : imutime   ex)    '0000012321' : 183737.123412
    time_imu = []
    event1array = np.loadtxt(os.path.join(baseDir,'event1.txt'),delimiter='\t')
    event2array = np.loadtxt(os.path.join(baseDir,'event2.txt'),delimiter='\t')

    for a in tqdm(lidar_filename):
        temp = os.path.join(tkysDir,'imu/{}.txt'.format(os.path.splitext(os.path.basename(a))[0]))
        with open(temp,'r') as f:
            ss = f.readline()
            time_imu.append(float(ss.split()[0]))
            time_name_dict[a.split('.')[0]]= float(ss.split()[0])

    with open(os.path.join(tkysDir,'pointCloudFrame_timestamp.txt'),'w') as k:
        for tm in time_imu:
            k.write(f'{tm}\n')
            
    print('IMAGE_______IMAGE_______IMAGE_______IMAGE_______IMAGE________________')
    with open(event1Mission,'r') as g:
        mission = g.readlines()
        numdict_reverse = Mission_completer(mission)# number:time
    file_mission_dict = get_file_mission(numdict_reverse,cameraDir)
    namename_dict = get_name_dict(file_mission_dict, time_name_dict,event1array,tkysDir)
    print('           COPYING_RENAMING')
    copying(namename_dict, cameraDir,tkysImageDir)
    
    print('LWIR________LWIR________LWIR________LWIR________LWIR________________')
    with open(event2Mission,'r') as z:
        mission_l = z.readlines()
        numdict_reverse_l = Mission_completer(mission_l)
    file_mission_dict = get_file_mission_lwir(numdict_reverse_l,lwirDir)
    namename_dict_l = get_name_dict_lwir(file_mission_dict, time_name_dict,event2array,tkysDir)
    print('           COPYING_RENAMING')
    copying_lwir(namename_dict_l, lwirDir,tkysLwirImageDir)
    