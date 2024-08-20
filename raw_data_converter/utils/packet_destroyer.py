import dpkt
import os
import time
import math
from tqdm import tqdm
import numpy as np
import datetime

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

def utc_to_weekseconds(utc,leapseconds):
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    tdiff = utc -epoch  + datetime.timedelta(seconds=leapseconds)
    gpsweek = tdiff.days // 7
    gpsdays = tdiff.days - 7*gpsweek
    gpsseconds = tdiff.seconds + 86400* (tdiff.days -7*gpsweek)
    return int(gpsseconds)+ int(tdiff.microseconds)/1000000

def totimestamp(dt, epoch=datetime.datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6



class PacketDestroyer():
    def __init__(self, pcap, Block = True, withzero = False, velo16=False):
        receivetime, buf = pcap #init 내의 지역변수
        eth =dpkt.ethernet.Ethernet(buf)
        self.pactime = receivetime
        packet_data = eth.data.data.data #1206bytes data without header
        self.packet_data = np.frombuffer(packet_data, dtype=np.uint8).astype(np.uint32)
        #self.packet_azimuth = (self.packet_data[2] + self.packet_data[3] * 256) / 100  #no need
        self.block_list = []
        self.timestamp = self.gettimestamp()
        self.realtime = self.gettime()
        #self.utc_time = totimestamp(self.realtime)
        self.week_time = utc_to_weekseconds(self.realtime, leapseconds=18)
        self.coordinate_stack = np.array([])
        if Block:
            self.getblock(withzero,velo16)
        

    def read_uint32(self, data, idx):
        return data[idx] + data[idx+1]*256 + data[idx+2]*256*256 + data[idx+3]*256*256*256


   
    def gettimestamp(self):
        timestamps = self.read_uint32(self.packet_data[1200:1204],0)
        return timestamps
    

    def gettime(self):
        utc = str(datetime.datetime.utcfromtimestamp(self.pactime))
        utc_date = utc.split()[0]
        utc_time = utc.split()[1]
        utc_hour = int(utc_time.split(':')[0])
        utc_min = int(utc_time.split(':')[1])
        timestamp_min = self.timestamp//60000000
        timestamp_sec = (self.timestamp%60000000)/1000000
        if abs(timestamp_min -utc_min) >50:
            if timestamp_min > utc_min:
                utc_hour-=1
            elif timestamp_min < utc_min:
                utc_hour+=1
        utc_time = f' {utc_hour}:{timestamp_min}:{timestamp_sec:.6f}'
        real = utc_date+utc_time
        try:
            utc_realtime = datetime.datetime.strptime(real,'%Y-%m-%d %H:%M:%S.%f')
        except:
            utc_realtime = datetime.datetime.strptime(real,'%Y-%m-%d %H:%M:%S.%f')
        return utc_realtime
    
    def getblock(self,boola,boolb):
        for bl in range(1,13):
            blocknum=bl-1 
            temp_block = BlockDestroyer(self.packet_data[100*bl-100:100*bl], blocknum, withzero = boola,velo16= boolb)
            if np.size(self.coordinate_stack)==0:
                self.coordinate_stack =temp_block.coordinate_array
            elif np.size(temp_block.coordinate_array)!=0:
                self.coordinate_stack= np.vstack((self.coordinate_stack,temp_block.coordinate_array))

class BlockDestroyer():
    def __init__(self,block, blocknumber,withzero,velo16):
        self.block = block #uint32 array 100bytes
        self.block_number = blocknumber
        self.block_azimuth = (block[2] + block[3] * 256) / 100
        self.firing = block[4:].reshape(32, 3)
        self.distances = (self.firing[:, 0] + self.firing[:, 1] * 256)*4/1000 #m 단위
        self.intensities = self.firing[:, 2]
        self.coordinate_array = np.array([])
        if velo16:
            self.getcoordinate_velo16()
        elif withzero:
            self.getcoordinate_withzero()
        else:
            self.getcoordinate_numpy()


    def putoffset(self,index):
        elevation_offset_list = [-25,-1,-1.667,-15.639,-11.31,0,-0.667,-8.843,-7.254,0.333,-0.333,-6.148,-5.333,1.333,0.667,-4,-4.667,1.667,1,-3.667,-3.333,3.333,2.333,-2.667,-3,7,4.667,-2.333,-2,15,10.333,-1.333]
        azimuth_offset_list = [1.4,-4.2,1.4,-1.4,1.4,-1.4,4.2,-1.4,1.4,-4.2,1.4,-1.4,4.2,-1.4,4.2,-1.4,1.4,-4.2,1.4,-4.2,4.2,-1.4,1.4,-1.4,1.4,-1.4,1.4,-4.2,4.2,-1.4,1.4,-1.4]
        return elevation_offset_list[index], azimuth_offset_list[index]

    def putoffset_velo16(self, index):
        elevation_offset_list = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15,-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        return elevation_offset_list[index]

    
    
    def getcoordinate_withzero(self):
        a = self.block_azimuth
        
        distance_list = self.distances.flatten()
        reflectivity_list = self.intensities.flatten()

        for i in range(len(distance_list)):
            R = distance_list[i]
            w,b = self.putoffset(i)
            c = a+b
            if c > 360:
                c = a+b-360

            c = math.radians(c)
            w = math.radians(w)
            x = R*math.cos(w)*math.sin(c)
            y = R*math.cos(w)*math.cos(c)
            z = R*math.sin(w)
            coordinate_arr = np.array([x,y,z,reflectivity_list[i],i,self.block_number*0.000055296])
            if np.size(self.coordinate_array) != 0:
                self.coordinate_array=np.vstack((self.coordinate_array,coordinate_arr))
            else:
                self.coordinate_array = coordinate_arr     

    def getcoordinate_velo16(self):
        c = self.block_azimuth
        
        distance_list = self.distances.flatten()
        reflectivity_list = self.intensities.flatten()

        for i in range(len(distance_list)):
            R = distance_list[i]
            if R == 0 or self.block_number%2==0:
                continue
            w = self.putoffset_velo16(i)
            
            c = math.radians(c)
            w = math.radians(w)
            x = R*math.cos(w)*math.sin(c)
            y = R*math.cos(w)*math.cos(c)
            z = R*math.sin(w)
            if i<16:
                coordinate_arr=np.array([x,y,z,reflectivity_list[i],i,(self.block_number)*0.000055296])
            elif i>15:
                coordinate_arr=np.array([x,y,z,reflectivity_list[i],i-16,(self.block_number+1)*0.000055296])
            else:
                pass

            if np.size(self.coordinate_array) != 0:
                self.coordinate_array=np.vstack((self.coordinate_array,coordinate_arr))
            else:
                self.coordinate_array = coordinate_arr 

    def getcoordinate_numpy(self):
        a = self.block_azimuth
        distance_arr = self.distances.reshape(-1,1)
        zero_index = np.where(distance_arr != 0)
        laser_id = np.arange(0,32).reshape(-1,1)
        laser_timeoffset = np.full(32,0.000055296*self.block_number).reshape(-1,1)
        reflectivity_arr = self.intensities.reshape(-1,1)
        elevation_offset_list = [-25,-1,-1.667,-15.639,-11.31,0,-0.667,-8.843,-7.254,0.333,-0.333,-6.148,-5.333,1.333,0.667,-4,-4.667,1.667,1,-3.667,-3.333,3.333,2.333,-2.667,-3,7,4.667,-2.333,-2,15,10.333,-1.333]
        azimuth_offset_list = [1.4,-4.2,1.4,-1.4,1.4,-1.4,4.2,-1.4,1.4,-4.2,1.4,-1.4,4.2,-1.4,4.2,-1.4,1.4,-4.2,1.4,-4.2,4.2,-1.4,1.4,-1.4,1.4,-1.4,1.4,-4.2,4.2,-1.4,1.4,-1.4]
        c_array = a+np.array(azimuth_offset_list).reshape(-1,1)
        c_array[c_array>360] -= 360
        c_array_rad = np.deg2rad(c_array)
        elevation_offset_list_rad = np.deg2rad(np.array(elevation_offset_list).reshape(-1,1))
        x = (distance_arr*np.cos(elevation_offset_list_rad)*np.sin(c_array_rad)).reshape(-1,1)
        y = (distance_arr*np.cos(elevation_offset_list_rad)*np.cos(c_array_rad)).reshape(-1,1)
        z = (distance_arr*np.sin(elevation_offset_list_rad)).reshape(-1,1)
        self.coordinate_array = np.hstack([x,y,z,reflectivity_arr,laser_id,laser_timeoffset])
        self.coordinate_array = self.coordinate_array[zero_index[0],:]


######## imu 랑 매치해서 자르는 부분##########################################################


def start_event1(event1):
    with open(event1,'r') as g:
        event1_file = g.readlines()
        start = float(event1_file[1].split()[0])
        return start

def save_numpy(initialData, imu_index,tkysDir):
    with open(os.path.join(tkysDir,f'pointCloudPackets/{str(imu_index).zfill(10)}.bin'),'wb') as f:
        saving = initialData.tobytes()        
        f.write(saving)



def startt(tkysDir,output,pcap,event1):
    start_event = start_event1(event1)
    with open(output,'r') as f:
        output_file = f.readlines()
        output_filen=[]
        output_file_new =[]
        for i in tqdm(range(len(output_file))):
            if output_file[i]=='\n':
                continue
            elif float(output_file[i].split()[0]) < start_event+0.05:
                continue
            else:
                output_filen.append(float(output_file[i].split()[0]))
                output_file_new.append(output_file[i])
            
        output_timestart = output_filen[0]-0.0026
        output_timeend = output_filen[-2]+0.0026

    with open(pcap,'rb') as f:
        pcap=dpkt.pcap.Reader(f)
        print('loading_end')
    
        imu_index =0
        pack_list =[]
        initialData=np.empty(shape=(0,6))
        start_time =[]
        end_time =[]

        for pcap_ in tqdm(pcap):
            eth = dpkt.ethernet.Ethernet(pcap_[1])
            try:
                if eth.data.data.sport == 8308:
                    continue
                elif not output_timestart< PacketDestroyer(pcap_, Block=False).week_time <output_timeend:
                    continue
                else:
                    temp = PacketDestroyer(pcap_)
                    if temp.coordinate_stack.size == 0:
                        pass
                    else:
                        initialData = np.vstack((initialData,temp.coordinate_stack))
                    pack_list.append(temp.week_time)
            except:
                continue
                
            if temp.week_time-output_filen[imu_index]>0.00251:
                with open(os.path.join(tkysDir,f'imu/{str(imu_index+1).zfill(10)}.txt'),'w') as g:
                    g.write(output_file_new[imu_index])
                save_numpy(initialData,imu_index+1,tkysDir)

                imu_index+=1
                start_time.append(pack_list[0])
                end_time.append(pack_list[-1])

                
                initialData=np.empty(shape=(0,6))
                pack_list = []
                
        np.savetxt(os.path.join(tkysDir,'pointCloudPackets_timstamp_start.txt'),np.array(start_time))
        np.savetxt(os.path.join(tkysDir,'pointCloudPackets_timstamp_end.txt'),np.array(end_time))


def start_velo16(tkysDir,output,pcap):
    with open(output,'r') as f:
        output_file = f.readlines()
        output_filen=[]
        output_file_new =[]
        for i in tqdm(range(len(output_file))):
            if output_file[i]=='\n':
                continue
            else:
                output_filen.append(float(output_file[i].split()[0]))
                output_file_new.append(output_file[i])
            
        output_timestart = output_filen[0]-0.0026
        output_timeend = output_filen[-1]+0.0026

    with open(pcap,'rb') as f:
        pcap=dpkt.pcap.Reader(f)
        print('loading_end')
    
        imu_index =0
        pack_list =[]
        initialData=np.empty(shape=(0,6))
        start_time =[]
        end_time =[]

        for pcap_ in tqdm(pcap):
            eth = dpkt.ethernet.Ethernet(pcap_[1])
            try:
                if eth.data.data.sport == 8308:
                    continue
                elif not output_timestart< PacketDestroyer(pcap_, Block=False).week_time <output_timeend:
                    continue
                else:
                    temp = PacketDestroyer(pcap_,velo16=True)
                    if temp.coordinate_stack.size == 0:
                        pass
                    else:
                        initialData = np.vstack((initialData,temp.coordinate_stack))
                    pack_list.append(temp.week_time)
            except:
                continue
                
            if temp.week_time-output_filen[imu_index]>0.00251:
                with open(os.path.join(tkysDir,f'imu/{str(imu_index+1).zfill(10)}.txt'),'w') as g:
                    g.write(output_file_new[imu_index])
                save_numpy(initialData,imu_index+1,tkysDir)

                imu_index+=1
                start_time.append(pack_list[0])
                end_time.append(pack_list[-1])

                
                initialData=np.empty(shape=(0,6))
                pack_list = []
                
        np.savetxt(os.path.join(tkysDir,'pointCloudPackets_timstamp_start.txt'),np.array(start_time))
        np.savetxt(os.path.join(tkysDir,'pointCloudPackets_timstamp_end.txt'),np.array(end_time))


#----------------------------------start------------------------------
if __name__ =='__main__':

    

           



    tkysDir = '/esail3/yunsoo/NewKitti_B/'
    event1 = '/esail3/Tackang/97CarsenseData/210803_Suwon_Zelkova/event1.txt'
    output ='/esail3/Tackang/97CarsenseData/210803_Suwon_Zelkova/pospac/Mission 1/Export/output_withAccel.txt'
    pcap ='/esail3/Tackang/97CarsenseData/210803_Suwon_Zelkova/2_lidar/2021-08-03-11-59-37_VLP_32C.pcap'
    startt(tkysDir,output,pcap,event1)



#event1 시작이랑 0.5이내로 차이날때 스위치 on
#기존방식대로 빙빙 돌고 imu 20개 누적시마다 코드+삭제

