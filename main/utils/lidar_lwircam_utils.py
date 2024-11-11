import numpy as np

def getCameraRotation(LR,UD,Rot):
        unknownRotation=np.array(
            [[0,1,0],
            [0,0,-1],
            [-1,0,0]],
            )
        rollRotation=np.array(
            [[1,0,0],
            [0,np.cos(Rot),-np.sin(Rot)],
            [0,np.sin(Rot),np.cos(Rot)]],
            )
        pitchRotation=np.array(
            [[np.cos(UD),0,np.sin(UD)],
            [0,1,0],
            [-np.sin(UD),0,np.cos(UD)]],
            )
        yawRotation=np.array(
            [[np.cos(LR),-np.sin(LR),0],
            [np.sin(LR),np.cos(LR),0],
            [0,0,1]],
            )
        rotationMatrix= np.dot(np.dot(np.dot(unknownRotation,rollRotation),pitchRotation),yawRotation)
        return rotationMatrix

def lidar2lwircam(pointCloud, PIXEL_WIDTH=640, PIXEL_HEIGHT=512):
        '''
        ### Notice
        project point cloud to image plane\n
        pixel coordinate added to the last two columns
        '''

        # Loading Parameters
        ''' velo2lwir.txt 
        Number: 1
        LidarImageOffset
        0
        Angle : LR, UD, Rot (in radians)
        1.549422,0.087593,0.013570
        TranslationMatrix
        0.013825,-0.030000,-0.161900
        intrinsicCameraParameter (fx,fy,cx,cy,k1,k2,p1,p2)
        434.905155,434.920710,324.480806,262.421627,-0.381497,0.117428,-0.006102,-0.000935
        --------------------------------------------------------------------------------
        '''
        pointCloud_frontView=pointCloud[pointCloud[:,1]>=0,:]
        LR = 1.549422
        UD = 0.087593
        Rot = 0.013570

        lidar2cam_translation = np.array(
            [[0.013825],
            [-0.030000],
            [-0.161900],]
            )
        lidar2cam_rotation = getCameraRotation(LR,UD,Rot)
        
        cameraIntrinsicParameter = np.array(
            [[434.905155, 0, 324.480806],
            [0, 434.920710, 262.421627],
            [0, 0, 1],]
            )
        # cameraDistortion = np.array([-0.126184,0.057164,-0.000468,-0.001805])
        cameraDistortion = np.array([-0.381497,0.117428,-0.006102,-0.000935])

        pointXYZ=pointCloud_frontView[:,0:3].T
        length=pointCloud_frontView.shape[0]
        
        k1=cameraDistortion[0]
        k2=cameraDistortion[1]
        p1=cameraDistortion[2]
        p2=cameraDistortion[3]

        '''LiDAR Coordinates To Camera Coordinates'''
        XYZc = np.dot(lidar2cam_rotation,(pointXYZ-lidar2cam_translation))
        
        #Normalized image plane
        xxx=XYZc[0,:]/XYZc[2,:]
        yyy=XYZc[1,:]/XYZc[2,:]
        xyu=np.vstack([xxx,yyy])
        
        #cameraDistortion fix
        ru2=xyu[0,:]**2+xyu[1,:]**2
        # kkk=1+k1*ru2+k2*ru2**2
        # kkk2= np.vstack([kkk,kkk])
        xu=xyu[0,:]
        yu=xyu[1,:]
        xyuu=xu*yu
        xuu=2*p1*xyuu+p2*(ru2+2*xu**2)
        yuu=p1*(ru2+2*yu**2)+2*p2*xyuu
        xyd=(1+k1*ru2+k2*ru2**2)*xyu+np.vstack([xuu,yuu])
        one=np.full((1,length),1)
        xydd=np.vstack([xyd,one])
        
        #getting pixel coordinates
        xyp=np.dot(cameraIntrinsicParameter,xydd)
        uv=np.around(np.transpose(xyp[0:2,:]))
        pointCloudWithPixel = np.hstack([pointCloud_frontView,uv])
        pointCloudInImage = pointCloudWithPixel[np.where((0<=pointCloudWithPixel[:,-2]) & (pointCloudWithPixel[:,-2]<=PIXEL_WIDTH) & (0<=pointCloudWithPixel[:,-1]) & (pointCloudWithPixel[:,-1]<=PIXEL_HEIGHT))]
        
        # pointCloud: 0)x; 1)y; 2)z; 3)intensity; 4)pixelU; 5)pixelV
        pointCloud=pointCloudInImage
        
        # runningTime=time.time()-startTime
        # print("[fusion]LIDAR2CAM DONE. Running Time : {}".format(runningTime))
        
        return pointCloud

def getThermalInfo(pointCloud,lwirImage):
    '''
    ### Notice
    pointCloud should have pixel coordinate at the last two columns
    '''
    # Label = 9999 is non-labeled
    defaultLabel= np.full((pointCloud.shape[0],1),9999)
    pointCloudWithLabel=np.hstack([pointCloud,defaultLabel])
    
    for i in range(0,pointCloud.shape[0]):
        pixelU = int(pointCloud[i,-2]-1)
        pixelY = int(pointCloud[i,-1]-1)
        pointCloudWithLabel[i,-1] = lwirImage[pixelY,pixelU]

    pointCloudWithTemperature = pointCloudWithLabel[pointCloudWithLabel[:,-1]!=9999,:]
    return pointCloudWithTemperature