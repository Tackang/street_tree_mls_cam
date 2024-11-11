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

def lidar2cam(pointCloud, PIXEL_WIDTH=1920, PIXEL_HEIGHT=1200):
        '''
        ### Notice
        project point cloud to image plane\n
        pixel coordinate added to the last two columns
        '''
        
        # Loading Parameters
        ''' velo2cam.txt 
        Number: 0
        LidarImageOffset
        -1
        Angle : LR, UD, Rot  (in radians)
        1.556375,0.005906,-0.000375
        TranslationMatrix
        -0.004300,-0.047500,-0.099400
        intrinsicCameraParameter (fx,fy,cx,cy,k1,k2,p1,p2)
        1058.688334,1059.889393,957.887598,614.085426,-0.126184,0.057164,-0.000468,-0.001805
        --------------------------------------------------------------------------------
        '''
        pointCloud_frontView=pointCloud[pointCloud[:,1]>=0,:]
        LR = 1.556375
        UD = 0.005906
        Rot = -0.000375
        # LR = 1.559375
        # UD = 0.005500
        # Rot = 0.001250

        lidar2cam_translation = np.array(
            [[-0.004300],
            [-0.047500],
            [-0.099400],]
            )
        lidar2cam_rotation = getCameraRotation(LR,UD,Rot)
        
        cameraIntrinsicParameter = np.array(
            [[1058.688334, 0, 957.887598],
            [0, 1059.889393, 614.085426],
            [0, 0, 1],]
            )
        cameraDistortion = np.array([-0.126184,0.057164,-0.000468,-0.001805])
        # cameraDistortion = np.array([-0.116809,0.057164,-0.000468,-0.001805])

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
        
        
        # runningTime=time.time()-startTime
        # print("[fusion]LIDAR2CAM DONE. Running Time : {}".format(runningTime))
        
        return pointCloudInImage

def getSegInfo(pointCloud, seg):
    '''
    ### Notice
    pointCloud should have pixel coordinate at the last two columns
    '''
    
    seg[pointCloud[:,-1].astype(np.int32)-1,pointCloud[:,-2].astype(np.int32)-1]*=2
    segLoc_x,segLoc_y = np.where(seg==510)
    segLoc=np.vstack([segLoc_y,segLoc_x]).T+1
    segId = np.argmin((pointCloud[:,-2][:,np.newaxis]-segLoc[:,0][np.newaxis,:])**2 + (pointCloud[:,-1][:,np.newaxis]-segLoc[:,1][np.newaxis,:])**2,axis=0)
    pointCloud=pointCloud[segId,:]
    # pointCloud=pointCloud[:,[0,1,2,3,6,-2,-1]
    # pointCloud=pointCloud[pointCloud[:,4]!=9999,:]

    return pointCloud

def getRGBInfo(pointCloud, image):
    '''
    ### Notice
    pointCloud should have pixel coordinate at the last two columns
    '''

    imageR = image[:,:,0]
    imageG = image[:,:,1]
    imageB = image[:,:,2]
    defaultLabel= np.full((pointCloud.shape[0],3),0)
    pointCloudRGB=np.hstack([pointCloud,defaultLabel])
    
    for i in range(0,pointCloud.shape[0]):
        pixelX = int(pointCloud[i,-2]-1)
        pixelY = int(pointCloud[i,-1]-1)
        pointCloudRGB[i,-3] = imageR[pixelY,pixelX]/255
        pointCloudRGB[i,-2] = imageG[pixelY,pixelX]/255
        pointCloudRGB[i,-1] = imageB[pixelY,pixelX]/255
    return pointCloudRGB

def getBBinfo(pointCloud,boundingBox,PIXEL_WIDTH=1920, PIXEL_HEIGHT=1200):

    # BoundingBox column : 0)classID; 1)xMin; 2)xMax; 3)yMin; 4)yMax; 5)confidence score;
    # Label = 9999 is non-labeled
    
    defaultLabel= np.full((pointCloud.shape[0],1),9999)
    pointCloud=np.hstack((pointCloud,defaultLabel))
    if type(boundingBox)==int:
        return pointCloud
    
    boundingBox=boundingBox[boundingBox[:,5].argsort()]
    
    for i in range(0,boundingBox.shape[0]):
        # get the interested pixel coordinates
        temp1=np.arange(boundingBox[i,1],boundingBox[i,2])
        temp2=np.arange(boundingBox[i,3],boundingBox[i,4])
        temp11,temp22=np.meshgrid(temp1,temp2)
        temp3=np.vstack((np.ravel(temp11),np.ravel(temp22))).T.astype(np.int64)
       
        # give value to interested pixels
        X=np.full((PIXEL_HEIGHT,PIXEL_WIDTH),0)
        X[temp3[:,1]-1,temp3[:,0]-1]=1
       
        # get pointcloud pixel coordinates
        Y=pointCloud[:,-8:-6].astype(np.int32)
        X[Y[:,1]-1,Y[:,0]-1]*=2
        loc_x,loc_y=np.where(X>=2)
        loc=np.vstack([loc_y,loc_x])
        location=np.transpose(loc)+1
        
        id = np.argmin(
            (pointCloud[:,-8][:,np.newaxis]-location[:,0][np.newaxis,:])**2 + (pointCloud[:,-7][:,np.newaxis]-location[:,1][np.newaxis,:])**2,
            axis=0
            )
        pointCloud[id,-1]=boundingBox[i,0] 
    
    # pointCloud=pointCloud[pointCloud[:,-1]!=9999,:]  
    
    return pointCloud

