<h1 align = "center">OpenCV下的单目标定，双目标定与立体校正(calibrateCamera, stereoCalibrate and stereoRectify)</h1>

# 1. 杂话   
&emsp;大伙儿应该都用过OpenCV和相机吧，所以今天咱们就来说说怎么使用两个相机拍摄的照片和OpenCV来进行标定和立体校正。相机标定的理论解释其实有很多啦，我就随便找两个写得不错的帖子给大家参考一下哈：

[fengye2two的帖子-标定
](https://blog.csdn.net/fengye2two/article/details/80686409)

[卍卐没想到的帖子-标定
](https://blog.csdn.net/a083614/article/details/78579163)

[瞻邈-立体校正](https://blog.csdn.net/xhtchina/article/details/126492195)

&emsp;总而言之，我就不那个班门弄斧关公面前耍大刀了，我就简单说说代码层面的实现。对了，我使用的数据在这里：

# 2. 单目标定
## 2.1 先看代码

```python
def calibrate_camera(images_folder):
    images_names = []
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 4 #number of checkerboard rows.
    columns = 7 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
        
            objpoints.append(objp)
            imgpoints.append(corners)
    
    # 单目标定 calibrateCamera
    # ret, mtx, dist, rvecs, tvecs = 
    # cv2.calibrateCamera(objpoints, imgpoints, imageSize, cameraMatrix, distCoeffs)
    #  输入参数：
    # objpoints: 物体点坐标的列表。这些是 3D 世界坐标，通常是棋盘格的角点坐标。
    # imgpoints: 图像点坐标的列表。这些是 2D 图像坐标，通常是从图像中检测到的棋盘格角点的坐标。
    # imageSize: 图像的大小，格式为 (width, height)。
    # cameraMatrix (可选): 初始的相机内参矩阵。如果传入 None，则函数会计算一个初始值。
    # distCoeffs (可选): 初始的畸变系数。如果传入 None，则函数会计算一个初始值。
    # 输出参数：
    # ret: 平均重投影误差（root mean square error, RMSE），表示校准结果的精度。
    # mtx: 相机内参矩阵（camera matrix）。
    # dist: 畸变系数（distortion coefficients）。
    # rvecs: 旋转向量列表，表示每个视角的旋转。
    # tvecs: 平移向量列表，表示每个视角的平移。
 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
 
    return mtx, dist, imgpoints, objpoints
```
## 2.2 一点解释
&emsp;首先是这个objp，这玩意说白了就是世界坐标系的坐标，但是这个世界坐标系不是咱们可以自己定义嘛对吧。方便起见，我就直接假设标定板的平面就是x-y平面，所以上面所有的焦点的z坐标不久都是0了嘛对吧哈哈哈。
&emsp;然后就是calibrateCamera这个函数，具体的输入参数是都写在上面了，需要注意的是，一般来说返回的误差也就是rmse在0.5以下会比较好，如果很大的话，那么你的标定图像就需要好好调整一下了，比如光照好一点，加入更多的角度之类的。
# 3. 双目标定
## 3.1 先看代码

```python
def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = []
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
    
    print(c1_images_names)
    print(c2_images_names)
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 4 #number of checkerboard rows.
    columns = 7 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (4, 7), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (4, 7), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            
    # 双目标定函数 
    # ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = 
    # cv2.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, 
    #                       cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
    #                       criteria=criteria, flags=flags)
    
    # 输入参数：
    # objectPoints: 物体点坐标的列表，类似 calibrateCamera 函数中的 objpoints。
    # imagePoints1: 左相机的图像点坐标列表。
    # imagePoints2: 右相机的图像点坐标列表。
    # cameraMatrix1: 左相机的初始内参矩阵。
    # distCoeffs1: 左相机的初始畸变系数。
    # cameraMatrix2: 右相机的初始内参矩阵。
    # distCoeffs2: 右相机的初始畸变系数。
    # imageSize: 图像的大小，格式为 (width, height)。
    
    # criteria (可选): 终止条件，用于优化算法的迭代过程。
        # cv2.TERM_CRITERIA_MAX_ITER：当达到最大迭代次数时停止。
        # cv2.TERM_CRITERIA_EPS：当参数变化小于设定的精度时停止。
        # 可以组合使用，例如cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS，表示当满足任意一个条件时停止。
        # 最大迭代次数：一个整数，指定最大迭代次数。例如，100表示最多迭代100次。
        # 精度阈值：一个浮点数，指定参数变化小于该值时停止迭代。例如，1e-5表示当参数变化小于0.00001时停止迭代。
    
    # flags (可选): 标志位，用于指定某些参数的固定或自由度。
        # cv2.CALIB_FIX_INTRINSIC：在标定过程中保持两个相机的内参数矩阵不变。这意味着在双目标定过程中不会重新估计每个相机的内参数矩阵（包括焦距、光轴中心等），而是使用单个相机标定结果中得到的内参数。
        # cv2.CALIB_USE_INTRINSIC_GUESS：使用传入的内参数作为初始猜测值，并在标定过程中对其进行优化。这对提高标定精度很有帮助，特别是在内参数已经较准确的情况下。
        # cv2.CALIB_FIX_PRINCIPAL_POINT：保持主点（光轴中心）固定不变。
        # cv2.CALIB_FIX_FOCAL_LENGTH：保持焦距不变。
        # cv2.CALIB_FIX_ASPECT_RATIO：保持焦距的长宽比不变。
        # cv2.CALIB_ZERO_TANGENT_DIST：假设切向畸变参数为零并保持不变。
        # cv2.CALIB_RATIONAL_MODEL：使标定函数使用一个带有6个畸变系数的合理模型。
        # cv2.CALIB_SAME_FOCAL_LENGTH：假设两个摄像头具有相同的焦距。
    
    # 输出参数：
    # ret: 平均重投影误差。
    # cameraMatrix1: 校准后的左相机内参矩阵。
    # distCoeffs1: 校准后的左相机畸变系数。
    # cameraMatrix2: 校准后的右相机内参矩阵。
    # distCoeffs2: 校准后的右相机畸变系数。
    # R: 旋转矩阵，将右相机坐标系转换到左相机坐标系。
    # T: 平移向量，将右相机坐标系转换到左相机坐标系。
    # E: 基础矩阵。
    # F: 本质矩阵。
    
 
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T, c1_images, c2_images
```

# 4. 立体校正
