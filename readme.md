<h1 align = "center">Single and Stereo Calibration and Rectification with OpenCV (calibrateCamera, stereoCalibrate, and stereoRectify)</h1>

# 1. Introduction  
&emsp;Everyone has probably used OpenCV and cameras before, so today, let's talk about how to use photos taken with two cameras and OpenCV to perform calibration and stereo rectification. There are many theoretical explanations for camera calibration, so I will just provide three well-written posts for reference:

[fengye2two的帖子-标定
](https://blog.csdn.net/fengye2two/article/details/80686409)

[卍卐没想到的帖子-标定
](https://blog.csdn.net/a083614/article/details/78579163)

[瞻邈-立体校正](https://blog.csdn.net/xhtchina/article/details/126492195)

&emsp;In short, I won’t dive deep into the theory. Instead, I will briefly discuss the implementation through code.
&emsp;For the calibration part, part of the code is referenced from:
[Temuge Batpurev's Blog](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html)
&emsp;For the code to draw epipolar lines, the reference is:
[逆光525的帖子-绘制极线](https://blog.csdn.net/qq_22059843/article/details/103400094)
&emsp;Additionally, the data and complete code I used are available here:
[Repo : Calibrate-and-Rectify
](https://github.com/QuanHaHQuan/Calibrate-and-Rectify/blob/main/readme.md)
# 2. 单目标定
## 2.1 先看代码

```python
chessboard_size = (9, 6)
frame_size = (640, 480)

# Setting up the object points of the chessboard
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
square_size = 1
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints_main = [] # 2d points in image plane
images = glob.glob('demo/left*.jpg')
images = sorted(images)
print(f"Found {len(images)} images for calibration")

for idx, image_file in enumerate(images):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret == True:
        objpoints.append(objp)

        # Refine the corner locations (to subpixel accuracy)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints_main.append(corners2)
        
    else:
        print(f"Chessboard corners not found in image: {image_file}")

# Camera calibration
ret_main, mtx_main, dist_main, rvecs_main, tvecs_main = cv2.calibrateCamera(objpoints, imgpoints_main, frame_size, None, None)

print(ret_main)

imgpoints_side = [] # 2d points in image plane.
images = glob.glob('demo/right*.jpg')
images = sorted(images)
print(f"Found {len(images)} images for calibration")

for idx, image_file in enumerate(images):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret == True:

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints_side.append(corners2)
        
    else:
        print(f"Chessboard corners not found in image: {image_file}")

# Camera calibration
ret_side, mtx_side, dist_side, rvecs_side, tvecs_side = cv2.calibrateCamera(objpoints, imgpoints_side, frame_size, None, None)

print(ret_side)
```
## 2.2 Explanation
&emsp;First, the objp variable represents the world coordinates in the chessboard coordinate system. For simplicity, I assume the calibration board is aligned with the xy-plane, so all z-coordinates of the points are zero.
&emsp;Next is the calibrateCamera function. The specific input parameters are all listed above. It's important to note that the returned error (RMSE) should ideally be below 0.5 for good calibration. If the error is large, you might need to adjust your calibration images, for example, by improving lighting conditions or adding more diverse angles.

## 2.3 calibrateCamera参数

```python
    # Single camera calibration using calibrateCamera
    # ret, mtx, dist, rvecs, tvecs = 
    # cv2.calibrateCamera(objpoints, imgpoints, imageSize, cameraMatrix, distCoeffs)
    # Input parameters:
    # objpoints: List of object points. These are the 3D points in the world coordinate system, usually the chessboard corners.
    # imgpoints: List of image points. These are the 2D points in the image plane, usually the detected chessboard corners.
    # imageSize: Size of the image, format as (width, height).
    # cameraMatrix (optional): Initial camera matrix. If None, the function will compute an initial value.
    # distCoeffs (optional): Initial distortion coefficients. If None, the function will compute an initial value.
    # Output parameters:
    # ret: Root mean square error (RMSE), indicating the accuracy of the calibration.
    # mtx: Camera matrix.
    # dist: Distortion coefficients.
    # rvecs: List of rotation vectors, representing the rotation of the camera for each view.
    # tvecs: List of translation vectors, representing the translation of the camera for each view.
```

# 3. 双目标定
## 3.1 先看代码

```python
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
ret, mtx_main, dist_main, mtx_side, dist_side, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_main, imgpoints_side,
    mtx_main, dist_main, mtx_side, dist_side,
    frame_size, criteria=criteria, flags=flags)
```
## 3.2 stereoCalibrate参数

```python
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
```

# 4. 立体校正
## 4.1 先看代码

```python
# 立体校正
# 立体校正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_main, dist_main, mtx_side, dist_side, frame_size, R, T)

# 对测试图像进行校正
def rectify_image(img, mtx, dist, R, P):
    h, w = img.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, R, P, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# 读取测试图片
img_main = cv2.imread('demo/left03.jpg')
img_side = cv2.imread('demo/right03.jpg')

# 校正图像
rectified_main = rectify_image(img_main, mtx_main, dist_main, R1, P1)
rectified_side = rectify_image(img_side, mtx_side, dist_side, R2, P2)

# 保存校正后的图像
cv2.imwrite('rectified_main.png', rectified_main)
cv2.imwrite('rectified_side.png', rectified_side)
```
## 4.2 一点解释
&emsp;其实也没啥好解释的吧，可能就是需要说一下这个initUndistortRectifyMap和这个remap函数是啥子。简单来说哈，第一个函数就是根据相机自己的内参数和两个相机之间的外参数，生成一个映射来消除图片的畸变（为啥要消除可以看上面的理论贴）。第二个函数就是一个映射过程，根据上一个函数生成的映射执行这个映射过程，得到矫正之后的图片。具体的参数我在下面也解释一下吧。

## 4.3 stereoRectify参数

```python
# R1, R2, P1, P2, Q, roi1, roi2 = 
# cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, 
#                   R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=(0, 0))
# 输入参数：
# cameraMatrix1: 左相机内参矩阵。
# distCoeffs1: 左相机畸变系数。
# cameraMatrix2: 右相机内参矩阵。
# distCoeffs2: 右相机畸变系数。
# imageSize: 图像的大小，格式为 (width, height)。
# R: 旋转矩阵，将右相机坐标系转换到左相机坐标系。
# T: 平移向量，将右相机坐标系转换到左相机坐标系。
# flags (可选): 校正类型标志。
# alpha (可选): 自由参数，范围为 [0, 1]，决定图像边缘区域的裁剪程度。
# newImageSize (可选): 新图像的大小。
# 输出参数：
# R1: 左相机的校正变换（旋转矩阵）。
# R2: 右相机的校正变换（旋转矩阵）。
# P1: 左相机的新投影矩阵。
# P2: 右相机的新投影矩阵。
# Q: 视差-深度映射矩阵。
# roi1: 左相机图像的有效区域。
# roi2: 右相机图像的有效区域。
```
## 4.4 initUndistortRectifyMap参数
```python
# 计算畸变和矫正的映射 initUndistortRectifyMap
# map1, map2 = 
# cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type)
# 输入参数：
# cameraMatrix: 相机内参矩阵。
# distCoeffs: 相机畸变系数。
# R: 校正变换（旋转矩阵）。
# newCameraMatrix: 新的相机矩阵。
# size: 图像的大小，格式为 (width, height)。
# m1type: 输出映射的类型，可以是 cv2.CV_32FC1 或 cv2.CV_16SC2 等。
# 输出参数：
# map1: 第一张映射表，用于 remap 函数。
# map2: 第二张映射表，用于 remap 函数。
```
## 4.5 remap参数

```python
# 重映射函数 remap
# dst = cv2.remap(src, map1, map2, interpolation, borderMode, borderValue)
# 输入参数：
# src: 输入图像。
# map1: 第一张映射表。
# map2: 第二张映射表。
# interpolation: 插值方法，如 cv2.INTER_LINEAR 或 cv2.INTER_CUBIC。
# borderMode (可选): 边界模式，定义如何处理图像边界，如 cv2.BORDER_CONSTANT 或 cv2.BORDER_REPLICATE。
# borderValue (可选): 边界值，如果使用 cv2.BORDER_CONSTANT 时使用。
# 输出参数：
# dst: 校正后的图像。
```

# 5. 绘制极线

## 5.1 先看代码
```python
plt.figure(figsize=(20, 20))

for i in range(0,1):  # 以第一对图片为例
    im_L=Image.fromarray(rectified_main) # numpy 转 image类
    im_R=Image.fromarray(rectified_side) # numpy 转 image 类

    width = im_L.size[0]*2
    height = im_L.size[1]

    img_compare = Image.new('RGBA',(width, height))
    img_compare.paste(im_L,box=(0,0))
    img_compare.paste(im_R,box=(640,0))
    
    #在已经极线对齐的图片上均匀画线
    for i in range(1,20):
        len=480/20
        plt.axhline(y=i*len, color='r', linestyle='-')
    plt.imshow(img_compare)
    plt.savefig('epipolar_lines.png', bbox_inches='tight', pad_inches=0)
    plt.show()
```
## 5.2 一点解释
&emsp;注意哈，因为有点怕麻烦，我这里不是去算了那个极线，然后画的线。我是直接画了水平线作为极线，然后手动去看在同一个水平线上的点是不是对应的。当然了对于标定图片，你也可以直接使用角点作为对应点，画一下极线来看一看。但是如果不是标定图片，可能就需要用SIFT关键点匹配了，这一点OpenCV官方有：[官方教程](https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html)

## 5.3 校正结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14e579a7727f4ce1a6f2198bfa9bef75.png)
&emsp;OK，基本符合预期哦~
