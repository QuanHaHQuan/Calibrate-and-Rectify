<h1 align = "center">Single and Stereo Calibration and Rectification with OpenCV (calibrateCamera, stereoCalibrate, and stereoRectify)</h1>

# 1. Introduction  
&emsp;Everyone has probably used OpenCV and cameras before, so today, let's talk about how to use photos taken with two cameras and OpenCV to perform calibration and stereo rectification. There are many theoretical explanations for camera calibration, so I will just provide three well-written posts for reference:

&emsp;[fengye2two的帖子-标定
](https://blog.csdn.net/fengye2two/article/details/80686409)

&emsp;[卍卐没想到的帖子-标定
](https://blog.csdn.net/a083614/article/details/78579163)

&emsp;[瞻邈-立体校正](https://blog.csdn.net/xhtchina/article/details/126492195)

&emsp;In short, I won’t dive deep into the theory. Instead, I will briefly discuss the implementation through code.
&emsp;For the calibration part, part of the code is referenced from:[Temuge Batpurev's Blog](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html)
&emsp;For the code to draw epipolar lines, the reference is:[逆光525的帖子-绘制极线](https://blog.csdn.net/qq_22059843/article/details/103400094)
# 2. Single Camera Calibration
## 2.1 Let's Look at the Code

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

## 2.3 calibrateCamera Parameters

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

# 3. Stereo Calibration
## 3.1 Let's Look at the Code

```python
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
ret, mtx_main, dist_main, mtx_side, dist_side, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_main, imgpoints_side,
    mtx_main, dist_main, mtx_side, dist_side,
    frame_size, criteria=criteria, flags=flags)
```
## 3.2 stereoCalibrate Parameters

```python
    # Stereo calibration function
    # ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = 
    # cv2.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, 
    #                       cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
    #                       criteria=criteria, flags=flags)
    
    # Input parameters:
    # objectPoints: List of object points, similar to the objpoints in calibrateCamera.
    # imagePoints1: List of image points from the left camera.
    # imagePoints2: List of image points from the right camera.
    # cameraMatrix1: Initial intrinsic matrix for the left camera.
    # distCoeffs1: Initial distortion coefficients for the left camera.
    # cameraMatrix2: Initial intrinsic matrix for the right camera.
    # distCoeffs2: Initial distortion coefficients for the right camera.
    # imageSize: Size of the image, specified as (width, height).
    
    # criteria (optional): Termination criteria for the optimization algorithm.
        # cv2.TERM_CRITERIA_MAX_ITER: Stop when the maximum number of iterations is reached.
        # cv2.TERM_CRITERIA_EPS: Stop when the change in parameters is below a specified threshold.
        # These can be combined, such as cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, indicating stopping when either condition is met.
        # Max iterations: An integer specifying the maximum number of iterations, e.g., 100 means at most 100 iterations.
        # Precision threshold: A floating point number specifying the threshold for stopping when parameter changes are below this value, e.g., 1e-5 means stop when changes are less than 0.00001.
    
    # flags (optional): Flags to specify certain parameters as fixed or to define their degree of freedom.
        # cv2.CALIB_FIX_INTRINSIC: Keep the intrinsic parameters of both cameras fixed during calibration. This means that the intrinsic matrices (including focal length, principal point, etc.) will not be re-estimated during stereo calibration, using the results from single camera calibration instead.
        # cv2.CALIB_USE_INTRINSIC_GUESS: Use the provided intrinsic parameters as initial guesses and optimize them during calibration. This can improve calibration accuracy, especially if the initial intrinsic parameters are already quite accurate.
        # cv2.CALIB_FIX_PRINCIPAL_POINT: Keep the principal points (optical centers) fixed.
        # cv2.CALIB_FIX_FOCAL_LENGTH: Keep the focal lengths fixed.
        # cv2.CALIB_FIX_ASPECT_RATIO: Keep the aspect ratio of the focal lengths fixed.
        # cv2.CALIB_ZERO_TANGENT_DIST: Assume zero tangential distortion and keep it fixed.
        # cv2.CALIB_RATIONAL_MODEL: Enable the use of a rational model with 6 distortion coefficients in the calibration function.
        # cv2.CALIB_SAME_FOCAL_LENGTH: Assume the same focal length for both cameras.
    
    # Output parameters:
    # ret: The average re-projection error.
    # cameraMatrix1: The calibrated intrinsic matrix for the left camera.
    # distCoeffs1: The calibrated distortion coefficients for the left camera.
    # cameraMatrix2: The calibrated intrinsic matrix for the right camera.
    # distCoeffs2: The calibrated distortion coefficients for the right camera.
    # R: The rotation matrix that transforms the coordinate system of the right camera to the coordinate system of the left camera.
    # T: The translation vector that transforms the coordinate system of the right camera to the coordinate system of the left camera.
    # E: The essential matrix.
    # F: The fundamental matrix.
```

# 4. Stereo Rectification
## 4.1 Let's Look at the Code

```python
# Stereo Rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_main, dist_main, mtx_side, dist_side, frame_size, R, T)

# Function to rectify images
def rectify_image(img, mtx, dist, R, P):
    h, w = img.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, R, P, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# Read the test images
img_main = cv2.imread('demo/left03.jpg')
img_side = cv2.imread('demo/right03.jpg')

# Rectify the images
rectified_main = rectify_image(img_main, mtx_main, dist_main, R1, P1)
rectified_side = rectify_image(img_side, mtx_side, dist_side, R2, P2)

# Save the rectified images
cv2.imwrite('rectified_main.png', rectified_main)
cv2.imwrite('rectified_side.png', rectified_side)
```
## 4.2 Explanation
&emsp;There's not much to explain here. It's mainly about understanding the functions initUndistortRectifyMap and remap. Simply put, the first function generates a mapping to remove distortions from the images based on the camera's intrinsic parameters and the extrinsic parameters between the two cameras. The second function applies this mapping to obtain the rectified images. I'll explain the parameters of these functions below.

## 4.3 stereoRectify Parameters

```python
# R1, R2, P1, P2, Q, roi1, roi2 = 
# cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, 
#                   R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=(0, 0))
# Input Parameters:
# cameraMatrix1: Intrinsic parameters of the left camera.
# distCoeffs1: Distortion coefficients of the left camera.
# cameraMatrix2: Intrinsic parameters of the right camera.
# distCoeffs2: Distortion coefficients of the right camera.
# imageSize: Size of the image in (width, height) format.
# R: Rotation matrix to transform the right camera's coordinate system to the left camera's coordinate system.
# T: Translation vector to transform the right camera's coordinate system to the left camera's coordinate system.
# flags (optional): Rectification type flag.
# alpha (optional): Free scaling parameter between [0, 1], controlling the cropping of image edges.
# newImageSize (optional): Size of the new image.
# Output Parameters:
# R1: Rectification transform (rotation matrix) for the left camera.
# R2: Rectification transform (rotation matrix) for the right camera.
# P1: New projection matrix for the left camera.
# P2: New projection matrix for the right camera.
# Q: Disparity-to-depth mapping matrix.
# roi1: Valid pixel region of the left camera image.
# roi2: Valid pixel region of the right camera image.
```
## 4.4 initUndistortRectifyMap Parameters
```python
# Compute undistortion and rectification transformation map
# map1, map2 = 
# cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type)
# Input Parameters:
# cameraMatrix: Intrinsic parameters of the camera.
# distCoeffs: Distortion coefficients of the camera.
# R: Rectification transformation (rotation matrix).
# newCameraMatrix: New camera matrix.
# size: Size of the image in (width, height) format.
# m1type: Type of the first output map, can be cv2.CV_32FC1 or cv2.CV_16SC2 etc.
# Output Parameters:
# map1: The first output map for the remap function.
# map2: The second output map for the remap function.
```
## 4.5 remap Parameters

```python
# Remap function
# dst = cv2.remap(src, map1, map2, interpolation, borderMode, borderValue)
# Input Parameters:
# src: Input image.
# map1: The first map.
# map2: The second map.
# interpolation: Interpolation method, like cv2.INTER_LINEAR or cv2.INTER_CUBIC.
# borderMode (optional): Border mode to handle image boundaries, like cv2.BORDER_CONSTANT or cv2.BORDER_REPLICATE.
# borderValue (optional): Border value used if borderMode is cv2.BORDER_CONSTANT.
# Output Parameters:
# dst: The rectified image.
```

# 5. Drawing Epipolar Lines

## 5.1 Let's Look at the Code
```python
plt.figure(figsize=(20, 20))

for i in range(0,1):  # Taking the first pair of images as an example
    im_L=Image.fromarray(rectified_main) # Convert numpy array to Image class
    im_R=Image.fromarray(rectified_side) # Convert numpy array to Image class

    width = im_L.size[0]*2
    height = im_L.size[1]

    img_compare = Image.new('RGBA',(width, height))
    img_compare.paste(im_L,box=(0,0))
    img_compare.paste(im_R,box=(640,0))
    
# Draw evenly spaced horizontal lines on the rectified images
    for i in range(1,20):
        len=480/20
        plt.axhline(y=i*len, color='r', linestyle='-')
    plt.imshow(img_compare)
    plt.savefig('epipolar_lines.png', bbox_inches='tight', pad_inches=0)
    plt.show()
```
## 5.2 Explanation
&emsp;Note that, to avoid complexity, I didn't calculate the actual epipolar lines but instead drew horizontal lines as epipolar lines and visually checked if points on the same horizontal line are corresponding points. For calibration images, you can also use the corner points as correspondences and draw the epipolar lines accordingly. However, for non-calibration images, you might need to use SIFT keypoint matching. For more details, you can refer to the [OpenCV official tutorial](https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html)

## 5.3 Rectification Result
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14e579a7727f4ce1a6f2198bfa9bef75.png)
&emsp;OK，it matches the expectation quite well!

# 6. What's more
&emsp;When there is a large angle between cameras, the process of stereo calibration and rectification might fail. Therefore, in the next post, we will look at how to use uncalibrated rectification functions: uncalibrated rectification as in stereo_rectify_uncalibrated.md.
