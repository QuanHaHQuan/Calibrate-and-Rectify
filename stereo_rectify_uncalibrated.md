<h1 align="center">Uncalibrated Rectification in OpenCV (stereoRectifyUncalibrated)</h1>

# 1. Introduction  
&emsp;In previous posts, we discussed some conventional calibration and rectification methods Camera Calibration, Stereo Calibration, and Stereo Rectification in OpenCV [(calibrateCamera, stereoCalibrate, and stereoRectify)](https://blog.csdn.net/c_h_q_/article/details/140413330?spm=1001.2014.3001.5501)，However, if there is a large angle between the two cameras, the stereo calibration and rectification steps might not be suitable, as referenced in this post [When Stereo Rectification Fails. ](https://stackoverflow.com/questions/57039780/opencv-stereo-image-rectification-doesnt-work-properly-when-the-angle-between-c/57040516#57040516)，To solve this issue, today we will discuss how to use uncalibrated rectification functions:

&emsp;By the way, the data and complete code I used are here: Repo: Calibrate-and-Rectify：[Repo : Calibrate-and-Rectify](https://github.com/QuanHaHQuan/Calibrate-and-Rectify/blob/main/readme.md)
# 2. Uncalibrated Rectification
## 2.1 Let's Look at the Code

```python
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

c1_images_names = sorted(glob.glob('demo/left*.jpg'))
c2_images_names = sorted(glob.glob('demo/right*.jpg'))
 
c1_images = []
c2_images = []
for im1, im2 in zip(c1_images_names, c2_images_names):
    _im = cv2.imread(im1, 1)
    c1_images.append(_im)
 
    _im = cv2.imread(im2, 1)
    c2_images.append(_im)

def find_corners(image_names):
    images = []
    for imname in image_names:
        im = cv2.imread(imname, 1)
        images.append(im)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 6 #number of checkerboard rows.
    columns = 9 #number of checkerboard columns.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
 
        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            imgpoints.append(corners)
    return imgpoints
 
imgpoints1 = find_corners(c1_images_names)
imgpoints2 = find_corners(c2_images_names)

# Convert corner coordinates to float32 format
points1all = np.array([pt.ravel() for pts in imgpoints1 for pt in pts], dtype=np.float32)
points2all = np.array([pt.ravel() for pts in imgpoints2 for pt in pts], dtype=np.float32)

# Calculate the fundamental matrix
F, mask = cv2.findFundamentalMat(points1all, points2all, cv2.FM_RANSAC)

# Image size
img_size = (640, 480)
   
img1 = cv2.imread('demo/left03.jpg', 1)
img2 = cv2.imread('demo/right03.jpg', 1)

img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

gray_left = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
# Find checkerboard corners
ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 9), None)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 9), None)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(ret_left, ret_right)
    
if ret_left and ret_right:
    # Refine corner coordinates
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

# Convert corner coordinates to float32 format
points1 = np.array([pt[0] for pt in corners_left], dtype=np.float32).reshape(-1, 2)
points2 = np.array([pt[0] for pt in corners_right], dtype=np.float32).reshape(-1, 2)

# Perform stereo rectification
retval, H1, H2 = cv2.stereoRectifyUncalibrated(points1, points2, F, img_size)

# Crop images to remove black borders
def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

if retval:
    # Compute rectification mappings
    img1_rectified = cv2.warpPerspective(img1, H1, img_size)
    img2_rectified = cv2.warpPerspective(img2, H2, img_size)

    # Display rectified images
    cv2.imwrite('Rectified Image 1.png', img1_rectified)
    cv2.imwrite('Rectified Image 2.png', img2_rectified)
else:
    print("Stereo rectification failed")
    
plt.figure(figsize=(20, 20))

for i in range(0,1):  # Using the first pair of images as an example
    im_L=Image.fromarray(img1_rectified) # numpy to Image class
    im_R=Image.fromarray(img2_rectified) # numpy to Image class

    width = im_L.size[0]*2
    height = im_L.size[1]

    img_compare = Image.new('RGBA',(width, height))
    img_compare.paste(im_L,box=(0,0))
    img_compare.paste(im_R,box=(640,0))
    
    # Draw evenly spaced lines on the epipolar aligned images
    for i in range(1,20):
        len=480/20
        plt.axhline(y=i*len, color='r', linestyle='-')
    plt.imshow(img_compare)
    plt.savefig('epipolar_lines_ro.png', bbox_inches='tight', pad_inches=0)
    plt.show()
```
## 2.2 Some Explanation
&emsp;Actually, there's not much to explain. The basic parameters are already described in the previous post. The main idea is that since these are calibration images, we use the corners as key points, then calculate the fundamental matrix using the matched key points and findFundamentalMat, which describes the mapping relationship between the two images. Finally, apply this mapping to the sample images using the stereoRectifyUncalibrated function.
## 2.3 findFundamentalMat Parameters
&emsp;findFundamentalMat is a function in OpenCV used to compute the fundamental matrix. The fundamental matrix describes the point correspondences between two images and is a key concept in stereo vision calculations. Below is a detailed explanation of this function:

```python
cv2.findFundamentalMat(points1, points2[, method[, ransacReprojThreshold[, confidence[, maxIters]]]]) -> fundamentalMatrix, mask

Input Parameters
points1:
Description: The list or array of detected feature points in the first image. It should be an Nx2 array of 2D point coordinates, where N is the number of feature points.

points2:
Description: The list or array of detected feature points in the second image. It should correspond one-to-one with points1 and have the same format.

method (optional):
Description: The method used to estimate the fundamental matrix. It can be one of the following:
cv2.FM_7POINT: Seven-point algorithm. Requires exactly seven points.
cv2.FM_8POINT: Eight-point algorithm. Requires at least eight points.
cv2.FM_RANSAC: Uses the RANSAC algorithm. Recommended for dealing with noisy data.
cv2.FM_LMEDS: Uses the Least-Median-of-Squares algorithm. Also used for dealing with noisy data.

ransacReprojThreshold (optional):
Description: The maximum allowed reprojection error for the RANSAC algorithm. Only effective when using the RANSAC method. The default value is 3.

confidence (optional):
Description: The confidence level for the RANSAC algorithm. Only effective when using the RANSAC method. The default value is 0.99.

maxIters (optional):
Description: The maximum number of iterations for the RANSAC algorithm. Only effective when using the RANSAC method. The default value is 1000.

Output Parameters
fundamentalMatrix:
Description: The computed fundamental matrix, which is a 3x3 matrix.

mask:
Description: The output mask identifying inliers (valid points). It is an Nx1 binary array, where N is the number of input feature points. Elements with a value of 1 indicate inliers, and elements with a value of 0 indicate outliers.

```

## 2.4 stereoRectifyUncalibrated Parameters

```python
cv2.stereoRectifyUncalibrated(points1, points2, F, imgSize[, H1[, H2[, threshold]]]) -> retval, H1, H2

Input Parameters
points1:
Description: The list or array of detected feature points in the left image. It should be an Nx2 array of 2D point coordinates, where N is the number of feature points.

points2:
Description: The list or array of detected feature points in the right image. It should correspond one-to-one with points1 and have the same format.

F:
Description: The fundamental matrix (3x3) that describes the mapping between the stereo images. This matrix is estimated from matched feature points.

imgSize:
Description: The size of the image as a tuple (width, height).

H1 (optional):
Description: The output parameter to store the 3x3 transformation matrix for the left image. If provided, the function writes the result into this parameter.

H2 (optional):
Description: The output parameter to store the 3x3 transformation matrix for the right image. If provided, the function writes the result into this parameter.

threshold (optional):
Description: The threshold value used for the RANSAC algorithm. This threshold determines the tolerance for considering a point correspondence as inlier. If not provided, the default value is used.

Output Parameters
retval:
Description: A flag indicating whether the function executed successfully. Returns True if successful and False otherwise.

H1:
Description: The 3x3 transformation matrix for the left image. This is only returned if the corresponding input parameter is provided and the function is successful.

H2:
Description: The 3x3 transformation matrix for the right image. This is only returned if the corresponding input parameter is provided and the function is successful.
```

# 3. 矫正结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b1987d524a6f4d7491753ededa797d5a.png)
&emsp;OK, the results are as expected~
