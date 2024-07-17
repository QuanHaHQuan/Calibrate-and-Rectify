<h1 align = "center">OpenCV下的无标定校正(stereoRectifyUncalibrated)</h1>

# 1. 杂话   
&emsp;咱们在之前的帖子里面讲了一些比较常规的标定和校正[OpenCV下的单目标定，双目标定与立体校正(calibrateCamera, stereoCalibrate and stereoRectify)](https://blog.csdn.net/c_h_q_/article/details/140413330?spm=1001.2014.3001.5501)，但是，如果说两个相机之间的转角比较大，那么可能并不适合用双目标定＋立体校正的步骤来做，参考这个贴子[立体校正失效的情况](https://stackoverflow.com/questions/57039780/opencv-stereo-image-rectification-doesnt-work-properly-when-the-angle-between-c/57040516#57040516)，所以为了解决这个问题，今天咱们就来说说如何使用非标定的校正函数：

&emsp;对了，我使用的数据和完整的代码在这里：[Repo : Calibrate-and-Rectify
](https://github.com/QuanHaHQuan/Calibrate-and-Rectify/blob/main/readme.md)
# 2. 无标定校正
## 2.1 先看代码

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

# 将角点坐标转换为float32格式
points1all = np.array([pt.ravel() for pts in imgpoints1 for pt in pts], dtype=np.float32)
points2all = np.array([pt.ravel() for pts in imgpoints2 for pt in pts], dtype=np.float32)

# 计算基础矩阵
F, mask = cv2.findFundamentalMat(points1all, points2all, cv2.FM_RANSAC)

# 图像大小
img_size = (640, 480)
   
img1 = cv2.imread('demo/left03.jpg', 1)
img2 = cv2.imread('demo/right03.jpg', 1)

img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

gray_left = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
# 找到棋盘格角点
ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 9), None)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 9), None)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(ret_left, ret_right)
    
if ret_left and ret_right:
    # 提高角点的精度
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

# 将角点坐标转换为float32格式
points1 = np.array([pt[0] for pt in corners_left], dtype=np.float32).reshape(-1, 2)
points2 = np.array([pt[0] for pt in corners_right], dtype=np.float32).reshape(-1, 2)

# 进行立体校正
retval, H1, H2 = cv2.stereoRectifyUncalibrated(points1, points2, F, img_size)

# 裁剪图像去除黑边
def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

if retval:
    # 计算校正映射
    img1_rectified = cv2.warpPerspective(img1, H1, img_size)
    img2_rectified = cv2.warpPerspective(img2, H2, img_size)

    # 显示校正后的图像
    cv2.imwrite('Rectified Image 1.png', img1_rectified)
    cv2.imwrite('Rectified Image 2.png', img2_rectified)
else:
    print("立体校正失败")
    
plt.figure(figsize=(20, 20))

for i in range(0,1):  # 以第一对图片为例
    im_L=Image.fromarray(img1_rectified) # numpy 转 image类
    im_R=Image.fromarray(img2_rectified) # numpy 转 image 类

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
    plt.savefig('epipolar_lines_ro.png', bbox_inches='tight', pad_inches=0)
    plt.show()
```
## 2.2 一点解释
&emsp;其实也没啥好解释的吧，一些基础的参数都在上面说的另外一篇帖子里面写好了。主要就是说一下思路，因为是标定图片，所以还是用角点做关键点，之后通过匹配的关键点和findFundamentalMat去计算基础矩阵，描述两个图片的映射关系。然后针对示例图片，应用这个映射，也就是stereoRectifyUncalibrated函数。
## 2.3 findFundamentalMat参数
&emsp;findFundamentalMat 是 OpenCV 中用于计算基础矩阵的函数。基础矩阵描述了两个图像之间的点对应关系，是立体视觉计算中的一个重要概念。以下是该函数的详细解释：

```python
cv2.findFundamentalMat(points1, points2[, method[, ransacReprojThreshold[, confidence[, maxIters]]]]) -> fundamentalMatrix, mask

输入参数
points1:
描述：在第一幅图像中检测到的特征点列表或数组。应该是 Nx2 的二维点坐标列表，其中 N 是特征点的数量。

points2:
描述：在第二幅图像中检测到的特征点列表或数组。应该与 points1 中的点一一对应，且格式相同。

method (可选):
描述：用于估算基础矩阵的方法。可以是以下之一：
cv2.FM_7POINT: 七点算法。要求点数量为 7。
cv2.FM_8POINT: 八点算法。要求点数量大于或等于 8。
cv2.FM_RANSAC: 使用 RANSAC 方法。推荐用于处理带有噪声的点。
cv2.FM_LMEDS: 使用最小中值平方算法。也用于处理带有噪声的点。

ransacReprojThreshold (可选):
描述：RANSAC 算法中的最大重投影误差阈值，仅在使用 RANSAC 方法时有效。默认值为 3。

confidence (可选):
描述：RANSAC 算法中的置信度，仅在使用 RANSAC 方法时有效。默认值为 0.99。

maxIters (可选):
描述：RANSAC 算法的最大迭代次数，仅在使用 RANSAC 方法时有效。默认值为 1000。


输出参数
fundamentalMatrix:
描述：计算得到的基础矩阵，大小为 3x3。

mask:
描述：输出参数，用于标识内点（inliers）。大小为 Nx1 的二值数组，其中 N 是输入特征点的数量。值为 1 的元素表示对应的特征点是内点，值为 0 的元素表示对应的特征点是外点（outliers）。
```

## 2.4 stereoRectifyUncalibrated参数

```python
cv2.stereoRectifyUncalibrated(points1, points2, F, imgSize[, H1[, H2[, threshold]]]) -> retval, H1, H2
输入参数

points1:
描述：在左图中检测到的特征点列表或数组。应该是 Nx2 的二维点坐标列表，其中 N 是特征点的数量。

points2:
描述：在右图中检测到的特征点列表或数组。应该与 points1 中的点一一对应，且格式相同。

F:
描述：基础矩阵（3x3），它是在立体图像对之间通过特征点匹配估算得到的。

imgSize:
描述：图像的尺寸，格式为 (width, height)，即图像的宽度和高度。

H1 (可选):
描述：输出参数，用于存储左图的 3x3 变换矩阵。如果提供了此参数，函数会将结果写入其中。

H2 (可选):
描述：输出参数，用于存储右图的 3x3 变换矩阵。如果提供了此参数，函数会将结果写入其中。

threshold (可选):
描述：用于 RANSAC 算法的阈值。RANSAC 算法用于估算变换矩阵时的容错范围。如果不提供，则使用默认值。


输出参数
retval:
描述：函数执行成功与否的标志。返回 True 表示成功，返回 False 表示失败。

H1:
描述：左图的 3x3 变换矩阵。只有在调用时提供了相应的参数，且函数执行成功时才会返回。

H2:
描述：右图的 3x3 变换矩阵。只有在调用时提供了相应的参数，且函数执行成功时才会返回。
```

# 3. 矫正结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b1987d524a6f4d7491753ededa797d5a.png)
&emsp;OK，基本符合预期哦~
