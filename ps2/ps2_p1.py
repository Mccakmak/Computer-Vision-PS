import numpy as np
import cv2 as cv
import glob

#stop the iteration when any of the below condition is met. epsilon = 30 accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((12*12,3), np.float32)
objp[:,:2] = np.mgrid[0:12,0:12].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob("input/*.tif")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (12,12), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (12,12), corners2, ret)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("******Distortion Matrix******")
print(dist)


for i in range(0,len(images)):
    img = cv.imread(images[i], cv.IMREAD_GRAYSCALE)
    h,w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    string="image-"+str(i+1)+".png"
    cv.imwrite("output/"+string,dst)
    
    
print("******Instrict Matrix******")
print(mtx)

projection_mat = []
rot_matrix = []
for i in range(0,len(rvecs)):
    rotation_matrix = np.zeros(shape=(3, 3))
    cv.Rodrigues(rvecs[i], rotation_matrix)[0]
    rot_matrix.append(rotation_matrix)
    projection_mat.append(np.column_stack((np.matmul(mtx,rotation_matrix), tvecs[i])))
    
print("******Projection Matrix******")
print(projection_mat)

essential_matrix = []
for i in range(0,len(tvecs)):
    tran_cross_matrix=np.array([[0,-tvecs[i][2][0],tvecs[i][1][0]],[tvecs[i][2][0],0,-tvecs[i][0][0]],[-tvecs[i][1][0],tvecs[i][0][0],0]])
    essential_matrix.append(np.matmul(tran_cross_matrix,rot_matrix[i]))
    
    
print("******Essential Matrix******")    
print(essential_matrix)


R1, R2, T = cv.decomposeEssentialMat(essential_matrix[0])

print('Rotation 1:')
print(R1)
print('Rotation 2:')
print(R2)
print('Translation:')
print(T)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c= img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)  
        img1 = cv.circle(img1,tuple(pt1[0]),0,color,-1)
        img2 = cv.circle(img2,tuple(pt2[0]),0,color,-1)
    return img1,img2

pts1 = imgpoints[7]
pts2 = imgpoints[8]
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)

img1 = cv.imread(images[7], cv.IMREAD_GRAYSCALE)
img2 = cv.imread(images[8], cv.IMREAD_GRAYSCALE)

img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

cv.imwrite("output/epipolar-line1.png",img5)
cv.imwrite("output/epipolar-line2.png",img3)