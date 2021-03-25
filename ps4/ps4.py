import cv2
import numpy as np

img1 = cv2.imread('input/scene2.png')
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('input/scene1.png')
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_gray,None)
kp2, des2 = sift.detectAndCompute(img2_gray,None)

img1_key = np.zeros((img1.shape))
img1_key=cv2.drawKeypoints(image=img1, outImage=img1_key, keypoints=kp1, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (0, 200, 100))

cv2.imwrite("output/key1.png",img1_key)

img2_key = np.zeros((img2.shape))
img2_key=cv2.drawKeypoints(image=img2, outImage=img2_key, keypoints=kp2, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (0, 200, 100))

cv2.imwrite("output/key2.png",img2_key)

match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.25*n.distance:
        good.append(m)

draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

img_matches = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv2.imwrite("output/matches.png",img_matches)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h,w = img1_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2_gray = cv2.polylines(img2_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    cv2.imwrite("output/common.png",img2_gray)
else:
    print("Not enough matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
dst[0:img2.shape[0],0:img2.shape[1]] = img2

cv2.imwrite("output/before_trim.png",dst)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

cv2.imwrite("output/stitched.png",trim(dst))





