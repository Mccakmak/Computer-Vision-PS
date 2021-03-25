#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import cv2
import random


# In[2]:


img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')

cv2.imwrite("output/Original_image1.png",img1)
cv2.imwrite("output/Original_image2.png",img2)


# In[3]:


img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

cv2.imwrite("output/Gray_image1.png",img1_gray)
cv2.imwrite("output/Gray_image2.png",img2_gray)


# In[4]:


sift = cv2.xfeatures2d.SIFT_create()
keypoints1_sift, descriptors1_sift= sift.detectAndCompute(img1_gray,None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2_gray,None)

surf = cv2.xfeatures2d.SURF_create(4000)
keypoints1_surf, descriptors1_surf = surf.detectAndCompute(img1_gray,None)
keypoints2_surf, descriptors2_surf = surf.detectAndCompute(img2_gray,None)

orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(img1_gray,None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(img2_gray,None)     


# In[5]:


color1 = (np.random.choice(range(256), size=3))

img1_key_sift = np.zeros((img1.shape))
img2_key_sift = np.zeros((img2.shape))
img1_key_surf= np.zeros((img1.shape))
img2_key_surf = np.zeros((img2.shape))
img1_key_orb = np.zeros((img1.shape))
img2_key_orb = np.zeros((img2.shape))

img1_key_sift=cv2.drawKeypoints(image=img1, outImage=img1_key_sift, keypoints=keypoints1_sift, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (int(color1[0]), int(color1[1]), int(color1[2])))
img2_key_sift=cv2.drawKeypoints(image=img2, outImage=img2_key_sift, keypoints=keypoints2_sift, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (int(color1[0]), int(color1[1]), int(color1[2])))
img1_key_surf=cv2.drawKeypoints(image=img1, outImage=img1_key_surf, keypoints=keypoints1_surf, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (int(color1[0]), int(color1[1]), int(color1[2])))
img2_key_surf=cv2.drawKeypoints(image=img2, outImage=img2_key_surf, keypoints=keypoints2_surf, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (int(color1[0]), int(color1[1]), int(color1[2])))
img1_key_orb=cv2.drawKeypoints(image=img1, outImage=img1_key_orb, keypoints=keypoints1_orb, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (int(color1[0]), int(color1[1]), int(color1[2])))
img2_key_orb=cv2.drawKeypoints(image=img2, outImage=img2_key_orb, keypoints=keypoints2_orb, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (int(color1[0]), int(color1[1]), int(color1[2])))

cv2.imwrite("output/Key1_sift.png",img1_key_sift)
cv2.imwrite("output/Key2_sift.png",img2_key_sift)
cv2.imwrite("output/Key1_surf.png",img1_key_surf)
cv2.imwrite("output/Key2_surf.png",img2_key_surf)
cv2.imwrite("output/Key1_orb.png",img1_key_orb)
cv2.imwrite("output/Key2_orb.png",img2_key_orb)


# In[6]:


bf = cv2.BFMatcher()
matches_sift = bf.knnMatch(descriptors1_sift,descriptors2_sift,k=2)
matches_surf = bf.knnMatch(descriptors1_surf,descriptors2_surf,k=2)
matches_orb = bf.knnMatch(descriptors1_orb,descriptors2_orb,k=2)

print("Sift Matches:",len(matches_sift))
print("Surf Matches:",len(matches_surf))
print("Orb Matches:",len(matches_orb))

bestMatches_sift = []
bestMatches_surf = []
bestMatches_orb = []
for m,n in matches_sift:
    if m.distance < 0.5*n.distance:
        bestMatches_sift.append([m])
for m,n in matches_surf:
    if m.distance < 0.5*n.distance:
        bestMatches_surf.append([m])
for m,n in matches_orb:
    if m.distance < 0.75*n.distance:
        bestMatches_orb.append([m])
        
print("Best Sift Matches",len(bestMatches_sift))
print("Best Surf Matches",len(bestMatches_surf))
print("Best Orb Matches",len(bestMatches_orb))

img_matches_sift = cv2.drawMatchesKnn(img1,keypoints1_sift,img2,keypoints2_sift,bestMatches_sift,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_surf = cv2.drawMatchesKnn(img1,keypoints1_surf,img2,keypoints2_surf,bestMatches_surf,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_orb = cv2.drawMatchesKnn(img1,keypoints1_orb,img2,keypoints2_orb,bestMatches_orb,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("output/SiftMatches.png",img_matches_sift)
cv2.imwrite("output/SurfMatches.png",img_matches_surf)
cv2.imwrite("output/OrbMatches.png",img_matches_orb)


# In[7]:


h_mat=[]
f=open("input/H1to2p", "r")
for line in f:
    h_mat.append([float(l) for l in line.strip().split()])
h_mat = np.asarray(h_mat)


# In[8]:


keypoints_sift = [keypoints1_sift,keypoints2_sift]
correct_match_sift=0
for i in range(len(matches_sift)):
    p=np.append(keypoints_sift[1][matches_sift[i][0].trainIdx].pt,1)
    m=np.append(keypoints_sift[0][matches_sift[i][0].queryIdx].pt,1)
    n=np.matmul(h_mat,m)
    n=n/n[2]
    if(abs(n[0]-p[0]) <=2 and abs(n[1]-p[1]) <=2):
        correct_match_sift=correct_match_sift + 1
        
print("Correct Match Sift:",correct_match_sift)

keypoints_surf = [keypoints1_surf,keypoints2_surf]
correct_match_surf=0
for i in range(len(matches_surf)):
    p=np.append(keypoints_surf[1][matches_surf[i][0].trainIdx].pt,1)
    m=np.append(keypoints_surf[0][matches_surf[i][0].queryIdx].pt,1)
    n=np.matmul(h_mat,m)
    n=n/n[2]
    if(abs(n[0]-p[0]) <=2 and abs(n[1]-p[1]) <=2):
        correct_match_surf=correct_match_surf + 1
        
print("Correct Match Surf:",correct_match_surf)

keypoints_orb = [keypoints1_orb,keypoints2_orb]
correct_match_orb=0
for i in range(len(matches_orb)):
    p=np.append(keypoints_orb[1][matches_orb[i][0].trainIdx].pt,1)
    m=np.append(keypoints_orb[0][matches_orb[i][0].queryIdx].pt,1)
    n=np.matmul(h_mat,m)
    n=n/n[2]
    if(abs(n[0]-p[0]) <=2 and abs(n[1]-p[1]) <=2):
        correct_match_orb=correct_match_orb + 1
        
print("Correct Match Orb:",correct_match_orb)

avg_sift=(len(keypoints1_sift)+len(keypoints2_sift))/2
repeatability_sift=correct_match_sift/avg_sift
avg_surf=(len(keypoints1_surf)+len(keypoints2_surf))/2
repeatability_surf=correct_match_surf/avg_surf
avg_orb=(len(keypoints1_orb)+len(keypoints2_orb))/2
repeatability_orb=correct_match_orb/avg_orb

print("Sift Repeatability:",repeatability_sift)
print("Surf Repeatability:",repeatability_surf)
print("Orb Repeatability:",repeatability_orb)


# In[9]:


def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(len(corr)):
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))
        
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print("Correspondence size: ", len(corr), " Number of inliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers


# In[10]:


def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


# In[11]:


def calculateHomography(correspondences):
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)
    matrixA = np.matrix(aList)
    u, s, v = np.linalg.svd(matrixA)
    h = np.reshape(v[8], (3, 3))
    h = (1/h.item(8)) * h
    return h


# In[12]:


correspondenceList = []
for match in bestMatches_sift:
        (x1, y1) = keypoints_sift[0][match[0].queryIdx].pt
        (x2, y2) = keypoints_sift[1][match[0].trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])
corrs = np.matrix(correspondenceList)


# In[13]:


estimation_thresh = 0.9
finalH, inliers = ransac(corrs, estimation_thresh)
print("Final homography: ", finalH)


# In[14]:


print("Given homography matrix:")
print(h_mat)
print("Ransac homography matrix:")
print(finalH)


# In[15]:


img_warp = cv2.warpPerspective(img1, h_mat, (img2.shape[1],img2.shape[0]))
img_warp_ransac = cv2.warpPerspective(img1, finalH, (img2.shape[1],img2.shape[0]))


# In[19]:


img_warp_ransac_gray=cv2.cvtColor(img_warp_ransac,cv2.COLOR_BGR2GRAY)
img_warp_gray=cv2.cvtColor(img_warp,cv2.COLOR_BGR2GRAY)
diff_img_gray = img_warp_ransac_gray - img_warp_gray

cv2.imwrite("output/Difference.png",diff_img_gray)

