import cv2
import time
import numpy as np

img=cv2.imread("input\\landscape.png")
width=img.shape[0]
height=img.shape[1]
halfw=int(width/2)
halfh=int(height/2)

crophalf_img = img[int((width-halfw)/2):int((width+halfw)/2),int((height-halfh)/2):int((height+halfh)/2)].copy()
cv2.imwrite("output\\cropped_img.png",crophalf_img)

exc_img=img.copy()
blue_img = img[:,:,0].copy()
red_img = img[:,:,2].copy()
exc_img[:,:,0] = red_img
exc_img[:,:,2] = blue_img
cv2.imwrite("output\\exchanged_img.png",exc_img)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output\\grayed_img.png",gray_image)

rep_img = img.copy()
crop100_img=rep_img[int((width-100)/2):int((width+100)/2),int((height-100)/2):int((height+100)/2)]


crop_gray_img=gray_image[int((width-100)/2):int((width+100)/2),int((height-100)/2):int((height+100)/2)]


crop100_img[:,:,0]=crop_gray_img.copy()
crop100_img[:,:,1]=crop_gray_img.copy()
crop100_img[:,:,2]=crop_gray_img.copy()

rep_img[int((width-100)/2):int((width+100)/2),int((height-100)/2):int((height+100)/2)]=crop100_img
cv2.imwrite("output\\replaced_img.png",rep_img)

rows,cols = img.shape[:2]
M = np.float32([[1,0,2],[0,1,0]])
shifted_img = cv2.warpAffine(img,M,(cols,rows))


dif_img=abs(shifted_img/255-img/255)
cv2.imwrite("output\\difference_img.png",dif_img*255)

img_sobel_x=cv2.Sobel(gray_image,-1,1,0)/255 
img_sobel_y=cv2.Sobel(gray_image,-1,0,1)/255 
cv2.imwrite("output\\sobelx_img.png",img_sobel_x*255)
cv2.imwrite("output\\sobely_img.png",img_sobel_y*255)
gradient_mag=np.uint8((img_sobel_x*img_sobel_x+img_sobel_y*img_sobel_y)**(0.5)*255)
cv2.imwrite("output\\gradient_magnitude_img.png",gradient_mag)


blur = cv2.GaussianBlur(img,(3,3),0)
lap_img = cv2.Laplacian(blur,cv2.CV_64F)
lap_img = lap_img/lap_img.max()
cv2.imwrite("output\\laplacian_img.png",lap_img*255)
