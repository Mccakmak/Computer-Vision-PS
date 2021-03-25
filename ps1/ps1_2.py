import cv2
import time
import numpy as np

cap=cv2.VideoCapture(0) 
r,frame=cap.read() 
num_frames = 1
filename="output\\video.avi"
codec = cv2.VideoWriter_fourcc(*"XVID")
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(filename,codec,10,(w*2,h),0) 
start_vid=time.time()
end=0
while r:
    start = time.time()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_sobel_x=cv2.Sobel(gray_frame,-1,1,0)/255 
    frame_sobel_y=cv2.Sobel(gray_frame,-1,0,1)/255 
    gradient_frame=np.uint8((frame_sobel_x*frame_sobel_x+frame_sobel_y*frame_sobel_y)**(0.5)*255)
    if cv2.waitKey(40) & 0xFF ==ord('q'):
        r=False
    elif end-start_vid >= 15:
        r=False
    else:
        end = time.time()
        seconds = (end - start)
        fps=num_frames/seconds
        fps_gray = cv2.putText(gray_frame, str(format(fps,'.2f')), (5,30), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.7, (255, 0, 0), 1, cv2.LINE_AA)
        fps_grad = cv2.putText(gradient_frame, str(format(fps,'.2f')), (5,30), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.7, (255, 0, 0), 1, cv2.LINE_AA)
        vid=np.concatenate((fps_gray, fps_grad), axis=1)
        cv2.imshow('Frame',vid)
        out.write(vid)
        r,frame=cap.read() #else, read new frame
cv2.destroyAllWindows()
cap.release()
out.release()