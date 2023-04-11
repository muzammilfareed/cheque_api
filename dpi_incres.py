import subprocess
import cv2
import numpy as np



def preprocess_for_image(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, img) = cv2.threshold(grayImage, 137, 255, cv2.THRESH_BINARY)
    return img


img= cv2.imread('img1476.jpg')
fresh_img = img.shape
print(fresh_img[:2])
shap_img = fresh_img[:2]
img = preprocess_for_image(img)
img = cv2.resize(img,shap_img,interpolation=cv2.INTER_AREA)

# img_binary = cv2.resize(img, (1050, 610))
cv2.imshow('asd', img)
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (10,10,500,500)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
lower_white = np.array([0, 0, 0], dtype=np.uint8)
upper_white = np.array([0,0,0], dtype=np.uint8)
mask = cv2.inRange(img, lower_white, upper_white) # could also use threshold
res = cv2.bitwise_not(img, img, mask)

img_binary = cv2.resize(img, (1050, 610))
#save image
# cv2.imwrite('D:/black-and-white.png',img_binary)
cv2.imshow('asd', img_binary)
cv2.waitKey(0)
# cv2.imwrite('kitchen_processed.png',img)
subprocess.run('convert -density 75 -units pixelsperinch kitchen_processed.png outfile.png')