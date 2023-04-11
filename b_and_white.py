import cv2

# img = cv2.imread('img1436.jpg')


#read image
img_grey = cv2.imread('static/iterations/success.jpg', cv2.IMREAD_GRAYSCALE)

# define a threshold, 128 is the middle of black and white in grey scale
thresh = 118

# threshold the image
img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]
img_binary = cv2.resize(img_binary, (1050, 610))
#save image
# cv2.imwrite('D:/black-and-white.png',img_binary)
cv2.imshow('asd', img_binary)
cv2.waitKey(0)