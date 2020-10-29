###################code part one ############
################# capturing and editing the image to start detecting the diffrence ####

from skimage.measure import compare_ssim
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

#####To take Photo in the same size
def findArea (image) :
    # image = cv2.imread('2.jpg')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([97, 33, 127], dtype="uint8")
    upper = np.array([179, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    k=0
    arrayArea = []
    for c in cnts:
        area = cv2.contourArea(c)
        cv2.drawContours(original,[c], 0, (0,0,0), 2)
        arrayArea.append(area)
        k += 1
    if k == 0:
        return 0,original
    else:
        return np.amax(arrayArea) ,original


def Histgram(img_color):
    labImage = cv2.cvtColor(img_color,cv2.COLOR_BGR2Lab)
    l,a,b = cv2.split(labImage)
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    clahe_img = clahe.apply(l)
    Uclahe_img =cv2.merge([clahe_img,a,b])
    Uclahe_img = cv2.cvtColor(Uclahe_img,cv2.COLOR_LAB2BGR)
    cv2.imshow("clahe",Uclahe_img)
    cv2.waitKey(0)
    img_color = Uclahe_img
    img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
    return img_color,img_gray



MIN_MATCH_COUNT = 10
img1_color = cv2.imread('before.png')       # queryImage
img2_color = cv2.imread('after.png')     # trainImage
train_area ,trainMask= findArea(img2_color)
xr , yr ,cr=img2_color.shape
cv2.imshow("Mask of train",trainMask)
###### Using Histogram to removr noise

img2_color,img2_gray = Histgram(img2_color)

cap = cv2.VideoCapture(0)
cap.set(3,yr)
cap.set(4,xr)
cap.set(10,130)

while 1:
    _ , frame_color = cap.read()
    if _ == False :
        print("There are no Video")
        break
    frame_gray = cv2.cvtColor(frame_color,cv2.COLOR_BGR2GRAY)
    area,mask = findArea(frame_color)
    cv2.imshow ("Live video",frame_color)
    if area >= train_area-500 and area <= train_area+500 :
        img1_color = frame_color
        img1_gray = frame_gray
        mask_image = mask
        break
    k = cv2.waitKey(3)
    if k == 27:
        break

img1_color,img1_gray = Histgram(img1_color)

cv2.imshow("mask of frame",mask_image)
cv2.waitKey(0)
cap.release()
###############___________________________________________________________________
##### match image and Reproject it using Homograph


orb = cv2.ORB_create(70)  #Registration works with at least 150 points

kp1, des1 = orb.detectAndCompute(img1_gray, None)  #kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(img2_gray, None)


matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(img1_color,kp1, img2_color, kp2, matches[:20], None)

cv2.imshow("Matches image", img3)
cv2.waitKey(0)

points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors


matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

####### Use homography
height, width, channels = img2_color.shape
pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)

#######  Reproject
im1Reg = cv2.warpPerspective(img1_color, matrix, (width, height))  #Applies a perspective transformation to an image.
cv2.imshow("Mask ", mask)
cv2.imshow("Registered image", im1Reg)
cv2.waitKey(0)

im1Reg_gray = cv2.cvtColor(im1Reg,cv2.COLOR_BGR2GRAY)
print("img1Regg=",im1Reg.shape)



(score, diff) = compare_ssim(img2_gray, im1Reg_gray, full=True)
print("Image similarity:", score)
if score >= 0.98:
    print ('there are no different')
# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] image1 we can use it with OpenCV
else :
    diff = (diff * 255).astype("uint8")

    # diff_gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
    thresh = cv2.threshold(diff, 0 , 100,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    # loop over the contours
    for cnts in contours:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        area = cv2.contourArea(cnts)
        # if area <= w*h+10 and area >= w*h-10 :
        #     continue
        if area >500 and area <1200 :
            (x, y, w, h) = cv2.boundingRect(cnts)
            cv2.rectangle(img2_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(im1Reg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #  cv.su
    # show the output imagescv.imshow("Original", img1_color)


    cv2.imshow("Original", img2_color)
    cv2.imshow("Modified  ", im1Reg)
    cv2.imshow("Diff", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

