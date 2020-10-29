######### code part 2 ##########
######### getting the images with the spicified colors(pink-white)######

import cv2
import numpy as np
from matplotlib import pyplot as plt

before = cv2.imread('after.png')
after = cv2.imread('after2.png')

# before = cv2.bilateralFilter(before,9,75,75)
# after = cv2.bilateralFilter(after,9,75,75)

before = cv2.GaussianBlur(before,(5,5),0)
after = cv2.GaussianBlur(after,(5,5),0)


hsv_frame1 = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
hsv_frame2 = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)

# Pink color
low_pink = np.array([125, 100, 30])
high_pink = np.array([179, 255, 255])


low_white = np.array([0,0,168], dtype=np.uint8)
high_white = np.array([110,111,255], dtype=np.uint8)

# Threshold the HSV image to get only pink and white colors
# Bitwise-AND mask and original image
##pink
pink_mask1 = cv2.inRange(hsv_frame1, low_pink, high_pink)
pink_mask2= cv2.inRange(hsv_frame2, low_pink, high_pink)
##white
white_mask1 = cv2.inRange(hsv_frame1, low_white, high_white)
white_mask2 = cv2.inRange(hsv_frame2, low_white, high_white)


res1= cv2.bitwise_and(before,before, mask= pink_mask1)
res2 = cv2.bitwise_and(after,after, mask= pink_mask2)
res3= cv2.bitwise_and(before, before, mask=white_mask1)
res4= cv2.bitwise_and(after, after, mask=white_mask2)

r,g,b=res4[151 ,  236]
print(r,g,b)
cv2.imshow("pink_b",  res1)
cv2.imshow("pink_a",  res2)
cv2.imshow("white_b", res3)
cv2.imshow("white_a", res4)

cv2.imwrite('pink_b.png', res1)
cv2.imwrite('pink_a.png', res2)
cv2.imwrite('white_b.png',res3)
cv2.imwrite('white_a.png',res4)

cv2.waitKey(0)
###################################
##############33
# ##(pink before and after) center pixel RGB values
            # b0,g0,r0=pink_b[cy,cx] #(pb)
            # b1,g1,r1=pink_a[cy,cx] #(pa)
            # ##(white before and after) center pixel RGB values
            # b2,g2,r2=white_b[cy,cx]
            # b3,g3,r3=white_a[cy,cx]

            # print(b0,g0,r0)
            # print(b1,g1,r1)
            # print(b2,g2,r2)
            # print(b3,g3,r3)
            #get RGB as integers

            # ##pink before(pb)
            # b_pb = int(b0)
            # g_pb = int(g0)
            # r_pb = int(r0)
            # ##pink after (pa)
            # b_pa = int(b1)
            # g_pa = int(g1)
            # r_pa = int(r1)
            # ##white before(wb)
            # b_wb = int(b2)
            # g_wb = int(g2)
            # r_wb = int(r2)
            # ##white after (wa)
            # b_wa = int(b3)
            # g_wa = int(g3)
            # r_wa = int(r3)
            # print(b_pb,g_pb,r_pb)
            # print(b_pa,g_pa,r_pa)
            # print(b_wb,g_wb,r_wb)
            # print(b_wa,g_wa,r_wa)
            # if (r_b>220 and g_b>220 and b_b>220): #WHITE before
            #     if r_a>=199 and (20<=g_a<=192) and (133<=b_a<=203): # pink after
            #         print('cured ')
            #     else :
            #         print('damaged ')
            # elif (r_b>=199) and (20<=g_b<=192) and (133<=b_b<=203) :#PINK before

            #     if (r_a>220 and g_a>220 and b_a>220): #WHITE after
            #         print('bleched')
            #     else :
            #         print('damaged ')
            # elif  r_b>=199 and (20<=g_b<=192) and (133<=b_b<=203): #PINK before
            #     if r_a>=199 and (20<=g_a<=192) and (133<=b_a<=203): # pink after
            #         print('growth')
            #     else:
            #         print('damaged ')

            # elif  r_b>220 and g_b>220 and b_b>220: #WHITE before
            #     if  r_a>220 and g_a>220 and b_a>220: #WHITE after
            #         print('growth ')
            #     else :
            #         print('damaged')
            # else :
            #     print('growth')
