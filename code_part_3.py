########### code part 3 #############
########### drawing contours and defining diff changes ########


from skimage.measure import compare_ssim
import cv2
import numpy as np

#capturong pic from video
# cap=cv2.VideoCapture(0)
# while 1:
#     ret,frame=cap.read()
#     cv2.imshow('Video',frame)
#     k = cv2.waitKey(3)
#     if k==ord('s') or k==ord('S'):
#         saved_frame=frame
#         break
#     elif k==27:
#         break
# read imges to compare
original_b=cv2.imread('after.png')
original_a=cv2.imread('after2.png')

#read images to compare after filteration
##pink image as main source
before = cv2.imread('pink_b.png')
after=cv2.imread('pink_a.png')

##white image as sec sourse
white_b=cv2.imread('white_b.png')
white_a=cv2.imread('white_a.png')
##resize
before = cv2.resize(before,(500, 500))
after = cv2.resize(after,(500, 500))
white_a=cv2.resize(white_b,(500, 500))
white_b=cv2.resize(white_b,(500, 500))

# Convert main images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(before_gray, after_gray, full=True)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()


# drawing coulred contours and defining changes
for (i,c) in enumerate(sorted_contours): #i == contours indexing
    # while area bigger than 40 draw contour
    area = cv2.contourArea(c)
    if area > 40:
        # for contour num one
        if i==0:
            x,y,w,h = cv2.boundingRect(c)
            #show contour only
            img_crop=before[y:y+h, x:x+w]
            img_crop2=after[y:y+h, x:x+w]

            #calculate countour center coordinates
            M = cv2.moments(contours[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("Centroid = ", cx, ", ", cy)

            #center of countour in both images after and before as there are n=in same size ::
            center_px=before[cy,cx]
            center_px1=after[cy,cx]


            #get center ponit BGR values of images and filtered images
            bb,gb,rb = before[cy,cx]
            ba,ga,ra = after[cy,cx]

            ##(white before and after) center pixel RGB values
            b2,g2,r2=white_b[cy,cx]
            b3,g3,r3=white_a[cy,cx]



            #RGB as integers
            ##before
            b_b = int(bb)
            g_b = int(gb)
            r_b = int(rb)
            ##after
            b_a = int(ba)
            g_a = int(ga)
            r_a = int(ra)

            ##white before(wb)
            b_wb = int(b2)
            g_wb = int(g2)
            r_wb = int(r2)
            ##white after (wa)
            b_wa = int(b3)
            g_wa = int(g3)
            r_wa = int(r3)


            # if substraction of RGB values == -ve ,
            if (b_b-b_a<0) or (g_b-g_a<0) or (r_b-r_a<0): #using main image (pink only)
                #two possible states
                # from (white - pink )=-ve  ,cured
                if b_wb or g_wb or r_wb : #using sec image (white only)
                    cr,cg,cb=[0,0,255]

                #from (non -pink)=-ve , growth
                else :
                    cr,cg,cb=[0,255,0]
            #if substraction of RGB values == +ve,
            elif (b_b-b_a>0) or (g_b-g_a>0) or (r_b-r_a>0): #using main image (pink only)
                #two possible states
                # from (pink - white )=+ve  ,bleached
                if b_wa==255 and g_wa==255 and r_wa==255 :
                    cr,cg,cb=[255,0,0]
                #from (pink ,non)=+ve ,damaged
                else :
                    cr,cg,cb=[255,255,0]

            # if substraction of RGB values == zero,
            elif (b_wb-b_wa==0) or (g_wb-g_wa==0) or (r_wb-r_wa==0) :#using sec image (white only)
                #from (pink -pink) or (white -white)=0 , growth
                cr,cg,cb=[0,255,0]

            # if substraction of RGB values == 255(white),
            elif (b_wb-b_wa==255) or (g_wb-g_wa==255) or (r_wb-r_wa==255):#using main image (pink only)
                #two possible states
                # if pink in after image  existed,
                #from (white -pink)==whiteonly(white-black)==255 ,cured
                if (b_a) or (g_a) or (r_a):
                    zeros='cured'
                    cr,cg,cb=[0,0,255]
                    print(z)
                #if pink after not existing
                #from(white to black)==0,damaged
                else :
                    cr,cg,cb=[255,255,0]

            #draw rectangle contours colored
            cv2.rectangle(before, (x, y), (x + w, y + h), (cr,cg,cb), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (cr,cg,cb), 2)
            cv2.drawContours(mask, [c], 0, (cr,cg,cb), -1)

            #show images
            cv2.imshow("img_crop_before0",img_crop)
            cv2.imshow('img_crop_after0',img_crop2)
            cv2.imwrite('contour_0_b.png',img_crop)
            cv2.imwrite('contour_0_a.png',img_crop2)
            cv2.imwrite('original_a.png',original_a)

        #for contour num two
        elif i==1:
            x,y,w,h = cv2.boundingRect(c)
            #show contour only
            img_crop=before[y:y+h, x:x+w]
            img_crop2=after[y:y+h, x:x+w]

            #calculate countour center coordinates
            M = cv2.moments(contours[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("Centroid = ", cx, ", ", cy)

            #center of countour in both images after and before as there are n=in same size ::
            center_px=before[cy,cx]
            center_px1=after[cy,cx]



            #get center ponit BGR values of images and filtered images
            bb,gb,rb = before[cy,cx]
            ba,ga,ra = after[cy,cx]

            ##(white before and after) center pixel RGB values
            b2,g2,r2=white_b[cy,cx]
            b3,g3,r3=white_a[cy,cx]



            #RGB as integers
            ##before
            b_b = int(bb)
            g_b = int(gb)
            r_b = int(rb)
            ##after
            b_a = int(ba)
            g_a = int(ga)
            r_a = int(ra)

            ##white before(wb)
            b_wb = int(b2)
            g_wb = int(g2)
            r_wb = int(r2)
            ##white after (wa)
            b_wa = int(b3)
            g_wa = int(g3)
            r_wa = int(r3)


            # if substraction of RGB values == -ve ,
            if (b_b-b_a<0) or (g_b-g_a<0) or (r_b-r_a<0): #using main image (pink only)
                #two possible states
                # from (white - pink )=-ve  ,cured
                if b_wb or g_wb or r_wb : #using sec image (white only)
                    cr,cg,cb=[0,0,255]

                #from (non -pink)=-ve , growth
                else :
                    cr,cg,cb=[0,255,0]
            #if substraction of RGB values == +ve,
            elif (b_b-b_a>0) or (g_b-g_a>0) or (r_b-r_a>0): #using main image (pink only)
                #two possible states
                # from (pink - white )=+ve  ,bleached
                if b_wa==255 and g_wa==255 and r_wa==255 :
                    cr,cg,cb=[255,0,0]
                #from (pink ,non)=+ve ,damaged
                else :
                    cr,cg,cb=[255,255,0]

            # if substraction of RGB values == zero,
            elif (b_wb-b_wa==0) or (g_wb-g_wa==0) or (r_wb-r_wa==0) :#using sec image (white only)
                #from (pink -pink) or (white -white)=0 , growth
                cr,cg,cb=[0,255,0]

            # if substraction of RGB values == 255(white),
            elif (b_wb-b_wa==255) or (g_wb-g_wa==255) or (r_wb-r_wa==255):#using main image (pink only)
                #two possible states
                # if pink in after image  existed,
                #from (white -pink)==whiteonly(white-black)==255 ,cured
                if (b_a) or (g_a) or (r_a):
                    zeros='cured'
                    cr,cg,cb=[0,0,255]
                    print(z)
                #if pink after not existing
                #from(white to black)==0,damaged
                else :
                    cr,cg,cb=[255,255,0]

            #draw rectangle contours colored
            cv2.rectangle(before, (x, y), (x + w, y + h), (cr,cg,cb), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (cr,cg,cb), 2)
            cv2.drawContours(mask, [c], 0, (cr,cg,cb), -1)
            cv2.imshow("img_crop_before1",img_crop)
            cv2.imshow('img_crop_after1',img_crop2)
            cv2.imwrite('contour_one_b.png',img_crop)
            cv2.imwrite('contour_one_a.png',img_crop2)
        elif i==2:
            x,y,w,h = cv2.boundingRect(c)
#show contour only
            img_crop=before[y:y+h, x:x+w]
            img_crop2=after[y:y+h, x:x+w]

            #calculate countour center coordinates
            M = cv2.moments(contours[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("Centroid = ", cx, ", ", cy)

            #center of countour in both images after and before as there are n=in same size ::
            center_px=before[cy,cx]
            center_px1=after[cy,cx]


            #get center ponit BGR values of images and filtered images
            bb,gb,rb = before[cy,cx]
            ba,ga,ra = after[cy,cx]

            ##(white before and after) center pixel RGB values
            b2,g2,r2=white_b[cy,cx]
            b3,g3,r3=white_a[cy,cx]


            #RGB as integers
            ##before
            b_b = int(bb)
            g_b = int(gb)
            r_b = int(rb)
            ##after
            b_a = int(ba)
            g_a = int(ga)
            r_a = int(ra)

            ##white before(wb)
            b_wb = int(b2)
            g_wb = int(g2)
            r_wb = int(r2)
            ##white after (wa)
            b_wa = int(b3)
            g_wa = int(g3)
            r_wa = int(r3)


            # if substraction of RGB values == -ve ,
            if (b_b-b_a<0) or (g_b-g_a<0) or (r_b-r_a<0): #using main image (pink only)
                #two possible states
                # from (white - pink )=-ve  ,cured
                if b_wb or g_wb or r_wb : #using sec image (white only)
                    cr,cg,cb=[0,0,255]

                #from (non -pink)=-ve , growth
                else :
                    cr,cg,cb=[0,255,0]
            #if substraction of RGB values == +ve,
            elif (b_b-b_a>0) or (g_b-g_a>0) or (r_b-r_a>0): #using main image (pink only)
                #two possible states
                # from (pink - white )=+ve  ,bleached
                if b_wa==255 and g_wa==255 and r_wa==255 :
                    cr,cg,cb=[255,0,0]
                #from (pink ,non)=+ve ,damaged
                else :
                    cr,cg,cb=[255,255,0]

            # if substraction of RGB values == zero,
            elif (b_wb-b_wa==0) or (g_wb-g_wa==0) or (r_wb-r_wa==0) :#using sec image (white only)
                #from (pink -pink) or (white -white)=0 , growth
                cr,cg,cb=[0,255,0]

            # if substraction of RGB values == 255(white),
            elif (b_wb-b_wa==255) or (g_wb-g_wa==255) or (r_wb-r_wa==255):#using main image (pink only)
                #two possible states
                # if pink in after image  existed,
                #from (white -pink)==whiteonly(white-black)==255 ,cured
                if (b_a) or (g_a) or (r_a):
                    zeros='cured'
                    cr,cg,cb=[0,0,255]
                    print(z)
                #if pink after not existing
                #from(white to black)==0,damaged
                else :
                    cr,cg,cb=[255,255,0]

            #draw rectangle contours colored
            cv2.rectangle(before, (x, y), (x + w, y + h), (cr,cg,cb), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (cr,cg,cb), 2)
            cv2.drawContours(mask, [c], 0, (cr,cg,cb), -1)



            #anything else
            cv2.imshow("img_crop_before2",img_crop)
            cv2.imshow('img_crop_after2',img_crop2)
            cv2.imwrite('contour_two_b.png',img_crop)
            cv2.imwrite('contour_two_a.png',img_crop2)




cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff',diff)
cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)
cv2.waitKey(0)
