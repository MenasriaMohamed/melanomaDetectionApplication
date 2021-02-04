import cv2
import numpy as np
from .Preprocess import Preprocess

class Contours:
    '''
        method 1
        get contours
    '''
    @staticmethod
    def contours1(img):
        # convert img to grayscale
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply threshold
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        # get contours
        c,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    '''
        method 2
        get contours
    '''
    @staticmethod
    def contours2(img):
        # equalize Y channel hist
        # img = Preprocess.equalizeHistYChannel(img)
        # remove artifacts
        # img = Preprocess.removeArtifact(img)
        # remove RGB artifact
        img=Preprocess.removeArtifactYUV(img)
        # apply OTSU threshold
        ret, thresh = Preprocess.OTSUThreshold(img)
        # search for contours and select the biggest one
        c, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        return cnt

    '''
        draw bounding circle
    '''
    @staticmethod
    def boundingCircle(img,contour):
        # get perimeter of contour
        perimeter = cv2.arcLength(contour, True)
        # get moment of contour
        M = cv2.moments(contour)
        # get center of gravity of contour
        xe = int(M["m10"] / M["m00"])
        ye = int(M["m01"] / M["m00"])
        # get center of circle around the contour
        radius = int(perimeter / (2 * np.pi))
        # draw the circle and its center
        cv2.circle(img, (xe, ye), radius=1, color=(255, 255, 255), thickness=2)
        cv2.circle(img, (xe, ye), radius=radius, color=(255, 255, 255), thickness=2)

    '''
        draw bounding rectangle
    '''
    @staticmethod
    def boundingRectangle(img,contour):
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img,(x,y),(x+w,y+h), color=(255, 255, 255), thickness=2)
    
    '''
        draw bounding rectangle
    '''
    @staticmethod
    def boundingRectangleRotated(img,contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rows,cols = img.shape[:2]
        # height and width of minAreaRect
        (x, y), (width, height), angle = rect
        cv2.putText(
            img,
            'lesion ('+str(round(height*0.026458333))+'cm x '+str(round(width*0.026458333))+'cm)',
            (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
        # line
        [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
        cv2.drawContours(img, [box], 0, (255, 255, 255), 2)


    '''
        get roundiness
    '''
    @staticmethod
    def roundness(contour):
        # get surface of contour
        area = cv2.contourArea(contour)
        # get perimeter of contour
        perimeter = cv2.arcLength(contour, True)
        # get roundness
        roundness = (4 * np.pi * area) / (perimeter * perimeter) * 100
        roundness = round(roundness, 2)