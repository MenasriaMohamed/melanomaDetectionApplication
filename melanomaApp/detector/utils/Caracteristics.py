import cv2
import numpy as np
import imutils
from .Preprocess import Preprocess
import math
from matplotlib import pyplot as plt

'''
    get ABCD, 7 points, menzies caracteristics
'''
class Caracteristics:
    '''
        needed for Assymetry A
        get roundness of the contour
    '''
    @staticmethod
    def roundness(img,contour):
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ellipse = cv2.fitEllipse(contour)
        # mask = cv2.ellipse(img, ellipse, (255, 255, 255), 1)
        blankImg = np.zeros(np.shape(imgray))
        # draw ellipse on empty image
        cv2.ellipse(blankImg, ellipse, (255, 255, 255), -1)
        # draw solid contour on empty image
        cv2.drawContours(blankImg, [contour], -1, (0, 0, 0), -1)
        # surface of intersection area
        contourArea = cv2.contourArea(contour)
        intersectionArea = contourArea - np.count_nonzero(blankImg)
        # get the ratio between intersectionArea and the contourArea
        roundness = intersectionArea/contourArea
        print(roundness)
        return roundness
    
    '''
        get Assymetry A of a lesion
    '''
    @staticmethod
    def assymetry(img,contour):
        # get fitted ellipse
        (cx, cy), (_, _), angle = cv2.fitEllipse(contour)
        x, y, w, h = cv2.boundingRect(contour)
        blankImg = np.zeros((h, w))
        contour = Caracteristics.translateContour(contour,x,y)
        cv2.drawContours(blankImg, [contour], -1, (255, 255, 255), -1)
        # cv2.line(blankImg, (int(w/2), int(h/2)), (int(w/2)+h, int(h/2)+h), (0, 255, 0), 1)
        blankImg = imutils.rotate_bound(blankImg, angle+90)
        v = np.sum(blankImg, axis=0)
        h = np.sum(blankImg, axis=1)
        # print(v)
        # print(h)
        cv2.imshow('t',blankImg)
    
    '''
        translates the contour Vector by dx,dy
    '''
    @staticmethod
    def translateContour(contour,dx,dy):
        for p in contour:
            p[0][0]-=dx
            p[0][1]-=dy
        return contour
    
    '''
        needed for Border B
        get Compact Index of a lesion
    '''
    @staticmethod
    def compactIndex(contour):
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        return (contourPerimeter**2) / (4*np.pi*contourArea)
    
    '''
        needed for Border B
        get regularity index
    '''
    @staticmethod
    def regularityIndex(contour):
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        return contourPerimeter / contourArea
    
    '''
        needed for Border B
        get regularity index 2
    '''
    @staticmethod
    def regularityIndexPercentage(contour):
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        # get circle with same piremeter as contour
        radius = int(contourPerimeter / (2 * np.pi))
        # circle area
        circleArea = np.pi * (radius ** 2)
        return contourArea / circleArea
    
    '''
        needed for Color C
        gets number of colors from kmeans centers
    '''
    @staticmethod
    def nbColors(img, contour):
        lesion = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lesion = Caracteristics.extractLesion(lesion, contour)
        lesion, centers = Preprocess.KMEANS(lesion, K=7)
        distances = np.array([])
        for i in range(0, len(centers)-1):
            for j in range(i+1 ,len(centers)):
                center = centers[i]
                center2 = centers[j]
                r = (float(center[0])-float(center2[0]))**2 + (float(center[1])-float(center2[1]))**2 + (float(center[2])-float(center2[2]))**2
                d = math.sqrt(r)
                distances = np.append(distances, d)
        # for i, center in enumerate(centers):
        #     for j, center2 in enumerate(centers):
        #         if(i != j):
        #             r = (float(center[0])-float(center2[0]))**2 + (float(center[1])-float(center2[1]))**2 + (float(center[2])-float(center2[2]))**2
        #             d = math.sqrt(r)
        #             distances = np.append(distances, d)
        s = np.sum(distances)
        print(s)
        # cv2.imshow('nb colors', lesion)
        return s
    
    '''
        needed for Color C
        gets number of colors from color histogram
    '''
    @staticmethod
    def nbColorsHist(img, contour):
        lesion = Caracteristics.extractLesion(img, contour)
        lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2HSV)
        lesionH, lesionS, lesionV = cv2.split(lesion)
        # plt.hist(lesionS.ravel(),256,[0,256])
        hist = cv2.calcHist( [lesion], [0, 1], None, [180, 256], [0, 180, 0, 256] )
        plt.imshow(hist)#, interpolation = 'nearest')
        plt.show(0)
        cv2.imshow('nb colors hsv', lesionS)
    
    '''
        needed for color C
        Kurtosis, color distribution
    '''
    @staticmethod
    def kurtosis(img, contour):
        lesion = Caracteristics.extractLesion(img, contour)
        lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2GRAY)
        area = cv2.contourArea(contour)
        summ = np.sum(lesion)
        mean = summ/area
        r = np.subtract(lesion, mean)
        r = np.power(r, 4)
        r = np.sum(r) / area
        print(r)
        # cv2.imshow('nb colors hsv', lesion)
        return r
    
    '''
        needed for color C
        color thresholds, https://alloyui.com/examples/color-picker/hsv.html
    '''
    @staticmethod
    def colorThreshold(img, contour):
        img = Preprocess.removeArtifactYUV(img)
        lesion = Caracteristics.extractLesion(img, contour)
        colors = [[255,255,255],[0,0,255],[0,0,0],[60,60,128],[60,60,128],[255,0,0]]
        colorCount = [0, 0, 0, 0, 0, 0]
        centers = [
            # white
            [200, 200, 200],
            # red
            [0, 0, 100],
            # black
            [64, 64, 64],
            # brown
            [98, 127, 174],
            [33, 66, 130],
            # grey blue
            [75, 84, 111]
        ]
        # resize ration
        ratio = 4
        h, w = np.shape(lesion)[:2]
        lesion = cv2.resize(lesion, (w//ratio, h//ratio))
        h, w = np.shape(lesion)[:2]
        for x in range(0, w):
            for y in range(0, h):
                if lesion[y][x][0]!=0 and lesion[y][x][1]!=0 and lesion[y][x][2]!=0:
                    distances = [0, 0, 0, 0, 0, 0]
                    center = lesion[y][x]
                    for i in range(0, len(centers)):
                        center2 = centers[i]
                        r = (float(center[0])-float(center2[0]))**2 + (float(center[1])-float(center2[1]))**2 + (float(center[2])-float(center2[2]))**2
                        d = math.sqrt(r)
                        distances[i] = d
                    minim = min(distances)
                    m = distances.index(minim)
                    # seuil = 15
                    lesion[y][x] = colors[m] #if minim<seuil else [0,255,0]
                    # if minim<seuil:
                    colorCount[m] += 1
        area = cv2.contourArea(contour)
        colorCount = np.array(colorCount)
        # print(colorCount)
        nbColors = np.count_nonzero(colorCount>(area*0.01))
        print(nbColors)
        # cv2.imshow('nb colors', lesion)
        return nbColors
    
    '''
        thresh numpy
    '''
    @staticmethod
    def colorThresholdNumpy(img, contour):
        img = Preprocess.removeArtifactYUV(img)
        lesion = Caracteristics.extractLesion(img, contour)
        colors = [[255,255,255],[0,0,255],[0,0,0],[60,60,128],[60,60,128],[255,0,0]]
        colorCount = [0, 0, 0, 0, 0, 0]
        centers = [
            # white
            [200, 200, 200],
            # red
            [0, 0, 100],
            # black
            [64, 64, 64],
            # brown
            [98, 127, 174],
            [33, 66, 130],
            # grey blue
            [75, 84, 111]
        ]
        distances = []
        for i in range(0, len(centers)):
            copy = np.copy(lesion)
            center = centers[i]
            copy[:,:,0] = np.subtract(copy[:,:,0], center[0])
            copy[:,:,1] = np.subtract(copy[:,:,1], center[1])
            copy[:,:,2] = np.subtract(copy[:,:,2], center[2])
            r = np.power(copy[:,:,0], 2) + np.power(copy[:,:,1], 2) + np.power(copy[:,:,2], 2)
            dd = np.sqrt(r)
            distances.append(dd)
        minDistances = np.min(distances, axis=0)
        minIndexes = np.argmin(distances, axis=0)
        h, w = np.shape(lesion)[:2]
        for x in range(0, w):
            for y in range(0, h):
                if lesion[y][x][0]!=0 and lesion[y][x][1]!=0 and lesion[y][x][2]!=0:
                    seuil = 35
                    m = minIndexes[y][x]
                    lesion[y][x] = colors[m]
        cv2.imshow('nb colors', lesion)

    '''
        needed for Color C
        extracts the lesion
    '''
    @staticmethod
    def extractLesion(img, contour):
        mask = np.zeros(img.shape, dtype='uint8')
        mask = cv2.drawContours(mask, [contour], -1, (255 , 255 , 255),thickness=cv2.FILLED)
        # mask = cv2.bitwise_not(mask)
        img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        lesion = cv2.bitwise_and(img, img, mask=mask)
        return lesion
    
    '''
        Distance Between the center of gravity of contour and  center of circle around the contour
    '''
    @staticmethod
    def AssymetryByDistanceByCircle(img, contour):
        # get moment of contour
        M = cv2.moments(contour)
        # get center of gravity of contour
        xe = int(M["m10"] / M["m00"])
        ye = int(M["m01"] / M["m00"])
        # get center of circle around the contour
        #  center gravity
        cv2.circle(img, (xe, ye), radius=2, color=(0, 255, 255), thickness=1)
        (xCiCe, yCiCe), radius = cv2.minEnclosingCircle(contour)
        xCiCe = int(xCiCe)
        yCiCe = int(yCiCe)
        cv2.circle(img, (xCiCe, yCiCe), radius=2, color=(0, 0, 255), thickness=1)
        asm = 100 - (Caracteristics.DistanceEuclidean(xe, ye, xCiCe, yCiCe) * 100 / radius)
        print(asm)
        return asm


    '''
      Start  All About Diameter
    '''
    # caluclate Diameter by Diameter of Circle around contour
    @staticmethod
    def DiameterByCircle(img, contour):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        # and draw the circle in blue
        img = cv2.circle(img, center, radius, (255, 0, 0), 2)
        return (radius*2)

    '''
      End  All About Diameter
    '''


    '''
        calculate the distance eclidiane
    '''

    def DistanceEuclidean(x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist
