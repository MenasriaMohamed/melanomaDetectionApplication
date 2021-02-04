import cv2
import numpy as np

class Preprocess:

    '''
        removes hair (artifacts) from an image
        morphologic close transformation
    '''
    @staticmethod
    def removeArtifact(img):
        # perform closing to remove hair
        kernel = np.ones((15,15),np.uint8)
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 2)
        return closing
    
    '''
        ASLM Noise Removal
        equalizes the Y channel of an YUV image, Y contains the intensity information
        https://www.opencv-srf.com/2018/02/histogram-equalization.html
        http://users.diag.uniroma1.it/bloisi/papers/bloisi-CMIG-2016-draft.pdf
    '''
    @staticmethod
    def equalizeHistYChannel(img):
        # convert image to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert image to RGB
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img
    
    '''
        DR noise removal
        apply morphologic close transformation on each channel of RGB image
        kernel of (11,11) based on hair size
    '''
    @staticmethod
    def removeArtifactRGB(img):
        # median filtre
        imgMedian = cv2.medianBlur(img, 5, 5)
        # split RGB channels
        imgB, imgG, imgR = cv2.split(imgMedian)
        # kernel of 11 * 11
        kernel = np.ones((11, 11), np.uint8)
        # perform morphologic closing on each RGB channel
        imgClosingB = cv2.morphologyEx(imgB, cv2.MORPH_CLOSE, kernel)
        imgClosingG = cv2.morphologyEx(imgG, cv2.MORPH_CLOSE, kernel)
        imgClosingR = cv2.morphologyEx(imgR, cv2.MORPH_CLOSE, kernel)
        # merge the 3 channels
        imgResult = cv2.merge((imgClosingB, imgClosingG, imgClosingR))
        return imgResult

    '''
        DR noise removal
        1) convert BGR to YUV, 2) process, 3) convert YUV to RGB for OTSU
        apply morphologic close transformation on each channel of YUV image
        kernel of (11,11) based on hair size
    '''
    @staticmethod
    def removeArtifactYUV(img):
        # median filtre
        imgMedian = cv2.medianBlur(img, 5, 5)
        # split YUV channels
        img_yuv = cv2.cvtColor(imgMedian, cv2.COLOR_RGB2YUV)
        imgV, imgU, imgY = cv2.split(img_yuv)
        # kernel of 11 * 11
        kernel = np.ones((11, 11), np.uint8)
        # perform morphologic closing on each RGB channel
        imgClosingV = cv2.morphologyEx(imgV, cv2.MORPH_CLOSE, kernel)
        imgClosingU = cv2.morphologyEx(imgU, cv2.MORPH_CLOSE, kernel)
        imgClosingY = cv2.morphologyEx(imgY, cv2.MORPH_CLOSE, kernel)
        # merge the 3 channels
        imgResult = cv2.merge((imgClosingV, imgClosingU, imgClosingY))
        # back to RGB
        imgResult = cv2.cvtColor(imgResult, cv2.COLOR_YUV2RGB)
        return imgResult
    
    '''
        apply OTSU threshold
    '''
    @staticmethod
    def OTSUThreshold(img):
        # blur the image
        blur = cv2.blur(img,(15,15))
        # apply OTSU threshold
        imgray = cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY)
        # remove tint effect
        # if Preprocess.hasTint(img):
        mask = Preprocess.removeTint(img)
        imgray = cv2.add(imgray, mask)
        # # noise removal
        kernel = np.ones((7,7),np.uint8)
        imgray = cv2.morphologyEx(imgray,cv2.MORPH_ERODE,kernel, iterations = 2)
        # cv2.imshow('mask',imgray)
        ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return ret, thresh

    '''
        uses the SLIC clustering to extract borders
        https://jayrambhia.com/blog/superpixels-slic
        returns the result img
    '''
    @staticmethod
    def SLIC(img):
        slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=cv2.ximgproc.MSLIC, region_size=300, ruler=0.075)
        color_img = np.zeros(img.shape, np.uint8)
        color_img[:] = (0, 0, 0)
        for n in range(2):
            slic.iterate(2)
        slic.enforceLabelConnectivity()
        mask = slic.getLabelContourMask(False)
        # stitch foreground & background together
        mask_inv = cv2.bitwise_not(mask)
        result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
        result = cv2.add(result_bg, result_fg)
        # cv2.imshow('SLIC',mask_inv)
        return result

    '''
        uses KMEANS clustering
    '''
    @staticmethod
    def KMEANS(img, K=5):
        # apply KMEANS
        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result = res.reshape((img.shape))
        return result, center

    '''
        returns a mask that removes tint effect from corners of the img
        https://stackoverflow.com/questions/42594993/gradient-mask-blending-in-opencv-python
    '''
    @staticmethod
    def removeTint(img):
        H,W = img.shape[:2]
        mask = np.zeros((H,W), np.uint8)
        cv2.circle(mask, (W//2, H//2), W//2 + W//50, (150,150,150), -1, cv2.LINE_AA)
        mask = cv2.blur(mask, (321,321))
        mask = 160 - mask
        return mask

    @staticmethod
    def hasTint(img):
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        H,W = imgray.shape[:2]
        mask = np.zeros((H,W), np.uint8)
        cv2.circle(mask, (W//2, H//2), W//2 + W//50, (255,255,255), -1, cv2.LINE_AA)
        mask = cv2.blur(mask, (21,21))
        # mask = 255 - mask
        mask = cv2.subtract(imgray, mask)
        seuil = 140
        return mask[10,10]<seuil