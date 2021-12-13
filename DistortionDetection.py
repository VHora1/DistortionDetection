import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

global filepathFirst, filepathsecond, FOVmap

centralImage = "path to central image"
path = glob.glob("path to the folder")

def main():
    org = cv.imread(r""+ centralImage +"", cv.IMREAD_GRAYSCALE)
    FOVmap = np.zeros(org.shape, np.float32)
    imArr = []

    #caluculating optical flow map for each image
    for image in path:
        im1 = cv.imread(r""+ centralImage +"", cv.IMREAD_GRAYSCALE)
        im2 = cv.imread(r""+ image +"", cv.IMREAD_GRAYSCALE)

        shiftX, shiftY = phaseCorrelation(im1, im2)
        im2 = Shift(im2, shiftY, shiftX)
        im1, im2 = Crop(im1, im2, shiftX, shiftY, im1.shape[0], im1.shape[1])
        flow = opticalFlow(im1, im2)

        addToArray(imArr, flow, org, shiftX, shiftY)

    #assinging pixel values to FOVmap
    for i in range(0, org.shape[0]):
        for j in range(0, org.shape[1]):
            vals = []
            for im in imArr:
                if im[i, j] != 0:
                    vals.append(im[i, j])
            val = sum(vals)/len(vals)

            FOVmap[i, j] = val * 2
    
    plt.imsave("selected dir", FOVmap)

def addToArray(list, im, org, x, y): 
    x = round(x)
    y = round(y)
    rows = org.shape[0]
    cols = org.shape[1]
    if y < 0:
        rows += y
        y = 0
    if x < 0:
        cols += x
        x = 0
    temp = np.zeros(org.shape, np.float32)
    temp[x:cols, y:rows] = im
    list.append(temp)

#optical flow calculation
def opticalFlow(im1, im2): 
    flow = cv.calcOpticalFlowFarneback(im1, im2, cv.CV_32FC2, pyr_scale=0.5, levels=1,
        winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
    FlowMat = (((flow[:,:,0])**2 + (flow[:,:,1])**2)**(1/2))
    return FlowMat

#image shift method
def Shift(img, x, y): 
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

#image crop method
def Crop(im1, im2, Xshift, Yshift, shape0, shape1): 
    Xshift = round(Xshift)
    Yshift = round(Yshift)

    if Yshift < 0:
        shape1 += Yshift
        Yshift = 0
    if Xshift < 0:
        shape0 += Xshift
        Xshift = 0
    a = im1[Xshift:shape0, Yshift:shape1]
    b = im2[Xshift:shape0, Yshift:shape1]
    return a, b
    
#phase correlation calculation
def phaseCorrelation(im1, im2): 
    chY = 0
    chX = 0
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    if im1.shape[0] > 1200 and im1.shape[1] > 1200:
        k = int(im1.shape[1]//8)
        l = int(im1.shape[0]//8)

        smallIm1 = cv.resize(im1, (k, l))
        smallIm2 = cv.resize(im2, (k, l))

        shift1, response1 = cv.phaseCorrelate(smallIm1, smallIm2)

        chX = round(shift1[1] * -8)
        chY = round(shift1[0] * -8)

        win = 50
        centX = im1.shape[0]//2
        centY = im1.shape[1]//2
        im1 = im1[(centX - win):(centX + win), (centY - win):(centY + win)]
        im2 = im2[(centX - chX - win):(centX - chX + win), (centY - chY - win):(centY - chY + win)]

    shift2, response2 = cv.phaseCorrelate(im1, im2)
    subY, subX = shift2

    phaseShiftY = chY - subY
    phaseShiftX = chX - subX

    return phaseShiftX, phaseShiftY

if __name__ == "__main__":
    main()