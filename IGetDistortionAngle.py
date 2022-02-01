import cv2 as cv
import numpy as np
from DistortionDectection import ImageAnalysis

def FOVDistortion(org, folderpath):
        path = glob.glob(folderpath)
        FOVmap = np.zeros(org.shape, np.float32)
        imArray = []

        for image in path:
            im1 = org.copy()
            im2 = cv.imread(r""+ image +"", cv.IMREAD_GRAYSCALE)

            shiftX, shiftY = ImageAnalysis.PhaseCorrelation(im1, im2)
            im2 = ImageAnalysis.Shift(im2, shiftY, shiftX)       
            im1, im2 = ImageAnalysis.__Crop(im1, im2, shiftX, shiftY, im1.shape[0], im1.shape[1])

            flow = ImageAnalysis.OpticalFlow(im1, im2)
            ImageAnalysis.__AddToArray(imArray, flow, org, shiftX, shiftY)

        ImageAnalysis.__CreateDistortionMatrix(org, FOVmap, imArray)
        return FOVmap
