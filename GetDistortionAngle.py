import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from DistortionDectection import ImageAnalysis

# Method summary:
# Calculates distortion vector field for lens FOV
# Params: Central image matrix, path to folder containing all the samples; Return: Distortion matrix for central image with vector field
def GetFOVDistortionField(org, folderpath):
    path = glob.glob(folderpath)
    FOVmapX = np.zeros(org.shape, np.float32)
    FOVmapY = np.zeros(org.shape, np.float32)
    imArrayX = []; imArrayY = []

    for image in path:
        im1 = org.copy()
        im2 = cv.imread(r""+ image +"", cv.IMREAD_GRAYSCALE)

        shiftX, shiftY = ImageAnalysis.PhaseCorrelation(im1, im2)
        im2 = ImageAnalysis.Shift(im2, shiftY, shiftX)       
        im1, im2 = ImageAnalysis.__Crop(im1, im2, shiftX, shiftY, im1.shape[0], im1.shape[1])

        flow = cv.calcOpticalFlowFarneback(im1, im2, cv.CV_32FC2, pyr_scale=0.5, levels=1, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
        ImageAnalysis.__AddToArray(imArrayY, flow[:, :, 0], org, shiftX, shiftY)
        ImageAnalysis.__AddToArray(imArrayX, flow[:, :, 1], org, shiftX, shiftY)
        
    for i in range(0, org.shape[0]):
        for j in range(0, org.shape[1]):
            vals = []
            for im in imArrX:
                if im[i, j] != 0:
                    vals.append(im[i, j])
                if len(vals) != 0:
                    val = sum(vals)/len(vals)
                    FOVmapX[i, j] = val
            for im in imArrY:
                if im[i, j] != 0:
                    vals.append(im[i, j])
                if len(vals) != 0:
                    val = sum(vals)/len(vals)
                    FOVmapY[i, j] = val
                        
    FOVmap = sqrt((FOVmapX) + (FOVmapY))
    GetQuiverPlot(FOVmapX, FOVmapY, FOVmap.shape[1], FOVmap.shpae[0])
    ImageAnalysis.GetContourPlot(FOVmap, FOVmap.shape[1], FOVmap.shape[0])
    
    return FOVmap

# __private mathod
# plots vector field from x and y matrices
def __GetQuiverPlot(matrix_X, matrix_Y):
    x, y = np.meshgrid(np.linspace(0, matrix_X.shape[1], 20), np.linspace(0, matrix_X.shape[0], 20))
    u = cv.resize(matrix_X, (20, 20))
    v = cv.resize(matrix_Y, (20, 20))
    plt.quiver(x, y, u, v, scale = 150)
    plt.axis('equal')
    plt.show()
