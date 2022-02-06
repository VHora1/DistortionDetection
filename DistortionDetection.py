import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

class ImageAnalysis:
    # Description: This Class contains methods for lens distortion computation and related functions
    
    # Method summary:
    # Computes subpixel offset between images using phase correlation
    # Args: Two image matrices; Out: Two offset values in pixels
    @staticmethod
    def PhaseCorrelation(im1, im2):
        chY = 0; chX = 0
        im1 = im1.astype(np.float64); im2 = im2.astype(np.float64)

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

    # Method summary:
    # Computes optical flow matrix
    # Args: Two image matrices; Out: Optical flow matrix
    @staticmethod
    def OpticalFlow(im1, im2):
        flow = cv.calcOpticalFlowFarneback(im1, im2, cv.CV_32FC2, pyr_scale=0.5, levels=1, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
        FlowMat = (((flow[:,:,0])**2 + (flow[:,:,1])**2)**(1/2))
        return FlowMat

    # Method summary:
    # Moves image matrix by given values
    # Args: An image, horizontal shift value and vertical shift value; Out: Shifted image matrix
    @staticmethod
    def Shift(im, x, y):
        transMat = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (im.shape[1], im.shape[0])
        return cv.warpAffine(im, transMat, dimensions)

    # __private method
    # Extracts common areas from images
    @staticmethod
    def __Crop(im1, im2, Xshift, Yshift, shape0, shape1): 
        Xshift = round(Xshift)
        Yshift = round(Yshift)

        if Yshift < 0: shape1 += Yshift; Yshift = 0
        if Xshift < 0: shape0 += Xshift; Xshift = 0

        IM1 = im1[Xshift:shape0, Yshift:shape1]
        IM2 = im2[Xshift:shape0, Yshift:shape1]
        return IM1, IM2

    # __private method
    # Creates matrix for distortion averaging
    @staticmethod
    def __AddToArray(list, im, org, x, y):
        x = round(x); y = round(y)
        rows = org.shape[0]; cols = org.shape[1]
        if y < 0:
            rows += y
            y = 0
        if x < 0:
            cols += x
            x = 0
        temp = np.zeros(org.shape, np.float32)
        temp[x:cols, y:rows] = im
        list.append(temp)

    # Method summary:
    # Computes distortion between two images
    # Args: Two image matrices; Out: Distortion matrix
    @staticmethod
    def CalcDistortion(im1, im2):

        shiftX, shiftY = ImageAnalysis.PhaseCorrelation(im1, im2)
        im2 = ImageAnalysis.Shift(im2, shiftY, shiftX)       
        im1, im2 = ImageAnalysis.__Crop(im1, im2, shiftX, shiftY, im1.shape[0], im1.shape[1])

        flow = ImageAnalysis.OpticalFlow(im1, im2)
        return flow

    # Method summary:
    # Computes distortion of FOV
    # Params: Central image matrix, path to folder containing all the samples; Return: Distortion matrix for central image
    @staticmethod
    def FOVDistortionField(org, folderpath):
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
        
            del im1, im2, image, 
             
        for i in range(0, org.shape[0]):
            for j in range(0, org.shape[1]):
                valsX = []
                valsY = []
                for im in imArrayX:
                    if im[i, j] != 0:
                        valsX.append(im[i, j])
                    if len(valsX) != 0:
                        val = sum(valsX)/len(valsX)
                        FOVmapX[i, j] = val
                for im in imArrayY:
                    if im[i, j] != 0:
                        valsY.append(im[i, j])
                    if len(valsY) != 0:
                        val = sum(valsY)/len(valsY)
                        FOVmapY[i, j] = val
                            
        FOVmap = cv.sqrt((FOVmapX)**2 + (FOVmapY)**2)
        ImageAnalysis.__GetQuiverPlot(FOVmapX, FOVmapY)
        ImageAnalysis.GetContourPlot(FOVmap, 64, 4)
        
        return FOVmap

    # __private method
    # Plots vector field from x and y matrices
    @staticmethod
    def __GetQuiverPlot(matrix_X, matrix_Y):
        x, y = np.meshgrid(np.linspace(0, matrix_X.shape[1], 20), np.linspace(0, matrix_X.shape[0], 20))
        u = cv.resize(matrix_X, (20, 20))
        v = cv.resize(matrix_Y, (20, 20))
        plt.quiver(x, y, u, v, scale = 150)
        plt.axis('equal')
        plt.show()

    # Method summary:
    # Plots contour plot of given matrix
    # Params: matrix, size of resulting plot, contour levels; Return: --
    @staticmethod
    def GetContourPlot(matrix, size, levels):
        matrix = cv.resize(matrix, (size, size))
        plt.contourf(matrix, levels)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.show()  
