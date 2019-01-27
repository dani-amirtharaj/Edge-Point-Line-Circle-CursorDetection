
# Setting up the program and reading input image
import cv2
import numpy as np
image=cv2.imread('Images/sobel.png',cv2.IMREAD_GRAYSCALE)
imageArray=np.asarray(image)

# Function to compute and add zero pads to the input image
def padImage(imageArray):
    
    paddedImage=[]
    imageHpad=[0 for i in range(len(imageArray[0])+2)]
    paddedImage.append(imageHpad)
    
    for row in range(len(imageArray)):
        paddedImageRow=[]
        paddedImageRow.append(0)
        
        for column in range(len(imageArray[0])):
            paddedImageRow.append(imageArray[row][column])
        paddedImageRow.append(0)
        paddedImage.append(paddedImageRow)
    
    paddedImage.append(imageHpad)
    return paddedImage              

# Function to apply both the x and y sobel filters on the input image 
#  and return the horizontal(y) and vertical(x) filtered results
def sobelFilter(imageArray):
    
    # x gradient filter to detect vertical edges     
    vFilter=[[-1,0,1],[-2,0,2],[-1,0,1]]
    
    # y gradient filter to detect horizontal edges 
    hFilter=[[-1,-2,-1],[0,0,0],[1,2,1]]
    
    # Window index to iterate over the window
    windowIndex=[-1,0,1]
    
    # Applying y sobel operator on image
    hFilterResult=[]
    for row in range(1,len(imageArray)-1):
        hFilterResultRow=[]
        for column in range(1,len(imageArray[0])-1):
            pixelResult=0
            for rIndex in windowIndex:
                for cIndex in windowIndex:
                    pixelResult+=imageArray[row+rIndex][column+cIndex
                                    ]*hFilter[rIndex+1][cIndex+1]
            hFilterResultRow.append(pixelResult)
        hFilterResult.append(hFilterResultRow)
    
    # Applying x sobel operator on image
    vFilterResult=[]
    for row in range(1,len(imageArray)-1):
        vFilterResultRow=[]
        for column in range(1,len(imageArray[0])-1):
            pixelResult=0
            for rIndex in windowIndex:
                for cIndex in windowIndex:
                    pixelResult+=imageArray[row+rIndex][column+cIndex
                                    ]*vFilter[rIndex+1][cIndex+1]
            vFilterResultRow.append(pixelResult)
        vFilterResult.append(vFilterResultRow)
        
    return hFilterResult, vFilterResult

# Calling zero padding function
paddedImage = padImage(imageArray)

# Applying sobel operators on image
hResult, vResult = sobelFilter(paddedImage)

# Min and Max values of each result used for normalizing
hResultmin = min(min(hResult))
vResultmin = min(min(vResult))
hResultmax = max(max(hResult))
vResultmax = max(max(vResult))

hSobelGrad = np.asarray(hResult)
vSobelGrad = np.asarray(vResult)

# Normalizing gradient image to eliminate negative values
hSobelGrad=(hSobelGrad-hResultmin)/(hResultmax-hResultmin);
vSobelGrad=(vSobelGrad-vResultmin)/(vResultmax-vResultmin);

# Display gradient images
cv2.imshow('Horizontal Sobel gradient',hSobelGrad)
cv2.imshow('Vertical Sobel gradient',vSobelGrad)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Store gradient images
cv2.imwrite('Results/Horizontal Sobel gradient.png',hSobelGrad*255)
cv2.imwrite('Results/Vertical Sobel gradient.png',vSobelGrad*255)

# Horizontal magnitude
hSobelMagnitude = np.sqrt(np.array(hResult) ** 2)
hSobelMagnitude /= np.max(hSobelMagnitude)

# Vertical magnitude
vSobelMagnitude = np.sqrt(np.array(vResult) ** 2)
vSobelMagnitude /= np.max(vSobelMagnitude)

# Total magnitude
sobelMagnitude = np.sqrt(np.array(hResult) ** 2 + np.array(vResult) ** 2)
sobelMagnitude /= np.max(sobelMagnitude)

# Display Magnitude images
cv2.imshow('Sobel Magnitude', sobelMagnitude)
cv2.imshow('Horizontal Sobel Magnitude', hSobelMagnitude)
cv2.imshow('Vertical Sobel Magnitude', vSobelMagnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Storing Magnitude images
cv2.imwrite('Results/Sobel Magnitude.png',sobelMagnitude*255)
cv2.imwrite('Results/Horizontal edge.png',hSobelMagnitude*255)
cv2.imwrite('Results/Vertical edge.png',vSobelMagnitude*255)