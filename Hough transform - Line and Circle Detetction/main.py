
# Importing necessary packages for this project
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setting seed for reproducibility
UBIT = 'damirtha'
np.random.seed(sum([ord(c) for c in UBIT]))


#-------Line Detection-------#

# Apply gradient mask on image for edge detection, given image and mask
def gradientFilter(image, mask):
    img_list = []
    for img_row in range(int(len(mask)/2), len(image)-int(len(mask)/2)):
        for img_col in range(int(len(mask[0])/2), len(image[0])-int(len(mask[0])/2)):
            img_list.append(np.mean(np.multiply(image[img_row-int(len(mask)/2):img_row+int(len(mask)/2)+1,
                                         img_col-int(len(mask[0])/2):img_col+int(len(mask[0])/2)+1]
                                   , mask)))
    return np.array(img_list).reshape(-1,len(image[0])-len(mask[0])+1)

image = cv2.imread('Images/hough.jpg')
binImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Laplacian mask for edge detection
laplacian = -np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
imageGrad = gradientFilter(binImage, laplacian)

# Creating thresholded binary image of edges detected
maxGradT = np.max(imageGrad)*0.1
imageGrad = np.array([255 if imageGrad[i,j]> maxGradT else 0 for i in range(len(imageGrad)) for j in range(len(imageGrad[0]))]).reshape(-1, len(imageGrad[0]))
imageGrad = np.pad(imageGrad, int(len(laplacian)/2),'edge')

# Calculating rho for a range of theta and the input pixel coordinates
imageGradX, imageGradY = np.where(imageGrad == 255)
thetaRange = np.transpose(np.array([i for i in range(-90,90)]).reshape(-1,1))
rhoCalc = np.round(np.matmul(imageGradX.reshape(-1,1), np.cos((thetaRange*np.pi/180)))+np.matmul(imageGradY.reshape(-1,1), np.sin((thetaRange*np.pi/180))))

# Calculation of hough space accumulator, in matrix - rhoTheta
rhoTheta = []
for i in range(np.int32(np.min(rhoCalc)),np.int32(np.max(rhoCalc)+1)):
    for j in range(thetaRange.size):
        rhoTheta.append(np.size(np.where(rhoCalc[:,j] == i)))
rhoTheta = np.array(rhoTheta).reshape(-1,thetaRange.size)

# Get only lines with atlease 120 hits in the accumulator cells
goodRho, goodTheta = np.where(rhoTheta>120)

imageResultRed = image.copy()
imageResultBlue = image.copy()
edges = np.zeros(imageGrad.shape)
indent = np.min(rhoCalc)
blueLines = set({})
redLines = set({})

# Plotting the detected lines on the image
for i,j in zip(goodRho, goodTheta):
    """ Considering lines within a bin of r-12 and r+12 and theta+1 and theta-1 
        to be one line to avoid labelling the same line as multiple lines. """
    ind = np.where((goodRho>=i) & (goodRho<i+25) & (goodTheta<j+4) & (goodTheta>=j))
    if goodRho[ind].size > 0 or goodTheta[ind].size > 0:
        i = np.mean(goodRho[ind])
        j = np.mean(goodTheta[ind])
    else:
        continue
        
    goodRho = np.delete(goodRho, ind)
    goodTheta = np.delete(goodTheta, ind)
    
    xTheta = np.cos((j-90)*np.pi/180)
    yTheta = np.sin((j-90)*np.pi/180)
    
    # Calculate y for each x based on the line equation, knowing r and theta
    for x in range(len(imageGrad)):
        
        # Calculating y value using hough to cartesian line conversion
        y = np.round(((i + indent) - (x*xTheta))/yTheta)
        if y >= len(imageGrad[0]):
            break
        if (y>=0):
            
            # Angle <-85 and >85 used to print only vertical lines
            if (j-90)<-85 or (j-90)>85:
                redLines.add((i,j))
                # line thinckness of +/- 2 used for visibility on image
                imageResultRed[x,int(y)-2:int(y)+2] = [0,255,0]
                edges[x,int(y)-3:int(y)+3] = 255
                
            # Angle <-35 and >-55 used to print only diagonal lines
            elif (j-90)>-55 and (j-90)<-35:
                blueLines.add((i,j))
                # line thinckness of +/- 2 used for visibility on image
                imageResultBlue[x,int(y)-2:int(y)+2] = [0,255,0]
                edges[x,int(y)-3:int(y)+3] = 255

print("Number of Red lines  : ", len(redLines))
print("Number of Blue lines : ", len(blueLines))
cv2.imwrite('Results/red_line_orig.jpg',imageResultRed)
cv2.imwrite('Results/blue_lines_orig.jpg',imageResultBlue)


#-------Circle Detection-------#

# Function to perform erosion on image, given image, structuring element and origin
def erosion(image, struct_elem, origin):
    img_list = []
    image = np.pad(image, int(len(struct_elem)/2),'edge')
    for img_row in range(origin[0], len(image)-len(struct_elem)+origin[0]+1):
        for img_col in range(origin[1], len(image[0])-len(struct_elem[0])+origin[1]+1):
            img_list.append([255 if np.array_equal(image[img_row-origin[0]:img_row-origin[0]+len(struct_elem),
                                         img_col-origin[1]:img_col-origin[1]+len(struct_elem[0])]
                                   , struct_elem) else 0 ])
    return np.array(img_list).reshape(-1,len(image[0])-len(struct_elem[0])+1)

# Function to perform dilation on image, given image, structuring element and origin
def dilation(image, struct_elem, origin):
    img_list = []
    image = np.pad(image, int(len(struct_elem)/2),'edge')
    for img_row in range(origin[0], len(image)-len(struct_elem)+origin[0]+1):
        for img_col in range(origin[1], len(image[0])-len(struct_elem[0])+origin[1]+1):
            img_list.append([0 if np.array_equal(image[img_row-origin[0]:img_row-origin[0]+len(struct_elem),
                                         img_col-origin[1]:img_col-origin[1]+len(struct_elem[0])]
                                   , np.logical_not(struct_elem)) else 255 ])
    return np.array(img_list).reshape(-1,len(image[0])-len(struct_elem[0])+1)

# Function to perform closing on image, given image, structuring element and origin
def closing(image, struct_elem, origin):
    image_dilated = dilation(image, struct_elem, origin)
    return erosion(image_dilated, struct_elem, origin)

# Perform morphology to be able to detect circles better
struct_elem = np.ones((5,5))*255
origin = (2,2)

# Remove lines detected
coinsOnly = imageGrad-edges
coinsOnly[np.where(coinsOnly<255)] = 0

# Get boundary of closing of coinsOnly
closingCoins = closing(coinsOnly, struct_elem, origin)
erosionClosingCoins = erosion(closingCoins, struct_elem, origin)
imageGrad = closingCoins-erosionClosingCoins

imageGradX, imageGradY = np.where(imageGrad == 255)

# Initializing up the accumulator
thetaRange = np.array([i for i in range(0,360)])
rRange = np.array([i for i in range(20,40)])
accumulator = np.array([[[0]*len(imageGrad[0])]*len(imageGrad)]*len(rRange))

# Performing the transform and setting up accumulator
for theta in thetaRange:
    for r in rRange:
        rcosTheta = int(np.round(r*np.cos(theta*np.pi/180)))
        rsinTheta = int(np.round(r*np.sin(theta*np.pi/180)))
        for x, y in zip(imageGradX, imageGradY):
            i = x-rcosTheta
            j = y-rsinTheta
            if (i>0 and j>0 and i<len(imageGrad) and j<len(imageGrad[0])):
                accumulator[r-20,i,j]+=1

imageCopy = image.copy()
# Get only circles with atleast 200 hits in the accumulator cells
goodR, goodI, goodJ = np.where(accumulator>200)

circles = set({})
for r, i, j in zip(goodR, goodI, goodJ):
    
    """ Considering lines within a bin of r-20 and r+20 and goodI-20 and goodI+20 and goodJ-20 and goodJ+20 
        to be one circle to avoid labelling the same line as multiple lines. """
    ind = np.where((goodR>=r-20) & (goodR<r+20) & (goodI<i+20) & (goodI>=i-20) & (goodJ<j+20) & (goodJ>=j-20))
    if goodR[ind].size > 0 or goodI[ind].size > 0 or goodJ[ind].size>0:
        r = int(np.round(np.mean(goodR[ind])))
        i = int(np.round(np.mean(goodI[ind])))
        j = int(np.round(np.mean(goodJ[ind])))
    else:
        continue
        
    circles.add((i,j,r))
    cv2.circle(imageCopy, (j,i), r+20, [0,255,0], thickness=2, lineType=8, shift=0)        

print("Number of coins detected: ", len(circles))
cv2.imwrite('Results/coin.jpg',imageCopy)

