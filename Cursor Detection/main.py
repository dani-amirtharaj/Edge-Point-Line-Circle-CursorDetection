
# Setting up the program
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Read images from the chosen folder, and save the template in a different variable
def getImagesFromFolder(set="A"):
    imagesList = []
    template=[]
    
    if set == "A":
        folder='Images/Set A'  
        
        for fileName in os.listdir(folder):
            if fileName == "template_new.png":
                template.append(cv2.imread(os.path.join(folder,fileName)))
            if "neg" in fileName:
                image = cv2.imread(os.path.join(folder,fileName))
                imagesList.append([image,'Negative'])
            if "pos" in fileName:
                image = cv2.imread(os.path.join(folder,fileName))
                imagesList.append([image,'Positive'])
        return imagesList,template
    
    elif set == "B":
        folder='Images/Set B'  
        
        for fileName in os.listdir(folder):
            for i in range(3):
                if fileName == "t"+str(i+1)+".png":
                    template.append(cv2.imread(os.path.join(folder,fileName)))
            if "neg" in fileName:
                image = cv2.imread(os.path.join(folder,fileName))
                imagesList.append([image,'Negative'])
            else:
                for i in range(3):
                    if "t"+str(i+1)+"_" in fileName:
                        image = cv2.imread(os.path.join(folder,fileName))
                imagesList.append([image,"t"+str(i+1)])
        return imagesList,template

# Function to return sobel edge magnitude of image
def getSobel(image):
    imageX=cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    imageY=cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    imageMagnitude = np.sqrt(imageX ** 2 + imageY ** 2)
    imageMagnitude /= np.max(imageMagnitude)
    return imageMagnitude

# Function to apply SIFT to image, given a template 
def applySIFT(image,template):
    sift = cv2.xfeatures2d.SIFT_create()
    keypointsImage, descriptorImage = sift.detectAndCompute(image1,None)
    keypointsTemplate, descriptorTemplate = sift.detectAndCompute(template,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptorImage,descriptorTemplate, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    imagePlot = cv2.drawMatchesKnn(image,keypointsImage,
                             template,keypointsTemplate,good,None,flags=2)
    plt.figure(figsize=(20,10)),plt.imshow(imagePlot),plt.show()
    
# Function to return difference of gaussian of image
def getDoG(image,kernelSize=3,sigma1=1, sigma2=(2**0.5)):
    return(cv2.GaussianBlur(image,(kernelSize,kernelSize),
                sigma1)-cv2.GaussianBlur(image,(kernelSize,kernelSize),sigma2))
    
# Function to return Laplacian of Gaussian of image
def getLoG(image):
    blur=cv2.GaussianBlur(image,(3,3),0)
    return cv2.Laplacian(blur,cv2.CV_32F)

def cursorDetection(set="A", THRESHOLD=0.65, method="LoG"):
    # Choose from set A or B
    images,template=getImagesFromFolder(set)

    IMAGE_INDEX=0
    LABEL_INDEX=1

    # Initializing lists to store final image after detection, location of all 
    # maxima and value of all maxima pixels after template matching
    imageResult=[]
    corrLocAll=[]
    corrMaxAll=[]


    for imgIndex in range(len(images)):

        image=images[imgIndex][IMAGE_INDEX].copy()
        label=images[imgIndex][LABEL_INDEX]
        corrLocTemp=[]
        corrMaxTemp=[]
        
        for tempIndex in range(len(template)):

            corrMax=[]
            resizedTemplate=[]
            corrScales=[]                                                                                                                                                                                                                                                                                                                                                                                                                       
            corrLoc=[]
            
            if method == "Sobel":
                #Applying sobel filter
                templateTransform=getSobel(cv2.cvtColor(
                    template[tempIndex],cv2.COLOR_BGR2GRAY))
                imageTransform=getSobel(cv2.cvtColor(
                    image,cv2.COLOR_BGR2GRAY))
            
            if method == "LoG":
                #Applying lapalcian of gaussian filter
                templateTransform=getLoG(cv2.cvtColor(
                    template[tempIndex],cv2.COLOR_BGR2GRAY))
                imageTransform=getLoG(cv2.cvtColor(
                    image,cv2.COLOR_BGR2GRAY))
            
            if method == "DoG":
                #Applying difference of gaussian filter
                templateTransform=getDoG(cv2.cvtColor(
                    template[tempIndex],cv2.COLOR_BGR2GRAY))
                imageTransform=getDoG(cv2.cvtColor(
                    image,cv2.COLOR_BGR2GRAY))
                
            if method == "None":
                #Applying no transformation
                templateTransform=template[tempIndex]
                imageTransform=image

            # Resizing template to ensure scale invariance
            templateScale=[x * 0.01 for x in range(80,150)]
            for i in templateScale:
                resizedTemplate.append(cv2.resize(templateTransform, 
                                                  (0,0), fx=i, fy=i))
            
            # Template matchinig for the entire range of resized templates
            for i in range(70):
                corrScales.append(cv2.matchTemplate(
                    imageTransform,resizedTemplate[i],cv2.TM_CCORR_NORMED))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrScales[i])

                corrLoc.append(max_loc)
                corrMax.append(max_val)
            corrMaxTemp.append(corrMax)
            corrLocTemp.append(corrLoc)
        corrMaxAll.append(corrMaxTemp)
        corrLocAll.append(corrLocTemp)

        # Identifying pixels with maximum correlation, and marking them on image
        # Here, the circles are color coded as red - normal cursor, 
        #blue - black cursor, green - hand
        for tempIndex in range(len(corrLocAll[imgIndex])):
            if max(corrMaxAll[imgIndex][tempIndex]) > THRESHOLD:
                R=(0,0,255)
                B=(255,0,0)
                G=(0,255,0)

                circleColor=[G,B,R]
                if set == "A":
                    circleColor[tempIndex]=R
                circleSize=[20,25,15]

                pixelMax=corrLocAll[imgIndex][tempIndex
                                    ][corrMaxAll[imgIndex][tempIndex
                                    ].index(max(corrMaxAll[imgIndex][tempIndex]))]
                pixelOffset=list(pixelMax)
                pixelOffset[0]=pixelOffset[0]+7
                pixelOffset[1]=pixelOffset[1]+10

                image = cv2.circle(image, tuple(pixelOffset), circleSize[
                                    tempIndex], circleColor[tempIndex], 2)
        imageResult.append(image)
    return imageResult, corrMaxAll, corrLocAll

# Main call to all functions, change parameter to call set A or B
imageResult, corrMaxAll, corrLocAll=cursorDetection("B",0.65,"LoG")
i=0
# Output all images one after the other with the detected cursors
for img in imageResult:
    i+=1
    cv2.imwrite("Results/Set B/Output_"+str(i)+".jpg",img)
