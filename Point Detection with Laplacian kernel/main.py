
# Importing necessary packages for this project
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setting seed for reproducibility
UBIT = 'damirtha'
np.random.seed(sum([ord(c) for c in UBIT]))


# Function to apply a mask on an image
def pointMask(image, mask):
    img_list = []
    for img_row in range(int(len(mask)/2), len(image)-int(len(mask)/2)):
        for img_col in range(int(len(mask[0])/2), len(image[0])-int(len(mask[0])/2)):
            img_list.append(np.mean(np.multiply(image[img_row-int(len(mask)/2):img_row+int(len(mask)/2)+1,
                                         img_col-int(len(mask[0])/2):img_col+int(len(mask[0])/2)+1]
                                   , mask)))
    return np.pad(np.array(img_list).reshape(-1,len(image[0])-len(mask[0])+1), int(len(mask)/2),'edge')

image=cv2.imread('Images/point.jpg', cv2.IMREAD_GRAYSCALE)

# Laplacian mask for point detection
mask = -np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
masked_image = pointMask(image, mask)

# Thresholding the masked image to get good points
x, y = np.where(masked_image>np.max(masked_image)*0.9)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
print("Points detected: ")
for i, j in zip(x, y):
        cv2.circle(image, (j,i), 5, [0,0,255], thickness=2, lineType=8, shift=0)
        print((i,j))
cv2.imwrite('Results/res_point.jpg',image)