from skimage import util
from skimage import *
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import time
from numba import jit, cuda
#--------------------------------
#RGB to greyscale by Aiyaz
@jit(target_backend='cuda')
def gray_scale(image):
    shape = image.shape
    r, c = shape[0], shape[1]

    if len(shape) == 2:
        return image

    # r, c, channel = image.shape
    r_c = image[:,:,0]
    g_c = image[:,:,1]
    b_c = image[:,:,2]
    r_cons = 0.2126
    g_cons = 0.7152
    b_cons =0.0722
    image_grey = (r_cons*r_c + g_cons*g_c + b_cons*b_c)
    for i in range(r):
        for j in range(c):
            image_grey[i][j] = round(image_grey[i][j])

    image_grey = image_grey/np.max(image_grey) # normalizing intensity levels
    image_grey = util.img_as_ubyte(image_grey)
    
    # print('Converted RGB to grey scale....')
    return image_grey


#--------------------------------

#--------------------------------
#mask fucntion by Steven and Aizaz
@jit(target_backend='cuda')
def log_mask(size, sigma , const):
    mask = np.ones((size, size))
    # mask = [0]*size
    # for i in range(size):
    #     mask[i]= [0]*size

    for i in range(size):
        for j in range(size):
            num = (((i-(size-1)/2)**2)+((j-(size-1)/2)**2) - (2*sigma**2))
            denum = (np.pi*2*(sigma**6))
            exp = np.exp(-((i-(size-1)/2)**2 + (j-(size-1)/2)**2)/(2*sigma**2))
            mask[i,j] = (num*exp)/denum
            # mask[i][j] = [i-(size-1)/2,j-(size-1)/2]
    
    # print(mask)
   #to get discrete values
    mask = np.round(mask*(-const/mask[(size-1)//2,(size-1)//2]))
    sum = np.sum(mask)
    # for i in mask:
    #     for j in i:
    #         sum += j
    
    # mask = np.array(([ 0, 0,-1, 0, 0], 
    #               [ 0,-1,-2,-1, 0], 
    #               [-1,-2,16,-2,-1],
    #               [ 0, 0,-1, 0, 0], 
    #               [ 0,-1,-2,-1, 0] ))
    # print('mask calculated....')
    
    if sum != 0:
        mask[(size-1)//2,(size-1)//2 ] -= sum
    sum = np.sum(mask)
    # print('sum of mask=', sum)
    # for i in mask:
    #     for j in i:
    #         sum += j
    return mask, sum





#--------------------------------
#image padding by Anish

def padding(image, mask_size):
    
    S = image.shape
    extras = int((mask_size-1)/2)
    image_pdd=cv2.copyMakeBorder(image, extras, extras, extras, extras,cv2.BORDER_CONSTANT, value=[0,0])
    # print('Padding done....')
    return image_pdd

#--------------------------------


#--------------------------------
#Applying mask by Akarsh
@jit(target_backend='cuda')
def apply_mask(image,mask):
  
    
    r,c = image.shape

    mask_size = len(mask)
    
    extras = int((mask_size-1)/2) # number of extra row and columns on each side
    img_log = np.zeros((r-mask_size+1,c - mask_size + 1)) #creating empty image of original size
    sum = 0

    for i in range(extras,r-extras): 
        for j in range(0,c-extras):
            for k in range(mask_size):
                for l in range(mask_size):
                    sum += int(mask[k][l]*image[i+k-extras][j+l-extras])

            img_log[i-extras][j-extras] = (image[i][j]+sum)
            sum = 0

    return img_log


#--------------------------------


#--------------------------------

#--------------------------------
#binary image using zero crossing by Sakshi
def zero_crossing(image, threshold):

    img_zeroc = image < threshold
    img_zeroc = img_zeroc*255
    return img_zeroc
    
#--------------------------------
@jit(target_backend='cuda')
def histogram(img_log):
    dupRemoved = []
    for x in img_log: #looping through outer list elements
        for y in x:    #looping through inner list elements
            if y not in dupRemoved: #removing duplicates and storing graycale values in list
                dupRemoved.append(y)

    sList = sorted(dupRemoved) #list elements in ascending order
    print('Histogram- duplicate removed.....')
# list for storing number of times the greyscale values repeated
    num = []

    for x in sList: 
        count = 0
        for y in img_log:
            for z in y:
                if x == z:
                    count += 1
        num.append(count)

    print('storing number of times the greyscale values repeated.....')
    num = np.array(num)
    sList = np.array(sList)

    print('Histogram Calculated....')

    return [sList,num]
#--------------------------------
# Final  Log function function
@jit(target_backend='cuda')
def apply_LOG(image, mask_size, const, sigma):
    start = time.time()

    image_grey = gray_scale(image) 
    mask, sum = log_mask(mask_size, sigma, const)
    print(mask)
    image_gry_pdd = padding(image_grey, mask_size)
    
    img_log = apply_mask(image_gry_pdd, mask)
    stop = time.time()
    print('LOG applied to the image.....')
    print("execution time : ",np.round((stop - start)), "s")
    return img_log, mask,sum
    # return parts_log


#--------------------------------