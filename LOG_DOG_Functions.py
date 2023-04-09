from skimage import util
from skimage import *
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import time
#--------------------------------
#RGB to greyscale by Aiyaz
def gray_scale(image):
    shape = image.shape
    r, c = shape[0], shape[1]

    if len(shape) == 2:
        pass

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
    print('Converted RGB to grey scale....')
    return image_grey


#--------------------------------

#--------------------------------
#mask fucntion by Steven and Aizaz
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
    
   #to get discrete values
    mask = np.round(mask*(-const/mask[(size-1)//2,(size-1)//2]))
    sum = 0
    for i in mask:
        for j in i:
            sum += j
    print('sum of mask=', sum)
    # mask = np.array(([ 0, 0,-1, 0, 0], 
    #               [ 0,-1,-2,-1, 0], 
    #               [-1,-2,16,-2,-1],
    #               [ 0, 0,-1, 0, 0], 
    #               [ 0,-1,-2,-1, 0] ))
    print('mask calculated....')
    return mask, sum



def G1(x, y, sig1):
    nom = np.exp(-((x**2)-(y**2))/(2*sig1**2)) 
    denom = np.sqrt(2*np.pi)*sig1
    return (nom/denom)

def G2(x, y, sig2):
    nom = np.exp(-((x**2)-(y**2))/(2*sig2**2)) 
    denom = np.sqrt(2*np.pi)*sig2
    return (nom/denom)

def DoG1(sigma, N):
    l = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            l[i,j] = G1((i-(N-1)/2),(j-(N-1)/2), sigma)
    return l

def DoG2(sigma, N):
    l = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            l[i,j] = G2((i-(N-1)/2),(j-(N-1)/2), sigma)
    return l


def DoG_mask(size, sigma1, sigma2 , const):

    mask1 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            num = np.exp(-(((i-(size-1)/2)**2)-((j-(size-1)/2)**2))/(2*sigma1**2))
            denum = np.sqrt(2*np.pi)*sigma1
            mask1[i,j] = (num/denum)
    
    mask2 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            num = np.exp(-(((i-(size-1)/2)**2)-((j-(size-1)/2)**2))/(2*sigma2**2))
            denum = np.sqrt(2*np.pi)*sigma2
            mask2[i,j] = (num/denum)
    
    # To get descrete values
    mask = np.round((mask1 - mask2)*const,2)
    print(mask)
    # mask = np.round(mask*(-const/mask[(size-1)//2,(size-1)//2]))
    return mask

#--------------------------------


#--------------------------------
#image padding by Anish

def padding(image, mask_size):
    S = image.shape
    extras = int((mask_size-1)/2)
    image_pdd=cv2.copyMakeBorder(image, extras, extras, extras, extras,cv2.BORDER_CONSTANT, value=[0,0])
    print('Padding done....')
    return image_pdd

#--------------------------------


#--------------------------------
#Applying mask by Akarsh
def apply_mask(image,mask, return_dict, x):
    global idnid
    pos = image[1]
    image = image[0]
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
    
    return_dict[x] = [img_log, pos]
    # return img_log


#--------------------------------


#--------------------------------
# Histogram by Ramesh
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

# plt.title("Histogram", fontsize = 24)
# plt.xlabel("grayscale values", fontstyle = 'oblique')
# plt.ylabel("number of repeated grayscale values",fontstyle = 'oblique')
# plt.bar(x, y, width = 0.5)
# plt.show()

def divide(img, mask_size):
    r,c = img.shape
    if c%2 !=0:
        img = np.delete(img, c-1, axis=1)


    mask_cut = (mask_size - 1) // 2

    #################dividing#################

# dividing into 2 parts

    height = img.shape[0]
    width = img.shape[1]

    width_cutoff = width // 2

    left1 = img[:, :width_cutoff + mask_cut]
    right1 = img[:, width_cutoff - mask_cut:]

# dividing into 4 parts

# first part

    height = left1.shape[0]
    width = left1.shape[1]

    width_cutoff = width // 2

    left11 = left1[:, :width_cutoff + mask_cut]
    right12 = left1[:, width_cutoff - mask_cut:]

# second part

    height = right1.shape[0]
    width = right1.shape[1]

    width_cutoff = width // 2

    left21 = right1[:, :width_cutoff + mask_cut]
    right22 = right1[:, width_cutoff - mask_cut:]

# dividing into 8 parts

# first part

    height = left11.shape[0]
    width = left11.shape[1]

    width_cutoff = width // 2

    l41 = left11[:, :width_cutoff + mask_cut]
    r41 = left11[:, width_cutoff - mask_cut:]


# second part

    height = right12.shape[0]
    width = right12.shape[1]

    width_cutoff = width // 2

    l42 = right12[:, :width_cutoff + mask_cut]
    r42 = right12[:, width_cutoff - mask_cut:]

# third part

    height = left21.shape[0]
    width = left21.shape[1]

    width_cutoff = width // 2

    l43 = left21[:, :width_cutoff + mask_cut]
    r43 = left21[:, width_cutoff - mask_cut:]


# forth part

    height = right22.shape[0]
    width = right22.shape[1]

    width_cutoff = width // 2

    l44 = right22[:, :width_cutoff + mask_cut]
    r44 = right22[:, width_cutoff-mask_cut:]

    return [[l41,0],[r41,1],[l42,2],[r42,3],[l43,4],[r43,5],[l44,6],[r44,7]]

def join_img(parts, mask_size):

    l = [0]*8
    for i in range(8):
        l[parts[i][1]] = parts[i][0]
        
    img_joined = cv2.hconcat([l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]])    
    return img_joined 
#--------------------------------

#--------------------------------
#binary image using zero crossing by Sakshi
def zero_crossing(image, threshold):

    img_zeroc = image < threshold
    img_zeroc = img_zeroc*255
    return img_zeroc
    
#--------------------------------

#--------------------------------
# Final  Log function function
def apply_LOG(image, mask_size, const, sigma):
    start = time.time()

    image_grey = gray_scale(image) 
    mask, sum = log_mask(mask_size, sigma, const)
    print(mask)
    image_gry_pdd = padding(image_grey, mask_size)
    
    # dividing the image into 8 parts for multiprocessing 
    parts = divide(image_gry_pdd, mask_size) # gives a list of 8 equal part of the image
    

    #-----------MULTIPROCESSING-----------#
    # by Akarsh and Ramesh

    #To store all return values of apply_mask function 
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Assigning each part to a a process which will run on it's assigned cpu, all 8 parts are going to run simultaneoulsy 
    pro = []
    for x in range(len(parts)):
        p = multiprocessing.Process(target=apply_mask, args=(parts[x], mask, return_dict, x))
        p.start()
        pro.append(p)
    
    # To wait for all parts to join back
    for y in pro:
        y.join()

    # Extracting values from return dictionary and joining the parts 
    parts_log = return_dict.values()
    img_log = join_img(parts_log, mask_size)

    stop = time.time()
    print('LOG applied to the image.....')
    print("execution time : ",np.round((stop - start)), "s")
    return img_log, mask,sum
    # return parts_log

def apply_DOG(image, mask_size, const, sigma1, sigma2):
    start = time.time()
    
    image_grey = gray_scale(image) 
    mask = np.fliplr(DoG_mask(mask_size, sigma1, sigma2, const))
    image_gry_pdd = padding(image_grey, mask_size)
    # dividing the image into 8 parts for multiprocessing 
    parts = divide(image_gry_pdd, mask_size) # gives a list of 8 equal part of the image
    

    # #-----------MULTIPROCESSING-----------#
    # #To store all return values of apply_mask function 
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Assigning each part to a a process which will run on it's assigned cpu, all 8 parts are going to run simultaneoulsy 
    pro = []
    for x in range(len(parts)):
        p = multiprocessing.Process(target=apply_mask, args=(parts[x], mask, return_dict, x))
        p.start()
        pro.append(p)
    
    # To wait for all parts to join back
    for y in pro:
        y.join()

    # Extracting values from return dictionary and joining the parts 
    parts_log = return_dict.values()
 
    img_log = join_img(parts_log, mask_size)

    # To calculate sum of mask
    sum = 0
    for i in mask:
        for j in i:
            sum += j
    print('sum of mask=', sum)

    r,c = img_log.shape
    print('DOG applied to the image.....')

    # img_log = img_log - np.min(img_log)
    # img_log = (img_log/np.max(img_log))*255
    
    print('DOG applied to the image.....')
    stop = time.time()
    print("execution time : ",np.round((stop - start)), "s")
    return img_log, mask, sum
 
#--------------------------------