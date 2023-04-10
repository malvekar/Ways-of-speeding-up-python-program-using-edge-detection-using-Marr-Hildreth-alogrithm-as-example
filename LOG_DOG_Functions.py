from skimage import util
from skimage import *
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import time

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
