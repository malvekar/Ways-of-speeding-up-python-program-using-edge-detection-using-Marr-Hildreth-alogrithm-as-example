"""
Author - Steven Vazhappully
Email - steventambi31@gmail.com
"""

import numpy as np

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
            mask[i,j] = (num*exp)/denum                                         #Generates a mask using the equations for Laplacian of Gaussian.
            # mask[i][j] = [i-(size-1)/2,j-(size-1)/2]                          
    
    # print(mask)
    
    mask = np.round(mask*(-const/mask[(size-1)//2,(size-1)//2]))                #Discretises the values of the generated mask.
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

#To view a generated mask, use a print statement and call the log_mask function with the desired value of size and sigma.