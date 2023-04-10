import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('/home/aken/College/dip/log_marr_hildreth/FINAL/edgeflower.jpg', 0)

def divide_image(img, mask_size):
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

parts = divide_image(img, 5)



# mapping

# part_dict =dict(l41 = 0, r41 = 1, l42 = 2, r42 = 3, l43 = 4, r43 = 5, l44 = 6, r44 = 7)

# print(part_dict[l41])

#################joining#################

def join_image(parts, mask_size):
    l = [0]*8
    mask_cut = (mask_size - 1) // 2
    cut = []
    for i in range(1,mask_cut+1):
        cut.append(-i)
    print(cut)   
    for i in range(8):
        parts[i][0] = np.delete(parts,cut, axis=1 )
        
    for i in range(8):
        print(parts[i][0].shape)      
    for i in range(8):
        l[parts[i][1]] = parts[i][0]
        
    img_joined = cv2.hconcat([l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]])    
    return img_joined    
    # img_joined = cv2.hconcat([l41, r41, l42, r42, l43, r43, l44, r44])

image_joined = join_image(parts, 5)

plt.imshow(parts[7][0], cmap='gray')
