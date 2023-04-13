from skimage import *
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from using_gpu import *
# from numba import jit, cuda
image = io.imread("/home/aken/Downloads/remy_loz-xEfKcZt47LI-unsplash.jpg")
Thres = 0
mask_size = 5
constant = 20
variance = 0.75



image_log = 0
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

 
# Sliders
ax_variance = plt.axes([0.85, 0.12, 0.02, 0.85])
sldr_variance = Slider(ax_variance,'σ',0.0001,1.4,0.75,color='green' ,orientation='vertical')
allowed_masksizes = [3,5,7,9,11]
ax_mask_size = plt.axes([0.95, 0.12, 0.02, 0.43]) # x,y,width,height
sldr_mask_size = Slider(ax_mask_size,'M_S',3,11,5,valstep=allowed_masksizes, orientation='vertical')
ax_msk_const = plt.axes([0.9, 0.12, 0.02, 0.6])
sldr_mask_const = Slider(ax_msk_const, 'C', 1, 150,20,orientation='vertical', valstep=[i for i in range(1, 150)])
ax_threshold = plt.axes([0.1, 0.02, 0.8, 0.03])
sldr_threshold = Slider(ax_threshold, 'T', -5000, 5000,0)

ax_reset = plt.axes([0.001, 0.09, 0.1, 0.04])
button = Button(ax_reset, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

# @jit(target_backend='cuda')
def plott(image, mask, summ):
    text_x = -550
    text_y = 330
    font_size = 22
    px = 0.1
    ax.clear()
    pos1 = ax.get_position() # get the original position 
    if len(mask)==11:
        font_size = 15
        text_x = -600
        px = 0.18
        ax.set_position([px, 0.10999999999999999, 0.775, 0.77])
    elif len(mask)==9:
        font_size = 17
        text_x = -600
        px = 0.16
        ax.set_position([px, 0.10999999999999999, 0.775, 0.77])
    elif len(mask) == 7:
        font_size = 17
        px = 0.15
        ax.set_position([px, 0.10999999999999999, 0.775, 0.77])
    # pos2 = [pos1.x0+0.06, pos1.y0 ,  pos1.width / 1, pos1.height / 1] 
    # print(pos2)
     # set a new position
    
    ax.text(text_x, text_y,' mask\n'+str(mask), fontsize = font_size)
    ax.text(text_x, text_y+200, 'sum='+str(summ), fontsize = 22)
    ax.imshow(image, cmap='gray')
    
def update_variance(val):
    # code to update variance #
    global image_log, variance
    variance = sldr_variance.val
    image_log, mask, summ = apply_LOG(image, mask_size, constant, variance)
    image_zc = zero_crossing(image_log, Thres)
    
    plott(image_zc, mask, summ)
    pass

def update_mask_size(val):
    # code to change mask size #
    global image_log, mask_size
    mask_size = sldr_mask_size.val
    image_log, mask, summ = apply_LOG(image, mask_size, constant, variance)
    image_zc = zero_crossing(image_log, Thres)
    plott(image_zc, mask, summ)
    pass

def update_constant(val):
    # code to update mask constant #
    global image_log, constant
    constant = sldr_mask_const.val
    image_log, mask, summ = apply_LOG(image, mask_size, constant, variance)
    image_zc = zero_crossing(image_log, Thres)
    plott(image_zc, mask, summ)
    pass


def update_threshold(val):
    global Thres
    Thres = sldr_threshold.val
    image_zc = zero_crossing(image_log, Thres)
    ax.imshow(image_zc, cmap='gray')
    
def reset(event):
    sldr_mask_size.reset()
    sldr_variance.reset()
    sldr_mask_const.reset()
    sldr_threshold.reset()

button.on_clicked(reset)
sldr_threshold.on_changed(update_threshold)
sldr_mask_const.on_changed(update_constant)
sldr_variance.on_changed(update_variance)
sldr_mask_size.on_changed(update_mask_size)
plt.show()