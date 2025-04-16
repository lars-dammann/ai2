from os.path import isfile, join
import tensorflow as tf

assert tf.__version__.startswith('2')
from os import listdir
import numpy as np
import segmentation_models as sm
import cv2

sm.set_framework('tf.keras')
sm.framework()
from PIL import Image
import tensorflow as tf
from scipy import ndimage
import pandas as pd

import copy
patch_size = (256, 256)
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

c1 = [(965, 950),
    (990, 925),
    (955, 960),
    (1045, 930),
    (935, 750),
    (965, 770),
    (1010, 825),
    (980, 830),
    (892, 775),
    (952, 777),
    (990, 790),
    (937, 812),
    (1002, 884),
    (910, 837),
    (967, 815),
    (920, 820),
    (975, 715),
    (915, 757),
    (990, 800),
    (896, 817),
    (1035, 927),
    (940, 895),
    (1020, 860),
    (893, 840),
    (980, 876),
    (1105, 910),
    (1143, 970),
    (1065, 973),
    (850, 862),
    (915, 880),
    (995, 870),
    (910, 917),
    (735, 760),
    (785, 800),
    (960, 860),
    (965, 910),
    (930, 790),
    (970, 820),
    (1060, 860),
    (980, 890),
    (840, 800),
    (890, 840),
    (1055, 860),
    (1180, 850),
    (910, 750),
    (950, 790),
    (1100, 845),
    (1140, 830),
    (975, 895),
    (1055, 860),
    (1225, 830),
    (1225, 835),
    (970, 1000),
    (1050, 965),
    (1080, 920),
    (970, 910),
    (1045, 850),
    (1140, 830),
    (1165, 815),
    (1110, 810),
    (925, 995),
    (1140, 940),
    (1120, 875),
    (930, 860),
    (975, 840),
    (975, 820),
    (947, 820),
    (1117, 797),
    (987, 865),
    (982, 855),
    (1022, 825),
    (1105, 830),
    (990, 820),
    (1015, 815),
    (1010, 817),
    (1112, 805),
    (875, 827),
    (892, 835),
    (930, 820),
    (980, 845),
    (1030, 810),
    (985, 810),
    (915, 855),
    (977, 875),
    (895, 905),
    (970, 880),
    (995, 830),
    (1027, 807),
    (900, 830),
    (810, 815),
    (955, 870),
    (920, 890),
    (990, 870),
    (885, 885),
    (1025, 880),
    (975, 895),
    (875, 790),
    (780, 740),
    (910, 735),
    (852, 747),
    (865, 790),
    (790, 800),
    (902, 810),
    (900, 760),
    (940, 845),
    (910, 865),
    (940, 840),
    (975, 810),
    (1070, 850),
    (1075, 820),
    (1045, 850),
    (1040, 840),
    (1032, 725),
    (1052, 717),
    (1025, 725),
    (1035, 705),
    (1010, 805),
    (1040, 790),
    (987, 755),
    (1005, 720),
    (985, 895),
    (1017, 912),
    (970, 902),
    (980, 880),
    (980, 880),
    (1010, 880),
    (965, 860),
    (970, 850),
    (1060, 910),
    (995, 860),
    (970, 835),
    (960, 775),
    (1005, 740),
    (1015, 760),
    (1060, 785),
    (1040, 785),
    (1005, 700),
    (980, 710),
    (1005, 690),
    (980, 720),
    (1000, 742),
    (1000, 730),
    (1000, 730),
    (910, 790),
    (990, 725),
    (1000, 705),
    (980, 670),
    (900, 685),
    (975, 880),
    (980, 875),
    (965, 855),
    (780, 830),
    (950, 925),
    (955, 900),
    (975, 855),
    (735, 795),
    (890, 850),
    (942, 920),
    (965, 835),
    (930, 850),
    (882, 910),
    (925, 880),
    (930, 780),
    (1040, 810),
    (1010, 830),
    (860, 780),
    (950, 840),
    (870, 800),
    (955, 800),
    (900, 820),
    (900, 750),
    (920, 830),
    (1035, 910),
    (1010, 920),
    (910, 865),
    (1000, 810),
    (970, 910),
    (1020, 910),
    (910, 870),
    (960, 825),
    (840, 780),
    (890, 820),
    (770, 847),
    (840, 880),
    (835, 805),
    (920, 790),
    (840, 795),
    (840, 850),
    (840, 770),
    (890, 860),
    (890, 810),
    (930, 810),
    (800, 820),
    (790, 800),
    (860, 840),
    (910, 740),
    (830, 870),
    (910, 940),
    (880, 870),
    (940, 810),
    (840, 900),
    (830, 870),
    (830, 790),
    (870, 830),
    (1015, 895),
    (920, 860),
    (940, 805),
    (1050, 755),
    (970, 850),
    (890, 880),
    (900, 890),
    (1040, 890),
    (880, 930),
    (840, 940),
    (850, 940),
    (1000, 905),
    (830, 870),
    (800, 845),
    (815, 805),
    (1050, 730),
    (750, 920),
    (705, 885),
    (770, 885),
    (1000, 850),
    (940, 870),
    (890, 840),
    (730, 780),
    (1000, 705),
    (950, 775),
    (950, 865),
    (950, 800),
    (1090, 770),
    (925, 770),
    (875, 775),
    (930, 770),
    (1145, 880),
    (975, 800),
    (905, 805),
    (1040, 940),
    (1140, 930),
    (1040, 800),
    (960, 810),
    (1000, 760),
    (1160, 880),
    (1080, 755),
    (950, 780),
    (960, 770),
    (1120, 755),
    (1110, 870),
    (1100, 890),
    (1030, 870),
    (1060, 860),
    (980, 740),
    (1020, 860),
    (1010, 770),
    (1000, 815),
    (950, 810),
    (990, 810),
    (1010, 775),
    (1070, 845),
    (985, 750),
    (965, 750),
    (1070, 800),
    (1055, 815),
    (1090, 845),
    (1090, 785),
    (1060, 840),
    (1100, 870),
    (1010, 780),
    (1070, 830),
    (950, 850),
    (1060, 860),
    (1030, 790),
    (1010, 855),
    (1040, 790),
    (1035, 805),
    (920, 820),
    (940, 795),
    (1080, 780),
    (1020, 720),
    (850, 770),
    (940, 765),
    (830, 770),
    (980, 730),
    (830, 755),
    (905, 745),
    (800, 755),
    (905, 740),
    (880, 755),
    (980, 765),
    (915, 715),
    (870, 790),
    (705, 770),
    (835, 745),
    (910, 700),
    (805, 750),
    (690, 847),
    (860, 800),
    (840, 730),
    (805, 710),
    (1070, 865),
    (1155, 840),
    (1223, 800),
    (875, 815),
    (1080, 850),
    (0, 0),
    (1210, 750),
    (915, 715),
    (935, 720),
    (1060, 840),
    (1190, 785),
    (1170, 750),
    (900, 910),
    (770, 855),
    (915, 805),
    (945, 750),
    (985, 850),
    (910, 845),
    (1045, 795),
    (1060, 720),
    (1075, 910),
    (1065, 965),
    (1155, 950),
    (1185, 925),
    (1125, 995),
    (1090, 1000),
    (1210, 985),
    (960, 925),
    (1080, 990),
    (1165, 730),
    (1170, 960),
    (1115, 905),
    (1075, 770),
    (0, 0),
    (1150, 960),
    (1185, 922),
    (1085, 790),
    (1010, 760),
    (1110, 995),
    (1145, 965),
    (1085, 940),
    (0, 0),
    (1075, 905),
    (1135, 870)]

c2 =[(977, 978),
    (975, 930),
    (945, 965),
    (1040, 930),
    (937, 748),
    (955, 775),
    (1000, 830),
    (985, 850),
    (888, 798),
    (942, 795),
    (998, 785),
    (953, 838),
    (1015, 910),
    (947, 832),
    (970, 795),
    (945, 805),
    (985, 720),
    (904, 753),
    (985, 785),
    (890, 813),
    (1022, 930),
    (945, 890),
    (1035, 872),
    (930, 845),
    (970, 884),
    (1115, 920),
    (1142, 972),
    (1073, 965),
    (835, 860),
    (915, 895),
    (985, 855),
    (905, 917),
    (730, 760),
    (790, 800),
    (950, 850),
    (967, 912),
    (920, 800),
    (975, 820),
    (1070, 860),
    (980, 880),
    (830, 810),
    (890, 840),
    (1055, 840),
    (1170, 850),
    (910, 755),
    (950, 800),
    (1100, 845),
    (1130, 820),
    (980, 895),
    (1065, 870),
    (1200, 840),
    (1225, 855),
    (950, 990),
    (1060, 950),
    (1095, 910),
    (960, 910),
    (1040, 850),
    (1140, 830),
    (1165, 815),
    (1100, 810),
    (910, 1000),
    (1145, 935),
    (1115, 880),
    (920, 860),
    (975, 840),
    (975, 820),
    (942, 810),
    (1115, 800),
    (985, 865),
    (967, 837),
    (1030, 825),
    (1120, 835),
    (990, 810),
    (1015, 800),
    (1010, 817),
    (1110, 802),
    (875, 827),
    (892, 835),
    (945, 817),
    (987, 845),
    (1032, 805),
    (985, 805),
    (920, 847),
    (985, 872),
    (905, 905),
    (968, 870),
    (1000, 835),
    (1020, 800),
    (890, 830),
    (805, 810),
    (955, 870),
    (915, 885),
    (995, 880),
    (885, 870),
    (1025, 875),
    (980, 890),
    (865, 795),
    (780, 745),
    (910, 730),
    (850, 740),
    (870, 770),
    (790, 800),
    (890, 797),
    (900, 760),
    (945, 845),
    (907, 872),
    (935, 845),
    (975, 805),
    (1060, 857),
    (1070, 830),
    (1030, 850),
    (1025, 840),
    (1022, 725),
    (1045, 727),
    (1015, 735),
    (1025, 710),
    (1000, 817),
    (1017, 780),
    (980, 755),
    (995, 730),
    (987, 915),
    (1012, 925),
    (965, 907),
    (960, 905),
    (980, 895),
    (995, 880),
    (955, 865),
    (970, 855),
    (1035, 915),
    (990, 870),
    (960, 840),
    (955, 785),
    (1005, 745),
    (1025, 755),
    (1065, 775),
    (1050, 775),
    (1000, 690),
    (1030, 695),
    (1005, 690),
    (1030, 695),
    (1015, 722),
    (1010, 730),
    (1010, 730),
    (915, 780),
    (1005, 725),
    (990, 685),
    (980, 670),
    (900, 690),
    (965, 885),
    (980, 860),
    (965, 850),
    (780, 830),
    (950, 920),
    (965, 900),
    (975, 855),
    (735, 795),
    (890, 850),
    (942, 915),
    (950, 845),
    (930, 845),
    (882, 905),
    (917, 872),
    (930, 780),
    (1040, 810),
    (1000, 830),
    (860, 780),
    (950, 840),
    (870, 800),
    (955, 800),
    (900, 820),
    (900, 750),
    (920, 835),
    (1035, 910),
    (1010, 920),
    (910, 865),
    (980, 810),
    (970, 905),
    (1020, 910),
    (910, 875),
    (960, 825),
    (840, 780),
    (900, 815),
    (770, 847),
    (840, 880),
    (835, 805),
    (920, 790),
    (840, 795),
    (840, 850),
    (840, 770),
    (890, 840),
    (890, 810),
    (930, 810),
    (800, 810),
    (790, 780),
    (860, 840),
    (900, 740),
    (830, 870),
    (920, 950),
    (880, 850),
    (950, 800),
    (840, 900),
    (830, 870),
    (830, 790),
    (870, 830),
    (1015, 895),
    (920, 860),
    (940, 805),
    (1050, 755),
    (970, 850),
    (870, 880),
    (890, 890),
    (1040, 890),
    (880, 930),
    (840, 940),
    (850, 940),
    (1000, 905),
    (830, 870),
    (800, 845),
    (815, 805),
    (1050, 730),
    (750, 920),
    (705, 885),
    (770, 885),
    (980, 860),
    (930, 870),
    (880, 830),
    (730, 780),
    (990, 715),
    (950, 775),
    (950, 865),
    (940, 800),
    (1090, 770),
    (925, 770),
    (875, 775),
    (930, 760),
    (1145, 880),
    (975, 800),
    (905, 805),
    (1040, 940),
    (1140, 910),
    (1040, 800),
    (970, 810),
    (1000, 760),
    (1160, 880),
    (1080, 755),
    (950, 780),
    (960, 770),
    (1120, 755),
    (1110, 870),
    (1100, 890),
    (1030, 870),
    (1060, 855),
    (980, 740),
    (1020, 845),
    (1010, 770),
    (1000, 815),
    (950, 810),
    (1000, 805),
    (1010, 775),
    (1070, 845),
    (985, 750),
    (965, 760),
    (1080, 800),
    (1055, 815),
    (1090, 840),
    (1090, 780),
    (1060, 855),
    (1100, 870),
    (1010, 780),
    (1070, 835),
    (950, 850),
    (1060, 860),
    (1030, 790),
    (1010, 855),
    (1040, 790),
    (1030, 805),
    (920, 820),
    (950, 780),
    (1080, 780),
    (1020, 720),
    (850, 770),
    (950, 775),
    (830, 770),
    (980, 710),
    (830, 755),
    (920, 770),
    (815, 775),
    (915, 760),
    (880, 755),
    (980, 765),
    (915, 715),
    (840, 800),
    (705, 770),
    (855, 750),
    (850, 730),
    (805, 775),
    (700, 855),
    (860, 810),
    (840, 730),
    (815, 720),
    (1070, 865),
    (1140, 830),
    (1223, 815),
    (875, 835),
    (1075, 865),
    (0, 0),
    (1215, 755),
    (940, 715),
    (935, 730),
    (1060, 850),
    (1190, 785),
    (1180, 730),
    (890, 900),
    (770, 855),
    (915, 805),
    (930, 740),
    (990, 830),
    (915, 840),
    (1050, 770),
    (1060, 710),
    (1080, 930),
    (1065, 965),
    (1155, 950),
    (1190, 920),
    (1120, 1000),
    (1090, 1000),
    (1210, 985),
    (0, 0),
    (1080, 990),
    (1160, 750),
    (1175, 950),
    (1140, 905),
    (1075, 790),
    (0, 0),
    (1100, 990),
    (1132, 967),
    (1082, 785),
    (1005, 770),
    (1140, 965),
    (1180, 920),
    (1085, 940),
    (0, 0),
    (1075, 910),
    (1120, 880)]

c1 = np.array(c1)
c2 = np.array(c2)

resize_coeff=1

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def mask(img, c, rad=670):
    mean_val = np.array([el for el in img.flatten() if el >= 0 and el < 0.1]).mean()
    if len(img.shape)==3:
        mean_val = np.array([el for el in img.reshape(-1, img.shape[-1]) ]).mean(0)
    mask = create_circular_mask(img.shape[0], img.shape[1], center=c, radius=rad)
    new_img = copy.deepcopy(img)
    new_img[~mask] = mean_val
    new_img = new_img[c[1]-rad:c[1]+rad, c[0]-rad:c[0]+rad]
    pad = ((96, 96), (96, 96))
    if len(img.shape)==3 and img.shape[2]==3:
        r_, g_, b_ = new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2]
        rb = np.pad(array=r_, pad_width=pad, mode='constant', constant_values=mean_val[0])
        gb = np.pad(array=g_, pad_width=pad, mode='constant', constant_values=mean_val[1])
        bb = np.pad(array=b_, pad_width=pad, mode='constant', constant_values=mean_val[2])
        new_img = np.dstack(tup=(rb, gb, bb))
    elif len(img.shape)==3 and img.shape[2]==1:
        new_img = np.pad(new_img[:, :, 0], mode='constant', pad_width=pad, constant_values=mean_val)
        new_img = np.expand_dims(new_img, axis=2)
    else:
        new_img = np.pad(new_img, mode='constant', pad_width=pad, constant_values=mean_val)
    return new_img

def patches(img, patch_size=patch_size, step=128):
    res_patches = []
    x_tops = []
    y_tops = []
    if step is None:
        for i in range(1, img.shape[0] // patch_size[0] * img.shape[1] // patch_size[1]-1):
            x_num = i // (img.shape[0] // patch_size[0])
            y_num = i % (img.shape[0] // patch_size[0])
            #print("For the patch {0} num_x is {1} and num_y is {2}".format(i, x_num, y_num))
            patch_temp = img[
                         patch_size[1] * y_num:patch_size[1] * y_num + patch_size[1],
                         patch_size[0] * x_num:patch_size[0] * x_num + patch_size[0]]
            if patch_temp.shape[0] == patch_size[0] and patch_temp.shape[1] == patch_size[1]:
                res_patches.append(patch_temp)
    else:
        x_top = 0
        y_top = 0
        i =0
        while y_top <= img.shape[1] - patch_size[1]:
            while x_top<= img.shape[0]-patch_size[0]:
                #print("For the patch {0} num_x is {1} and num_y is {2}".format(i, x_top, y_top))
                patch_temp = img[
                             y_top:y_top + patch_size[1],
                             x_top:x_top + patch_size[0]]
                if patch_temp.shape[0] == patch_size[0] and patch_temp.shape[1] == patch_size[1]:
                    res_patches.append(patch_temp)
                x_tops.append(x_top)
                y_tops.append(y_top)
                i=i+1
                x_top = x_top+step
            x_top = 0
            y_top = y_top + step
    return res_patches#, x_tops, y_tops

def patches_overlap_reconstruct(preds, orig_heat, patch_size=patch_size, step=128):
    nums_matrix = np.full((orig_heat.shape[0], orig_heat.shape[1]), 0)
    vals_matrix = np.full((orig_heat.shape[0], orig_heat.shape[1]), 0.0)
    for i in range(len(preds)):
        x_top = step*(i % ((orig_heat.shape[1]-patch_size[1]) // step+1))
        y_top = step*(i // ((orig_heat.shape[0]-patch_size[0])// step+1))
        nums_matrix[y_top:y_top + patch_size[1],
                             x_top:x_top + patch_size[0]] += 1
        vals_matrix[y_top:y_top + patch_size[1], x_top:x_top + patch_size[0]] += preds[i][:, :, 0]
    vals_matrix = vals_matrix/np.clip(nums_matrix, a_min=1, a_max=len(preds))
    return vals_matrix

def interp(data, method=None, filename=None, resize_coeff=None, pad=((0, 0), (0, 0))):
    try:
        mask = np.isnan(data)
        if method is None:
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            if resize_coeff is not None:
                data = np.pad(data, mode='constant', pad_width=pad, constant_values=0)
                data = ndimage.zoom(data, 1.0 / resize_coeff)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!The problem is {e}")
        print(f"The type of data is {type(data)}, and data itself is {data} in file {filename}")
    return np.expand_dims(data, axis=2) # data


def resize_profilometer(filepath, resize_coeff, greysc=False):
    pic_profilometer_temp = Image.fromarray(cv2.imread(filepath)[:1664, :1792])
    width_temp, height_temp = pic_profilometer_temp.size
    pic_profilometer_temp= pic_profilometer_temp.resize(
        (int(width_temp / resize_coeff), int(height_temp / resize_coeff)))
    pic_profilometer_temp = np.array(pic_profilometer_temp)
    return pic_profilometer_temp / 255
def norm_01(X, mean_val=0, std_val=1, min_val=-0.03, max_val=0.01, par=0):
    if par==0:
        X_std = (X - min_val) / (max_val - min_val)
    if par==1:
        X_std = (X - mean_val) / std_val
    return X_std

def return_back(X, mean_val=0, std_val=1, min_val=-0.424, max_val=0.089, par=0):
    if par==0:
        X_std = X*(max_val - min_val)+min_val
    if par==1:
        X_std = X*std_val + mean_val
    return X_std

class relative_MAE(tf.losses.Loss):

    def __init__(self):
        super(relative_MAE, self).__init__()
        self.mae = tf.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, y_true, y_pred):
        mae_temp = K.abs(y_true - y_pred) / (K.abs(y_true) + 0.0001)
        mae_rel = tf.reduce_mean(mae_temp) * 100
        return mae_rel


def RevHub_loss(labels, predictions, c = 0.1):
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")

    # Make sure shape do match
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    # Get absolute error for each pixel in batch
    abs_error = tf.abs(tf.subtract(predictions, labels), name='abs_error')
    c = c * tf.reduce_max(abs_error)
    RevHub_loss = tf.where(abs_error <= c,
                          abs_error,
                          (tf.square(abs_error) + tf.square(c)) / (2 * c))

    loss = tf.reduce_mean(RevHub_loss)

    return loss

def get_volume(hmap, pixsize=(7.406, 7.406, 0.1)):
    h, w = hmap.shape[:2]
    mask = create_circular_mask(h, w)
    masked_img = hmap.copy()
    masked_img[~mask] = 0
    all_hmap = (np.clip(masked_img, None, 0))
    all_hmap = all_hmap.flatten()
    vol = abs(sum(all_hmap)) * pixsize[0] * resize_coeff * pixsize[1] * resize_coeff * pixsize[2] * resize_coeff / 10 ** 9
    return vol*13100

def plot_evaluation(prof, heat, pred1, pred2, pred3, filename_temp):
    vol_loss1_t = get_volume(return_back(pred1))
    vol_loss2_t = get_volume(return_back(pred2))
    vol_loss3_t = get_volume(return_back(pred3))
    vol_loss_gt = get_volume(return_back(heat))
    mae1_temp = mean_absolute_error(norm_heat.flatten(), pred1_img.flatten())
    mae2_temp = mean_absolute_error(norm_heat.flatten(), pred2_img.flatten())
    mae3_temp = mean_absolute_error(norm_heat.flatten(), pred3_img.flatten())
    fig, axes = plt.subplots(nrows=1, ncols=5)
    fig.suptitle("Prediction for {0} \nVolume loss (predicted): {1} and {3} \nVolume loss (on GT): {2}".format(filename_temp, vol_loss1_t, vol_loss_gt, vol_loss2_t))
    #fig.suptitle(filename_temp)
    axes[0].set_title("Original input", fontdict={'fontsize': 10,
                                                  'fontweight': 1,
                                                  'verticalalignment': 'baseline',
                                                  'horizontalalignment': 'center'}, loc='center')
    axes[0].imshow(prof)
    im = axes[1].imshow(heat, vmin=0, vmax=1, cmap='plasma')
    axes[1].set_title('Normalized heatmap (GT)', fontdict={'fontsize': 10,
                                                         'fontweight': 1,
                                                         'verticalalignment': 'baseline',
                                                         'horizontalalignment': 'center'}, loc='center')
    axes[2].imshow(pred1, vmin=0, vmax=1, cmap='plasma')
    axes[2].set_title('Predicted heatmap (Pred1)\n for Reverse Huber\nVolume loss (predicted): {:.4f}'.format(vol_loss1_t), fontdict={'fontsize': 10,
                                                            'fontweight': 1,
                                                            'verticalalignment': 'baseline',
                                                            'horizontalalignment': 'center'}, loc='center')

    axes[3].imshow(pred2, vmin=0, vmax=1, cmap='plasma')
    axes[3].set_title('Predicted heatmap (Pred2)\n for MAE loss\nVolume loss (predicted): {:.4f}'.format(vol_loss2_t),
                      fontdict={'fontsize': 10,
                                'fontweight': 1,
                                'verticalalignment': 'baseline',
                                'horizontalalignment': 'center'}, loc='center')
    axes[4].imshow(pred2, vmin=0, vmax=1, cmap='plasma')
    axes[4].set_title('Predicted heatmap (Pred3)\n for MAE loss\nVolume loss (predicted): {:.4f}'.format(vol_loss3_t),
                      fontdict={'fontsize': 10,
                                'fontweight': 1,
                                'verticalalignment': 'baseline',
                                'horizontalalignment': 'center'}, loc='center')
    #axes[3].set_title('Diff (Pred-GT)', fontdict={'fontsize': 10,
    #                                              'fontweight': 1,
    #                                              'verticalalignment': 'baseline',
    #                                              'horizontalalignment': 'center'}, loc='center')
    #axes[3].imshow(pred_img - orig_heat, vmin=-0.25, vmax=0.25,
    #               cmap='gray')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

def plot_metric(metric_arr, name):
    red_square = dict(markerfacecolor='r', marker='s')
    fig5, ax5 = plt.subplots()
    ax5.set_title(name)
    ax5.boxplot(metric_arr, vert=False, flierprops=red_square)

inp_size=patch_size
dataDirectory = r"/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics"
savingDirectory = r"/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics" #r"/Volumes/USB DISK"

directoriesHeatmap = [r"/20220617/rawdata/after/4-1-1",
                      r"/20220617/rawdata/after/4-1-2",
                      r"/20220617/rawdata/after/4-1-3",
                      r"/20220617/rawdata/after/4-1-4",
                      r"/20220617/rawdata/after/4-1-5",
                      r"/20220617/rawdata/after/4-1-6",
                      r"/20220823/rawdata/after/2-3-1",
                      r"/20220823/rawdata/after/2-3-2",
                      r"/20220823/rawdata/after/2-3-3",

                      r"/20220823/rawdata/after/2-3-4",
                      r"/20220823/rawdata/after/2-3-5",
                      r"/20220823/rawdata/after/2-3-6",
                      r"/20220823/rawdata/after/2-3-7",
                      r"/20220823/rawdata/after/2-3-8",
                      r"/20220823/rawdata/after/2-3-9",
                      r"/20220823/rawdata/after/2-3-10",

                      r"/20220823/rawdata/after/2-2-1",
                      r"/20220823/rawdata/after/2-2-2",
                      r"/20220823/rawdata/after/2-2-3",
                      r"/20220823/rawdata/after/2-2-4",
                      r"/20220823/rawdata/after/2-2-5",
                      r"/20220823/rawdata/after/2-2-6",
                      r"/20220823/rawdata/after/2-2-7",
                      r"/20220823/rawdata/after/2-2-8",
                      r"/20220823/rawdata/after/2-2-9",
                      r"/20220823/rawdata/after/2-2-10",
                      r"/20220823/rawdata/after/2-2-11",

                      r"/20220719/rawdata/after/4-3-1",
                      r"/20220719/rawdata/after/4-3-2",
                      r"/20220719/rawdata/after/4-3-3",
                      r"/20220719/rawdata/after/4-3-4",
                      r"/20220719/rawdata/after/4-3-5",
                      r"/20220719/rawdata/after/4-3-6", r"/20220719/rawdata/after/4-3-7",
                      r"/20220719/rawdata/after/4-3-8", r"/20220719/rawdata/after/4-3-9",
                      r"/20220719/rawdata/after/4-3-10",
                      r"/20220719/rawdata/after/4-3-11", r"/20220719/rawdata/after/4-3-12",

                      r"/20220719/rawdata/after/4-4-1",
                      r"/20220719/rawdata/after/4-4-2",
                      r"/20220719/rawdata/after/4-4-3",
                      r"/20220719/rawdata/after/4-4-4",
                      r"/20220719/rawdata/after/4-4-5",
                      r"/20220719/rawdata/after/4-4-6",
                      r"/20220719/rawdata/after/4-4-7",
                      r"/20220719/rawdata/after/4-4-8",
                      r"/20220719/rawdata/after/4-4-9",
                      r"/20220719/rawdata/after/4-4-10",
                      r"/20220719/rawdata/after/4-4-11",
                      r"/20220719/rawdata/after/4-4-12",

                      r"/20220719/rawdata/after/4-5-1",
                      r"/20220719/rawdata/after/4-5-2",
                      r"/20220719/rawdata/after/4-5-3",
                      r"/20220719/rawdata/after/4-5-4",
                      r"/20220719/rawdata/after/4-5-5",
                      r"/20220719/rawdata/after/4-5-6",
                      r"/20220719/rawdata/after/4-5-7",
                      r"/20220719/rawdata/after/4-5-8",
                      r"/20220719/rawdata/after/4-5-9",
                      r"/20220719/rawdata/after/4-5-10",
                      r"/20220719/rawdata/after/4-5-11", r"/20220719/rawdata/after/4-5-12",

                      r"/20220719/rawdata/after/4-6-1",
                      r"/20220719/rawdata/after/4-6-2",
                      r"/20220719/rawdata/after/4-6-3",
                      r"/20220719/rawdata/after/4-6-4",
                      r"/20220719/rawdata/after/4-6-5",
                      r"/20220719/rawdata/after/4-6-6", r"/20220719/rawdata/after/4-6-7",
                      r"/20220719/rawdata/after/4-6-8", r"/20220719/rawdata/after/4-6-9",
                      r"/20220719/rawdata/after/4-6-10",
                      r"/20220719/rawdata/after/4-6-11", r"/20220719/rawdata/after/4-6-12",
                      r"/20220906/rawdata/after/2-5-1",
                      r"/20220906/rawdata/after/2-5-2",
                      r"/20220906/rawdata/after/2-5-3",
                      r"/20220906/rawdata/after/2-5-5",
                      r"/20220906/rawdata/after/2-5-6",
                      r"/20220906/rawdata/after/2-5-7",
                      r"/20220906/rawdata/after/2-5-8",
                      r"/20220906/rawdata/after/2-5-9",
                      r"/20220906/rawdata/after/2-5-10",
                      r"/20220906/rawdata/after/2-5-11",
                      r"/20220906/rawdata/after/2-5-12"]

directoriesProfilometer = [r"/20220617/profilometer/before/4-1-1",
                           r"/20220617/profilometer/before/4-1-2",
                           r"/20220617/profilometer/before/4-1-3", r"/20220617/profilometer/before/4-1-4",
                           r"/20220617/profilometer/before/4-1-5",
                           r"/20220617/profilometer/before/4-1-6",
                           r"/20220823/profilometer/before/2-3-1",
                           r"/20220823/profilometer/before/2-3-2",
                           r"/20220823/profilometer/before/2-3-3",
                           r"/20220823/profilometer/before/2-3-4",
                           r"/20220823/profilometer/before/2-3-5",
                           r"/20220823/profilometer/before/2-3-6",
                           r"/20220823/profilometer/before/2-3-7",
                           r"/20220823/profilometer/before/2-3-8",
                           r"/20220823/profilometer/before/2-3-9",
                           r"/20220823/profilometer/before/2-3-10",

                           r"/20220823/profilometer/before/2-2-1",
                           r"/20220823/profilometer/before/2-2-2",
                           r"/20220823/profilometer/before/2-2-3",
                           r"/20220823/profilometer/before/2-2-4",
                           r"/20220823/profilometer/before/2-2-5",
                           r"/20220823/profilometer/before/2-2-6",
                           r"/20220823/profilometer/before/2-2-7",
                           r"/20220823/profilometer/before/2-2-8",
                           r"/20220823/profilometer/before/2-2-9",
                           r"/20220823/profilometer/before/2-2-10",
                           r"/20220823/profilometer/before/2-2-11",

                           r"/20220719/profilometer/before/4-3-1",
                           r"/20220719/profilometer/before/4-3-2",
                           r"/20220719/profilometer/before/4-3-3",
                           r"/20220719/profilometer/before/4-3-4",
                           r"/20220719/profilometer/before/4-3-5",
                           r"/20220719/profilometer/before/4-3-6", r"/20220719/profilometer/before/4-3-7",
                           r"/20220719/profilometer/before/4-3-8", r"/20220719/profilometer/before/4-3-9",
                           r"/20220719/profilometer/before/4-3-10",
                           r"/20220719/profilometer/before/4-3-11", r"/20220719/profilometer/before/4-3-12",

                           r"/20220719/profilometer/before/4-4-1",
                           r"/20220719/profilometer/before/4-4-2",
                           r"/20220719/profilometer/before/4-4-3",
                           r"/20220719/profilometer/before/4-4-4",
                           r"/20220719/profilometer/before/4-4-5",
                           r"/20220719/profilometer/before/4-4-6",
                           r"/20220719/profilometer/before/4-4-7",
                           r"/20220719/profilometer/before/4-4-8", r"/20220719/profilometer/before/4-4-9",
                           r"/20220719/profilometer/before/4-4-10",
                           r"/20220719/profilometer/before/4-4-11", r"/20220719/profilometer/before/4-4-12",

                           r"/20220719/profilometer/before/4-5-1",
                           r"/20220719/profilometer/before/4-5-2",
                           r"/20220719/profilometer/before/4-5-3",
                           r"/20220719/profilometer/before/4-5-4",
                           r"/20220719/profilometer/before/4-5-5",
                           r"/20220719/profilometer/before/4-5-6",
                           r"/20220719/profilometer/before/4-5-7",
                           r"/20220719/profilometer/before/4-5-8",
                           r"/20220719/profilometer/before/4-5-9",
                           r"/20220719/profilometer/before/4-5-10",
                           r"/20220719/profilometer/before/4-5-11", r"/20220719/profilometer/before/4-5-12",

                           r"/20220719/profilometer/before/4-6-1",
                           r"/20220719/profilometer/before/4-6-2",
                           r"/20220719/profilometer/before/4-6-3",
                           r"/20220719/profilometer/before/4-6-4",
                           r"/20220719/profilometer/before/4-6-5",
                           r"/20220719/profilometer/before/4-6-6", r"/20220719/profilometer/before/4-6-7",
                           r"/20220719/profilometer/before/4-6-8", r"/20220719/profilometer/before/4-6-9",
                           r"/20220719/profilometer/before/4-6-10",
                           r"/20220719/profilometer/before/4-6-11", r"/20220719/profilometer/before/4-6-12",

                           r"/20220906/profilometer/before/2-5-1",
                           r"/20220906/profilometer/before/2-5-2",
                           r"/20220906/profilometer/before/2-5-3",
                           r"/20220906/profilometer/before/2-5-5",
                           r"/20220906/profilometer/before/2-5-6",
                           r"/20220906/profilometer/before/2-5-7",
                           r"/20220906/profilometer/before/2-5-8",
                           r"/20220906/profilometer/before/2-5-9",
                           r"/20220906/profilometer/before/2-5-10",
                           r"/20220906/profilometer/before/2-5-11",
                           r"/20220906/profilometer/before/2-5-12"]

model1_path = "model_adjRevHuber_init_Adam_lr0.001_RevHub_1_ep35"
model2_path = "model_RevHuber_init_Adam_lr0.001_RevHub_1_ep37"
model3_path = "model_RevHuber_init_Adam_lr0.001_RevHub_1_ep37" #ToDo add layer

train_thres = int(len(directoriesProfilometer)*0.6)
val_thres = int(len(directoriesProfilometer)*0.8)

to_use = np.ones(len(c1)).astype(int)
to_use[[341, 333, 331, 330, 328, 327, 305, 290]] = 0

filenames_prof_before_full = []
filenames_heatmap_full = []
filenames_prof_after_full = []
directories = []
for i in range(len(directoriesProfilometer)):
    filenames_temp = list(sorted(listdir(dataDirectory + directoriesProfilometer[i])))
    before_temp = [dataDirectory+directoriesProfilometer[i]+'/'+filenames_temp[j] for j in range(len(filenames_temp))]
    after_temp = [before_temp[j].replace('before', 'after') for j in range(len(before_temp))]
    heatmap_temp = [(after_temp[j].replace('profilometer', 'rawdata')).replace('png', 'csv') for j in range(len(after_temp))]
    directories_temp = [i for j in range(len(before_temp))]
    filenames_prof_after_full.extend(after_temp)
    filenames_prof_before_full.extend(before_temp)
    filenames_heatmap_full.extend(heatmap_temp)
    directories.extend(directories_temp)
directories = np.array(directories)
before_temp = np.array(before_temp)
after_temp = np.array(after_temp)
filenames_prof_after_full = np.array(filenames_prof_after_full)
filenames_prof_before_full = np.array(filenames_prof_before_full)
filenames_heatmap_full = np.array(filenames_heatmap_full)

# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

opt = 'Adam'
lr = 0.01
loss = 'Huber'

losses = {
        "intermediate_output0": RevHub_loss,
        "intermediate_output1": RevHub_loss,
        "intermediate_output2": RevHub_loss,
        "output": RevHub_loss
    }
lossWeights = {"intermediate_output0": 1.0,
               "intermediate_output1": 1.0,
               "intermediate_output2": 1.0,
               "output": 1.0}

def new_rsq(target, output):
    rsq = tfa.metrics.r_square.RSquare()
    return rsq(return_back(target.numpy(), par=1, mean_val=-0.0059, std_val=0.0112), return_back(output.numpy(), par=1, mean_val=-0.0059, std_val=0.0112))

metrics = [tf.keras.metrics.MeanAbsoluteError(), tfa.metrics.r_square.RSquare(), new_rsq]

model1 = tf.keras.models.load_model(model1_path, custom_objects={'new_rsq':new_rsq, 'relative_MAE': relative_MAE(), 'RevHub_loss':RevHub_loss})
model2 = tf.keras.models.load_model(model2_path, custom_objects={'new_rsq':new_rsq, 'relative_MAE': relative_MAE(), 'RevHub_loss':RevHub_loss})
model3 = tf.keras.models.load_model(model3_path, custom_objects={'new_rsq':new_rsq, 'relative_MAE': relative_MAE(), 'RevHub_loss':RevHub_loss})

#create variables for MAEs, Rscores etc. and original and predicted volume loss difference [ToDo which other metrics?]
vol_loss1 = []
vol_loss2 = []
vol_loss3 = []
vol_loss_gt = []

mae1 = []
mae2 = []
mae3 = []

r2_1 = []
r2_2 = []
r2_3 = []

for i in range(train_thres):
    filenames_heatmap = filenames_heatmap_full[directories == i]
    filenames_input = filenames_prof_before_full[directories == i]
    filenames_after = filenames_prof_after_full[directories == i]
    c1_temp = c1[directories == i]
    c2_temp = c2[directories == i]
    to_use_temp = to_use[directories == i]

    for j in range(len(filenames_input)):
        prof_filename = filenames_input[j]
        heat_filename = filenames_heatmap[j]##

        orig_img = mask(resize_profilometer(prof_filename, resize_coeff=1), c=c1_temp[j])
        orig_heat = mask(
            interp(pd.read_csv(filepath_or_buffer=heat_filename, skiprows=22, engine='python').to_numpy()[:1664, :1792],
                   filename=heat_filename, resize_coeff=1), c=c2_temp[j])
        p_heat = np.array(patches(orig_heat))
        p_prof = np.array(patches(orig_img))

        norm_heat = norm_01(orig_heat)

        pred1 = model1.predict(p_prof)[3]
        pred2 = model2.predict(p_prof)[3]
        pred3 = model3.predict(p_prof)[3]
        pred1_img = patches_overlap_reconstruct(pred1, orig_heat)
        pred2_img = patches_overlap_reconstruct(pred2, orig_heat)
        pred3_img = patches_overlap_reconstruct(pred3, orig_heat)
        vol_loss_gt_temp = get_volume(orig_heat)
        vol_loss1_temp = get_volume(return_back(pred1_img))
        vol_loss2_temp = get_volume(return_back(pred2_img))
        vol_loss3_temp = get_volume(return_back(pred3_img))
        mae1_temp = mean_absolute_error(norm_heat.flatten(), pred1_img.flatten())
        mae2_temp = mean_absolute_error(norm_heat.flatten(), pred2_img.flatten())
        mae3_temp = mean_absolute_error(norm_heat.flatten(), pred3_img.flatten())
        #if mae1_temp>0.3:
        #    print("For file {} the MAE is {} for model 1".format(prof_filename, mae1_temp))
        #if mae2_temp>0.3:
        #    print("For file {} the MAE is {} for model 2".format(prof_filename, mae2_temp))
        #if mae3_temp>0.3:
        #    print("For file {} the MAE is {} for model 1".format(prof_filename, mae3_temp))
        r2_1_temp = r2_score(norm_heat.flatten(), pred1_img.flatten())
        r2_2_temp = r2_score(norm_heat.flatten(), pred2_img.flatten())
        r2_3_temp = r2_score(norm_heat.flatten(), pred3_img.flatten())
        #if i%10==0 and j==1:
        #    plot_evaluation(orig_img, norm_heat, pred1_img, pred2_img, pred3_img, prof_filename)
        #calculate other metrics on predicted image
        #extend arrays by metrics
        mae1.append(mae1_temp)
        mae2.append(mae2_temp)
        mae3.append(mae3_temp)
        r2_1.append(r2_1_temp)
        r2_2.append(r2_2_temp)
        r2_3.append(r2_3_temp)
        vol_loss1.append(vol_loss1_temp)
        vol_loss2.append(vol_loss2_temp)
        vol_loss3.append(vol_loss3_temp)
#
        vol_loss_gt.append(vol_loss_gt_temp)
#        print("I am done with image number {} with name {}".format(4*i+j+1, prof_filename))

print("Done with train")
#Boxplots etc. of metrics for the evaluation
#plot_metric(mae1, name="Train MAE for model 1")
#plot_metric(mae2, name="Train MAE for model 2")
#plot_metric(mae3, name="Train MAE for model 3")
#plot_metric(vol_loss1-vol_loss_gt, name="Train vol_loss for model 1")
#plot_metric(vol_loss2-vol_loss_gt, name="Train vol_loss for model 2")
#plot_metric(vol_loss3-vol_loss_gt, name="Train vol_loss for model 3")
#plot_metric(r2_1, name="Train R^2 for model 1")
#plot_metric(r2_2, name="Train R^2 for model 2")
#plot_metric(r2_3, name="Train R^2 for model 3")

vol_loss1 = []
vol_loss2 = []
vol_loss3 = []
vol_loss_gt = []

mae1 = []
mae2 = []
mae3 = []

r2_1 = []
r2_2 = []
r2_3 = []

for i in range(train_thres, val_thres):
    filenames_heatmap = filenames_heatmap_full[directories == i]
    filenames_input = filenames_prof_before_full[directories == i]
    filenames_after = filenames_prof_after_full[directories == i]
    c1_temp = c1[directories == i]
    c2_temp = c2[directories == i]
    to_use_temp = to_use[directories == i]

    for j in range(len(filenames_input)):
        prof_filename = filenames_input[j]
        heat_filename = filenames_heatmap[j]

        orig_img = mask(resize_profilometer(prof_filename, resize_coeff=1), c=c1_temp[j])
        orig_heat = mask(
            interp(pd.read_csv(filepath_or_buffer=heat_filename, skiprows=22, engine='python').to_numpy()[:1664, :1792],
                   filename=heat_filename, resize_coeff=1), c=c2_temp[j])
        p_heat = np.array(patches(orig_heat))
        p_prof = np.array(patches(orig_img))

        norm_heat = norm_01(orig_heat)

        pred1 = model1.predict(p_prof)[3]
        pred2 = model2.predict(p_prof)[3]
        pred3 = model3.predict(p_prof)[3]
        pred1_img = patches_overlap_reconstruct(pred1, orig_heat)
        pred2_img = patches_overlap_reconstruct(pred2, orig_heat)
        pred3_img = patches_overlap_reconstruct(pred3, orig_heat)
        vol_loss_gt_temp = get_volume(orig_heat)
        vol_loss1_temp = get_volume(return_back(pred1_img))
        vol_loss2_temp = get_volume(return_back(pred2_img))
        vol_loss3_temp = get_volume(return_back(pred3_img))
        mae1_temp = mean_absolute_error(norm_heat.flatten(), pred1_img.flatten())
        mae2_temp = mean_absolute_error(norm_heat.flatten(), pred2_img.flatten())
        mae3_temp = mean_absolute_error(norm_heat.flatten(), pred3_img.flatten())
        if mae1_temp>0.3:
            print("For file {} the MAE is {} for model 1".format(prof_filename, mae1_temp))
        if mae2_temp>0.3:
            print("For file {} the MAE is {} for model 2".format(prof_filename, mae2_temp))
        if mae3_temp>0.3:
            print("For file {} the MAE is {} for model 1".format(prof_filename, mae3_temp))
        r2_1_temp = r2_score(norm_heat.flatten(), pred1_img.flatten())
        r2_2_temp = r2_score(norm_heat.flatten(), pred2_img.flatten())
        r2_3_temp = r2_score(norm_heat.flatten(), pred3_img.flatten())
        r2_1.append(r2_1_temp)
        r2_2.append(r2_2_temp)
        r2_3.append(r2_3_temp)
        #if i%10==0 and j==1:
        #    plot_evaluation(orig_img, norm_heat, pred1_img, pred2_img, pred3_img, prof_filename)
        #calculate other metrics on predicted image
        #extend arrays by metrics
        mae1.append(mae1_temp)
        mae2.append(mae2_temp)
        mae3.append(mae3_temp)
        vol_loss1.append(vol_loss1_temp)
        vol_loss2.append(vol_loss2_temp)
        vol_loss3.append(vol_loss3_temp)
        vol_loss_gt.append(vol_loss_gt_temp)
        r2_1.append(r2_1_temp)
        r2_2.append(r2_2_temp)
        r2_3.append(r2_3_temp)

#Boxplots etc. of metrics for the evaluation
print("Done with validation")
#plot_metric(mae1, name="Validation MAE for model 1")
#plot_metric(mae2, name="Validation MAE for model 2")
#plot_metric(mae3, name="Validation MAE for model 3")
#plot_metric(vol_loss1-vol_loss_gt, name="Validation vol_loss for model 1")
#plot_metric(vol_loss2-vol_loss_gt, name="Validation vol_loss for model 2")
#plot_metric(vol_loss3-vol_loss_gt, name="Validation vol_loss for model 3")
#plot_metric(r2_1, name="Validation R^2 for model 1")
#plot_metric(r2_2, name="Validation R^2 for model 2")
#plot_metric(r2_3, name="Validation R^2 for model 3")

vol_loss1 = []
vol_loss2 = []
vol_loss3 = []
vol_loss_gt = []

mae1 = []
mae2 = []
mae3 = []

r2_1 = []
r2_2 = []
r2_3 = []

for i in range(val_thres, len(directoriesProfilometer)):
    filenames_heatmap = filenames_heatmap_full[directories == i]
    filenames_input = filenames_prof_before_full[directories == i]
    filenames_after = filenames_prof_after_full[directories == i]
    c1_temp = c1[directories == i]
    c2_temp = c2[directories == i]
    to_use_temp = to_use[directories == i]

    for j in range(len(filenames_input)):
        prof_filename = filenames_input[j]
        heat_filename = filenames_heatmap[j]

        orig_img = mask(resize_profilometer(prof_filename, resize_coeff=1), c=c1_temp[j])
        orig_heat = mask(
            interp(pd.read_csv(filepath_or_buffer=heat_filename, skiprows=22, engine='python').to_numpy()[:1664, :1792],
                   filename=heat_filename, resize_coeff=1), c=c2_temp[j])
        p_heat = np.array(patches(orig_heat))
        p_prof = np.array(patches(orig_img))

        norm_heat = norm_01(orig_heat)

        pred1 = model1.predict(p_prof)[3]
        pred2 = model2.predict(p_prof)[3]
        pred3 = model3.predict(p_prof)[3]
        pred1_img = patches_overlap_reconstruct(pred1, orig_heat)
        pred2_img = patches_overlap_reconstruct(pred2, orig_heat)
        pred3_img = patches_overlap_reconstruct(pred3, orig_heat)
        vol_loss_gt_temp = get_volume(orig_heat)
        vol_loss1_temp = get_volume(return_back(pred1_img))
        vol_loss2_temp = get_volume(return_back(pred2_img))
        vol_loss3_temp = get_volume(return_back(pred3_img))
        mae1_temp = mean_absolute_error(norm_heat.flatten(), pred1_img.flatten())
        mae2_temp = mean_absolute_error(norm_heat.flatten(), pred2_img.flatten())
        mae3_temp = mean_absolute_error(norm_heat.flatten(), pred3_img.flatten())
        if mae1_temp>0.3:
            print("For file {} the MAE is {} for model 1".format(prof_filename, mae1_temp))
        if mae2_temp>0.3:
            print("For file {} the MAE is {} for model 2".format(prof_filename, mae2_temp))
        if mae3_temp>0.3:
            print("For file {} the MAE is {} for model 1".format(prof_filename, mae3_temp))
        r2_1_temp = r2_score(norm_heat.flatten(), pred1_img.flatten())
        r2_2_temp = r2_score(norm_heat.flatten(), pred2_img.flatten())
        r2_3_temp = r2_score(norm_heat.flatten(), pred3_img.flatten())
        r2_1.append(r2_1_temp)
        r2_2.append(r2_2_temp)
        r2_3.append(r2_3_temp)
        #if i%10==0 and j==1:
       #    plot_evaluation(orig_img, norm_heat, pred1_img, pred2_img, pred3_img, prof_filename)
        #calculate other metrics on predicted image
        #extend arrays by metrics
        mae1.append(mae1_temp)
        mae2.append(mae2_temp)
        mae3.append(mae3_temp)
        vol_loss1.append(vol_loss1_temp)
        vol_loss2.append(vol_loss2_temp)
        vol_loss3.append(vol_loss3_temp)
        vol_loss_gt.append(vol_loss_gt_temp)
        r2_1.append(r2_1_temp)
        r2_2.append(r2_2_temp)
        r2_3.append(r2_3_temp)

#Boxplots etc. of metrics for the evaluation
print("Done with test")
#plot_metric(mae1, name="Test MAE for model 1")
#plot_metric(mae2, name="Test MAE for model 2")
#plot_metric(mae3, name="Test MAE for model 3")
#plot_metric(vol_loss1-vol_loss_gt, name="Test vol_loss for model 1")
#plot_metric(vol_loss2-vol_loss_gt, name="Test vol_loss for model 2")
#plot_metric(vol_loss3-vol_loss_gt, name="Test vol_loss for model 3")
#plot_metric(r2_1, name="Test R^2 for model 1")
#plot_metric(r2_2, name="Test R^2 for model 2")
#plot_metric(r2_3, name="Test
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# R^2 for model 3")