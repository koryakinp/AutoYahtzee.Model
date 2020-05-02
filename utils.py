import cv2
import math
import uuid
import glob
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans


def prepare_image_data(filename):
    image = cv2.imread(filename)
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(-1,1)
    clt = MiniBatchKMeans(n_clusters = 2)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w))
    
    quant = np.int32(quant)
    quant[quant == quant.min()] = -1
    quant[quant == quant.max()] = 1
    
    return quant

def compute_conv(image_data, kernels):
    image_data = np.expand_dims(image_data, axis=0)
    image_data = np.expand_dims(image_data, axis=3)
    image_data = np.float32(image_data)

    kernels = np.expand_dims(kernels, axis=0)
    kernels = kernels.transpose(2, 3, 0, 1)
    
    res = tf.nn.conv2d(image_data, kernels, [1,1,1,1], padding='SAME')
    res = tf.squeeze(res)
    sess = tf.Session()
    with sess.as_default():
        res = res.eval()
        
    return list(np.int32(res.transpose(2,0,1)))

def pad_kernels(kernels):
    max_kernel_w = 0

    for kernel in kernels:
        if kernel.shape[0] > max_kernel_w:
            max_kernel_w = kernel.shape[0]

    temp_kernels = np.zeros((max_kernel_w,max_kernel_w))

    for kernel_id in range(len(kernels)):
        kernel = kernels[kernel_id]
        if kernel.shape[0] < max_kernel_w:
            diff = max_kernel_w - kernel.shape[0]

            if diff % 2 == 0:
                diff = diff/2
            else:
                diff = (diff/2) + 1

            diff = np.int(diff)

            kernels[kernel_id] = np.pad(kernel, diff, mode='constant', constant_values=0)

            if kernels[kernel_id].shape[0] > max_kernel_w:
                kernels[kernel_id] = np.delete(kernels[kernel_id], 1, 0)
                kernels[kernel_id] = np.delete(kernels[kernel_id], 1, 1)
                
    return kernels

def draw_dot(img, radius, offset_from_center):
    center = np.array([np.uint8((img.shape[0] - 1)/2), np.uint8((img.shape[1] - 1)/2)])
    dot_position = center - np.array(offset_from_center)
    return cv2.circle(img, tuple(dot_position), radius=radius, color=(1,1,1), thickness=-1)

def build_kernel(side, size, dot_radius, ofc, border, angle):
    
    ofc = 7
    
    img = np.full([size, size], 2, dtype=np.uint8)   
    if side == 1:
        img = draw_dot(img, dot_radius, [0,0])
    elif side == 2:
        img = draw_dot(img, dot_radius, [ofc,ofc])
        img = draw_dot(img, dot_radius, [-ofc,-ofc])
    elif side == 3:
        img = draw_dot(img, dot_radius, [0,0])
        img = draw_dot(img, dot_radius, [ofc,ofc])
        img = draw_dot(img, dot_radius, [-ofc,-ofc])
    elif side == 4:
        img = draw_dot(img, dot_radius, [ofc,ofc])
        img = draw_dot(img, dot_radius, [-ofc,-ofc])
        img = draw_dot(img, dot_radius, [-ofc,ofc])
        img = draw_dot(img, dot_radius, [ofc,-ofc])
    elif side == 5:
        img = draw_dot(img, dot_radius, [ofc,ofc])
        img = draw_dot(img, dot_radius, [-ofc,-ofc])
        img = draw_dot(img, dot_radius, [-ofc,ofc])
        img = draw_dot(img, dot_radius, [ofc,-ofc])
        img = draw_dot(img, dot_radius, [0,0])
    elif side == 6:
        
        off_v = 1
        off_h = 2
        
        img = np.full([size,size],2, dtype=np.uint8)
        img = draw_dot(img, dot_radius, [ofc - off_v,ofc + off_h])
        img = draw_dot(img, dot_radius, [-ofc + off_v,-ofc - off_h])
        img = draw_dot(img, dot_radius, [-ofc + off_v,ofc + off_h])
        img = draw_dot(img, dot_radius, [ofc - off_v,-ofc - off_h])
        img = draw_dot(img, dot_radius, [ofc - off_v,0])
        img = draw_dot(img, dot_radius, [-ofc + off_v,0])
        
    img = rotate_image(img, angle)
    img = np.int8(img)
    
    img[img == 1] = 10
    img[img == 2] = 20
    img[img == 20] = 1
    img[img == 10] = -1
    
    return img

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(0,0,0))

def build_empty_kernels(size=33, radius=4, ofc=7, border=2):
    kernels = []

    for angle in range(0, 90, 10):
        kernels.append(build_kernel(0, size, radius, ofc, border, angle))
                
    kernels = pad_kernels(kernels)
            
    return kernels


def build_dice_kernels(size=30, radius=3, ofc=7, border=2):
    kernels = []
    dice_sides = []

    for side in range(1,7):
        if side in [1,4,5]:
            for angle in range(0, 90, 2):
                kernels.append(build_kernel(side, size, radius, ofc, border, angle))
                dice_sides.append(side)   
        else:
            for angle in range(0, 180, 2):
                kernels.append(build_kernel(side, size, radius, ofc, border, angle))
                dice_sides.append(side)
    
    kernels = pad_kernels(kernels)
    
    return kernels, dice_sides

def get_dice_images(image, kernels):
    peaks = []
    dices = []
    size = 17

    original = image.copy()
    
    counter = 0
    
    while True:
        max_val = -999999
        
        conv_results = compute_conv(image, kernels)
        
        for conv_result in conv_results:
            maxx = conv_result.max()
            
            if maxx > max_val:
                max_val = maxx
                peak = np.where(conv_result==maxx)
       
        if(max_val < 300):  
            break 
                     
        cx = peak[1][0]
        cy = peak[0][0]
        
        dices.append(original[cy - size:cy + size, cx - size:cx + size].copy())
        peaks.append(peak)
        image = cv2.circle(image, (cx, cy), 17,(0,0,0), -1)
        
        if counter > 10:
            break
            
        counter += 1
        
    return dices

def predict(dice_images, dice_sides, dice_kernels):
    labels = []

    for dice_image in dice_images:

        label = 0
        max_val = -999999

        conv_results = compute_conv(dice_image, dice_kernels)
        
        for kernel_id in range(len(conv_results)):
            cur_max = conv_results[kernel_id].max()
            
            if cur_max > max_val:
                max_val = cur_max
                label = dice_sides[kernel_id]

        labels.append(label)
        
    return labels

def get_labels(labels):
    labels.sort()
    res = ''
    for label in labels:
        res += str(label)

    return res

def process_image(filename, dice_kernels, dice_sides, empty_kernels):
    image = prepare_image_data(filename)
    dice_images = get_dice_images(image, empty_kernels)
    labels = predict(dice_images, dice_sides, dice_kernels)
    result = get_labels(labels)
    return result