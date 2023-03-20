# import libraries

import cv2
import numpy as np


# helper functions

'''
Returns rect with dimensions clamped within a specified size.
'''
def clamp_extents(w, h, rect):
    
    x1 = rect[2]
    y1 = rect[0]
    x2 = rect[3]
    y2 = rect[1]

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h
    
    return (y1, y2, x1, x2)


# main functions

'''
Returns crops of dimension size from a given image.
'''
def crop(img, size_x, size_y, padding = 0):

    height = img.shape[0]
    width = img.shape[1]
    y =  -(height // -size_y)
    x = -(width // -size_x)
    
    # split image
    imgs = []
    for i in range (y):
        for j in range(x):
            x1 = j * size_x - padding
            y1 = i * size_y - padding
            x2 = (j + 1) * size_x + padding
            y2 = (i + 1) * size_y + padding

            rect = clamp_extents(width, height, (y1, y2, x1, x2))
            imgs.append(img[rect[0]:rect[1], rect[2]:rect[3]])

    return imgs

'''
Draws openCV rectangles of specified color and thickness at bounding box coordinates.
'''
def draw_boxes(img, boxes, color, thickness = 5, fill = 0):
    boxes = list(set(boxes))
    overlay = np.zeros(img.shape, dtype=np.uint8)
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1), (x2,  y2), color, -1)

    img = cv2.addWeighted(img, 1, overlay, fill, 0)

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
