"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import os
import time
import argparse
import enum
import cv2
import numpy as np
import pycuda.autoinit  # This is needed for initializing CUDA driver
import xml.etree.ElementTree as ET # used to read annotation files

from math import sqrt
from utils.yolo_classes import get_cls_dict, get_cls_from_int
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.image_tiler import crop, draw_boxes

## Benchmarking Variables ##
batch_size = 1
WINDOW_NAME = 'TrtYOLO'

avg_fps = 0.0
avg_inference_time = 0.0
avg_latency = 0.0
avg_loss = 0.0
avg_iou = 0.0

avg_cpu_percent = 0.0
avg_cpu_raw = 0.0
avg_cpu_temp = 0.0
cpu_peak = 0.0
cpu_raw_peak = 0.0
cpu_temp_peak = 0.0

avg_gpu_percent = 0.0
avg_gpu_raw = 0.0
avg_gpu_temp = 0.0
gpu_peak = 0.0
gpu_raw_peak = 0.0
gpu_temp_peak = 0.0

class Pattern(enum.IntEnum):
    NONE = 0
    STATIC = 1
    ATTENTIVE = 2
    PREDICTIVE = 3

def gpu_usage():
    global avg_gpu_raw, gpu_peak, gpu_raw_peak
    gpuLoadFile="/sys/devices/gpu.0/load"
    with open(gpuLoadFile, 'r') as gpuFile:
        gpuusage=float(gpuFile.read())
    avg_gpu_raw += gpuusage
    gpu_value = gpuusage / 10
    if gpuusage > gpu_raw_peak:
        gpu_raw_peak = gpuusage
        gpu_peak = gpu_value
    return gpu_value

def cpu_usage():
        global avg_cpu_raw, cpu_peak, cpu_raw_peak
        cpuLoadFile="/proc/meminfo"
        cpuTotal = 0.0
        cpufree = 0.0	
        with open(cpuLoadFile, 'r') as fh:
            lines = fh.read()
            fh.close()
          
            for line in lines.split('\n'):
                fields = line.split(' ', 2)
                if fields[0] == 'MemTotal:':
                    cpuTotal = [float (s) for s in fields[2].split() if s.isdigit()][0]
                elif fields[0] == 'MemAvailable:':
                    cpufree = [float (s) for s in fields[2].split() if s.isdigit()][0]

        cpuusage = cpuTotal - cpufree
        cpu_value = (cpuusage / cpuTotal) * 100
        avg_cpu_raw += cpuusage / 1000
        if (cpuusage > cpu_raw_peak):
            cpu_raw_peak = cpuusage / 1000
            cpu_peak = cpu_value
        return cpu_value

def gpu_temperature():
        global gpu_temp_peak
        tempLoadFile="/sys/devices/virtual/thermal/thermal_zone2/temp"
        with open(tempLoadFile,'r') as tempFile:
            temp=tempFile.read()
        gpu_temp_value=float(temp)/1000
        if gpu_temp_value > gpu_temp_peak:
            gpu_temp_peak = gpu_temp_value
        return gpu_temp_value

def cpu_temperature():
        global cpu_temp_peak
        tempLoadFile1="/sys/devices/virtual/thermal/thermal_zone1/temp"	
        with open(tempLoadFile1,'r') as tempFile1:
            temp1=tempFile1.read()
        cpu_temp_value=float(temp1)/1000
        if cpu_temp_value > cpu_temp_peak:
            cpu_temp_peak = cpu_temp_value
        return cpu_temp_value


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    # init loop
    full_scrn = False
    fps = 0.0
    tic = time.time()
    x_scale = cam.img_width / 1920
    y_scale = cam.img_height / 1080

    cols = 2
    for i in range(2,args.tile_size):
        if args.tile_size % i == 0:
            cols = i
    rows = args.tile_size//cols

    tile_width = cam.img_width//cols
    tile_height = cam.img_height//rows

    cols -= 1
    rows -= 1

    # open video writer
    file_name = args.video.split('/')[-1].split('.')[0] 
    if args.save:
        output = args.output + file_name + '_annotated.avi'
        frame_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (cam.img_width, cam.img_height))

    # open annotation ground truth
    if args.benchmark:
        try:
            path = args.ground + file_name + '.xgtf'
            tree = ET.parse(path)
            root = tree.getroot()
            tree_string = ET.tostring(root, encoding='utf8', method='xml')
            root = ET.fromstring(tree_string)
            ious = 0
            print('Ground truth: ' + path)
        except Exception as e:
            root = None
    
    # loop video
    frame = 1
    dets = []
    while True:

        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        beep = time.time()
        
        # PARSE FRAME
        if args.pattern == Pattern.STATIC: # STATIC PATTERN
        
            # tile image
            imgs = crop(img, tile_width, tile_height, args.tile_padding) 

      	    # loop and recorded active boxes
            col, row = 0, 0
            boxes = []
            confs = []
            clss = []
            vis_boxes = []
            for i in imgs:

                x = col*tile_width
                y = row*tile_height        
                x1 = max(x - args.tile_padding, 0)
                y1 = max(y - args.tile_padding, 0)

                # detect in images
                bb, cf, cl = trt_yolo.detect(i, conf_th)
                for j in range(len(bb)):
                    bounds = bb[j]
               	    box = (bounds[0]+x1, bounds[1]+y1, bounds[2]+x1, bounds[3]+y1)
                    boxes.append(box)
                    confs.append(cf[j])
                    clss.append(cl[j])

                if args.vis:
                    x2 = min(x + tile_width + args.tile_padding, img.shape[1])
                    y2 = min(y + tile_height + args.tile_padding, img.shape[0])        
                    vis_boxes.append((x1, y1, x2, y2))

                col+=1
                if col > cols:
                    col = 0
                    row += 1

            if args.vis:
                img = draw_boxes(img, vis_boxes, (0,0,255))
                    

        elif args.pattern == Pattern.ATTENTIVE: # ATTENTIVE PATTERN

            high_res_width = tile_width // 2
            high_res_height = tile_height // 2

            # low res tile
            low_res = crop(img, tile_width, tile_height, args.tile_padding) 

      	    # loop and recorded active boxes
            col, row = 0, 0
            boxes = []
            confs = []
            clss = []
            
            vis_crops = []
            vis_low = []
            vis_low_dets = []
            vis_high = []
            for i in low_res:

                x = col*tile_width
                y = row*tile_height  

                # get detections
                bb_low, cf_low, cl_low = trt_yolo.detect(i, conf_th)
                if len(bb_low) > 0:

                    crop_start = [max(x-args.tile_padding,0), max(y-args.tile_padding,0)]
                    crop_end = [min(x + tile_width + args.tile_padding, img.shape[1]), min(y + tile_height + args.tile_padding, img.shape[0])]
                    coord = crop_start[:]
                    
                    # high res tile
                    high_res = crop(i, high_res_width , high_res_height, args.tile_padding)

                    for j in high_res:

                        
                        x1 = max(coord[0] - args.tile_padding, crop_start[0])
                        y1 = max(coord[1] - args.tile_padding, crop_start[1])
                        
                        # final detections
                        bb_high, cf_high, cl_high = trt_yolo.detect(j, conf_th)
                        for k in range(len(bb_high)):
                            bounds = bb_high[k]
                            box = (bounds[0]+x1, bounds[1]+y1, bounds[2]+x1, bounds[3]+y1)
                            boxes.append(box)
                            confs.append(cf_high[k])
                            clss.append(cl_high[k])
                        
                        if args.vis:
                            x2 = min(coord[0] + high_res_width + args.tile_padding, crop_end[0])
                            y2 = min(coord[1] + high_res_height + args.tile_padding, crop_end[1])
                            vis_high.append((x1, y1, x2, y2))

                        coord[0] = coord[0] + high_res_width
                        if coord[0] >= crop_end[0]:
                            coord[0] = crop_start[0]
                            coord[1] = coord[1] + high_res_height
                            if coord[1] > crop_end[1]:
                                coord[1] = crop_end[1]

                if args.vis:
                    x1 = max(x-args.tile_padding,0)
                    y1 = max(y-args.tile_padding,0)
                    x2 = min(x + tile_width + args.tile_padding, img.shape[1])
                    y2 = min(y + tile_height + args.tile_padding, img.shape[0])    
                    if len(bb_low) > 0:
                        for bb in bb_low:
                            vis_low_dets.append((bb[0]+x1, bb[1]+y1, bb[2]+x1, bb[3]+y1))
                            vis_low.append((x1, y1, x2, y2))
                    else:
                        vis_crops.append((x1, y1, x2, y2))

                col+=1
                if col > cols:
                    col = 0
                    row += 1

            if args.vis:
                img = draw_boxes(img, vis_crops, (0,0,255))
                img = draw_boxes(img, vis_high, (255, 0, 0), 3, 0.1)   
                img = draw_boxes(img, vis_low, (0,0,255))
                img = draw_boxes(img, vis_low_dets, (0,0,255), 2, fill=0.2)          

        elif args.pattern == Pattern.PREDICTIVE: # PREDICTIVE PATTERN

            # tile image
            imgs = crop(img, tile_width, tile_height, args.tile_padding) 
            
            col, row = 0, 0
            boxes = []
            confs = []
            clss = []
            evaluated_tiles = [] # tiles that have already been evaluated 
            vis_crops = []
            vis_active = []
            vis_checking = []
            vis_inactive = []

            for i in imgs:
                
                x = col*tile_width
                y = row*tile_height 
                x1 = max(x-args.tile_padding,0)
                y1 = max(y-args.tile_padding,0)
                x2 = min(x + tile_width + args.tile_padding, img.shape[1])
                y2 = min(y + tile_height + args.tile_padding, img.shape[0]) 

                if args.vis:
                    vis_crops.append((x1,y1, x2, y2))
                
                # grab expected dets
                expected = {} # {name:count}
                for det in dets:
                    bounds = det[1]

                    # check overlap
                    bounds_x1 = bounds[0]
                    bounds_y1 = bounds[1]
                    bounds_x2 = bounds[2]
                    bounds_y2 = bounds[3]
 
                    if bounds_x1 < x2 - (2 * args.tile_padding) and bounds_x2 > x1 + (2 * args.tile_padding) and bounds_y1 < y2 - (2 * args.tile_padding) and bounds_y2 > y1 + (2 * args.tile_padding):
                        name = det[0]
                        if not name in expected:
                            expected[name] = 1
                        else:
                            expected[name] += 1

                # dynamically check crops from perimeter sweep
                # also perform worst-case scenario for first frame
                if frame == 1 or len(expected) > 0 or col == 0 or row == 0 or col == cols or row == rows:

                    names = []
                    bbs, cf, cl = trt_yolo.detect(i, conf_th)
                    if len(bbs) > 0:
                        if args.vis:
                            vis_active.append((x1, y1, x2, y2))
                        for j in range(len(bbs)):
                            bb = bbs[j]
                            boxes.append((bb[0]+x1, bb[1]+y1, bb[2]+x1, bb[3]+y1))
                            confs.append(cf[j])
                            clss.append(cl[j])
                            names.append(cls_dict.get(cl[j], 'CLS{}'.format(int(cl[j]))))
                            evaluated_tiles.append((col,row))
                    elif args.vis:
                        vis_inactive.append((x1, y1, x2, y2))

                    # missing detections from expected
                    for name in expected:
                        if expected[name] > names.count(name):
                            eval_tiles = []
                            if args.vis:
                                vis_checking.append((x1, y1, x2, y2))

                            # CHECK CORNERS
                            if col == 0 and row == 0: # top-left corner
                                eval_tiles.append((1,1))
                            elif col == 0 and row == rows: # bottom-left corner
                                eval_tiles.append((1, rows-1))
                            elif col == cols and row == 0: # top-right corner
                                eval_tiles.append((cols-1, 1))
                            elif col == cols and row == rows: # bottom-right corner
                                eval_tiles.append((cols-1, rows-1))

                            # CHECK BORDERS   
                            elif (row == 0 and col > 0 and col < cols): # top border
                                if col != 1:
                                    eval_tiles.append((col-1, row+1))
                                if col != cols - 1:
                                    eval_tiles.append((col+1, row+1))
                                eval_tiles.append((col, row+1))
                            elif (row == rows and col > 0 and col < cols): # bottom border
                                if col != 1:
                                    eval_tiles.append((col-1, row-1))
                                if col != cols - 1:
                                    eval_tiles.append((col+1, row-1))
                                eval_tiles.append((col, row-1))
                            elif (col == 0 and row > 0 and row < rows): # left border
                                if row != 1:
                                    eval_tiles.append((col+1, row-1))
                                if row != rows - 1:
                                    eval_tiles.append((col+1, row+1))
                                eval_tiles.append((col+1, row))
                            elif (col == cols and row > 0 and row < rows): # right border
                                if row != 1:
                                    eval_tiles.append((col-1, row-1))
                                if row != rows - 1:
                                    eval_tiles.append((col-1, row+1))
                                eval_tiles.append((col-1, row))

                            # CHECK CENTER
                            else:
                                if col + 1 < cols: # right
                                    eval_tiles.append((col + 1, row))
                                if col - 1 > 0: # left
                                    eval_tiles.append((col-1, row))
                                if row - 1 > 0: # top
                                    eval_tiles.append((col, row-1))
                                if row + 1 < rows: # bottom
                                    eval_tiles.append((col, row+1))
                                
                                if col + 1 < cols and row - 1 > 0: # top right
                                    eval_tiles.append((col + 1 , row - 1))
                                if col - 1 > 0 and row - 1 > 0: # top left
                                    eval_tiles.append((col-1,row-1))
                                if col + 1 < cols and row + 1 < rows: # bottom right
                                    eval_tiles.append((col+1, row+1))
                                if col - 1 > 0 and row + 1 < rows: # bottom left 
                                    eval_tiles.append((col-1, row+1))
                                
                            # Evaluate new quads
                            for tile in eval_tiles:
                                if evaluated_tiles.count(tile) > 0:
                                   continue
                                
                                temp_col, temp_row = 0, 0
                                for eval_img in imgs:
                                    if temp_col == tile[0] and temp_row == tile[1]:
                                         
                                         tile_x = tile[0] * tile_width
                                         tile_y = tile[1] * tile_height
                                         tile_x1 = max(tile_x-args.tile_padding, 0)
                                         tile_y1 = max(tile_y-args.tile_padding, 0)
                                         tile_x2 = min(tile_x + tile_width + args.tile_padding, img.shape[1])
                                         tile_y2 = min(tile_y + tile_height+args.tile_padding, img.shape[0]) 
                                         bb, cf, cl = trt_yolo.detect(eval_img, conf_th)
                                         evaluated_tiles.append(tile)
                                         if len(bb) > 0:
                                             if args.vis:
                                                 vis_active.append((tile_x1, tile_y1, tile_x2, tile_y2))
                                             for k in range(len(bb)):
                                                 bounds = bb[k]
                                                 boxes.append((bounds[0]+tile_x1, bounds[1]+tile_y1, bounds[2]+tile_x1, bounds[3]+tile_y1))
                                                 confs.append(cf[k])
                                                 clss.append(cl[k])
                                         elif args.vis:
                                             vis_checking.append((tile_x1, tile_y1, tile_x2, tile_y2))
                                       
                                    temp_col += 1
                                    if temp_col > cols:
                                        temp_col = 0
                                        temp_row += 1
                                    
                col+=1
                if col > cols:
                    col = 0
                    row += 1

            if args.vis:
                if frame == 1:
                    img = draw_boxes(img, vis_crops, (255,0,0), fill = 0.2)
                else:
                    img = draw_boxes(img, vis_inactive, (100,100,100), fill = 0.2)
                    img = draw_boxes(img, vis_active, (255,0,0), fill = 0.2)
                    img = draw_boxes(img, vis_checking, (0,0,255), fill = 0.2)            

        else:
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
        
        # FINISH UP

        # draw img
        img = vis.draw_bboxes(img, boxes, confs, clss)

        # save to file
        if args.save:
            frame_writer.write(img)

        # record dets     
        n = 0
        dets = []
        for box in boxes:
            dets.append([cls_dict.get(clss[n], 'CLS{}'.format(int(clss[n]))), box])
            n += 1

        # run benchmark on frame
        if args.benchmark:
            global avg_fps, avg_latency, avg_loss, avg_iou, avg_cpu_percent, avg_gpu_percent, avg_cpu_temp, avg_gpu_temp, cpu_peak, gpu_peak, gpu_temp_peak, cpu_temp_peak

            # fps
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)

            # latency
            latency = toc - beep 
           
            # iou, accuracy from ground truth
            # --- This is parsing for .xgtf format as provided by the PEVID high res dataset --- #
            hits, misses = 0, 0
            iou = 0
            if root:
                truth_boxes = []
                truth_confs = []
                truth_clss = []
                for child in root.getchildren()[1].iter():
                        
                    # find object and check class name
                    if child.tag == '{http://lamp.cfar.umd.edu/viper#}object':

                        # check frame
                        for child_frame_span in child.attrib['framespan'].split(' '):
                            child_span = child_frame_span.split(':')
                            if frame >= int(child_span[0]) and frame <= int(child_span[1]):
                                    
                                for child_bbox in child.getchildren()[0].iter():
                                    if child_bbox.tag == '{http://lamp.cfar.umd.edu/viperdata#}bbox':
                                        for box_frame_span in child_bbox.attrib['framespan'].split(' '):
                                            box_span = box_frame_span.split(':')
                                            if frame >= int(box_span[0]) and frame <= int(box_span[1]):
                                                
                                                # ground truth boxes
                                                hits += 1
                                                child_name = child.attrib['name'].lower()
                                                bx1 = int(child_bbox.attrib['x'])
                                                by1 = int(child_bbox.attrib['y'])
                                                width = int(child_bbox.attrib['width'])
                                                height = int(child_bbox.attrib['height'])
                                                bx2 = bx1 + width
                                                by2 = by1 + height

                                                # upscale bbox 1920x1080 => video size
                                                x_diff = (x_scale * width) - width
                                                y_diff = (y_scale * height) - height
                                                bx1 = int(np.round(bx1 * x_scale))
                                                by1 = int(np.round(by1 * y_scale))
                                                bx2 = int(np.round(bx2 * x_scale))
                                                by2 = int(np.round(by2 * y_scale))

                                                truth_boxes.append((bx1,by1,bx2,by2))
                                                truth_clss.append('')
                                                truth_confs.append(child_name)
                                
                                                # compare detections
                                                missed = True
                                                if len(dets) > 0:
                                                    for det in dets:
                                                        name = det[0]

                                                        # check name
                                                        if name != child_name:
                                                            continue

                                                        # check overlap
                                                        ax1 = det[1][0]
                                                        ay1 = det[1][1]
                                                        ax2 = det[1][2]
                                                        ay2 = det[1][3]

                                                        left = max(ax1, bx1)
                                                        right = min(ax2, bx2)
                                                        top = max(ay1, by1)
                                                        bot = min(ay2, by2)
 
                                                        if right < left or bot < top:
                                                            continue

                                                        # calculate iou
                                                        hits += 1
                                                        intersection = (right - left) * (bot - top)
                                                        areaa = (ax2 - ax1) * (ay2 - ay1)
                                                        areab = (bx2 - bx1) * (by2 - by1)
                                                        iou = intersection / float(areaa + areab - intersection)
                                                        avg_iou = ((avg_iou * ious) + iou) / (ious + 1)
                                                        missed = False
                                                        ious += 1
                                                if missed:
                                                    misses += 1

                # draw ground truth
                img = vis.draw_bboxes(img, truth_boxes, truth_confs, truth_clss)
         
            # benchmark hardware
            cpu_load = cpu_usage()
            gpu_load = gpu_usage()
            cpu_temp = cpu_temperature()
            gpu_temp = gpu_temperature()

            # average values
            avg_fps =  ((avg_fps * frame) + fps) / (frame + 1)
            avg_latency = ((avg_latency * frame) + latency) / (frame + 1)
            avg_loss = ((avg_loss * frame) + ((misses/hits) if hits > 0 else 0)) / (frame + 1)     
            avg_cpu_percent = ((avg_cpu_percent * frame) + cpu_load) / (frame + 1)
            avg_cpu_temp = ((avg_cpu_temp * frame) + cpu_temp) / (frame + 1)        
            avg_gpu_percent = ((avg_gpu_percent * frame) + gpu_load) / (frame + 1)    
            avg_gpu_temp = ((avg_gpu_temp * frame) + gpu_temp) / (frame + 1)
           
            tic = toc

        # show image
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        elif key == 32: # Space key: pause program
             while cv2.waitKey(1) != 32:
                 continue
        frame += 1
    if args.save:
        frame_writer.release()
        print('Saved to: ' + output)

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument('--input', type=str, default='input/',
                        help='input directory, e.g. input/')
    parser.add_argument('--output', type=str, default='output/',
                        help='output directory to save annotated files, e.g. output/')
    parser.add_argument('--ground', type=str, default='ground/',
                        help='ground truth directory, e.g. ground/')
    parser.add_argument('--tile_size', type=int, default=4,
      help='size of cropped tiles')
    parser.add_argument('--tile_padding', type=int, default=0,
      help='size of overlap tolerance bordering crops')
    parser.add_argument('--pattern', type=int, default=0,
      help='0 - NONE'
           '1 - STATIC' 
           '2 - ATTENTIVE'
           '3 - PREDICTIVE')
    parser.add_argument('--benchmark', type=bool, default=False,
      help='set whether to benchmark performance on device')
    parser.add_argument('--runs', type=int, default=1,
      help='number of runs to perform')
    parser.add_argument('--vis', type=bool, default=False,
      help='whether or not to visualize the algorithm')
    parser.add_argument('--save', type=bool, default=False,
      help='whether or not write annotated frames to output folder')
    args = parser.parse_args()
    print('\n')
    return args


def main():

    # parse arguments
    global args
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    if args.tile_size == 0:        
        print("ERROR: tile_size cannot be zero")
        return
    elif args.tile_size > 2:
        prime_flag = 0
        for i in range(2, int(sqrt(args.tile_size))+1):
            if (args.tile_size % i) == 0:
                prime_flag = 1
                break
        if prime_flag == 0:
            print("ERROR: tile_size cannot be prime")
            return

    run = 0
    start = time.time()
    global cls_dict
    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
    print('\n')

    if args.benchmark:
        global avg_fps, avg_latency, avg_loss, avg_iou, avg_cpu_percent, avg_cpu_raw, avg_gpu_percent, avg_gpu_raw, avg_cpu_temp, avg_gpu_temp, cpu_peak, gpu_peak, gpu_temp_peak, cpu_temp_peak
        avg_avg_fps = 0.0
        avg_avg_latency = 0.0
        avg_avg_loss = 0.0
        avg_avg_iou = 0.0
        avg_avg_cpu_percent = 0.0
        avg_avg_cpu_raw = 0.0
        avg_avg_cpu_temp = 0.0
        avg_avg_gpu_percent = 0.0
        avg_avg_gpu_temp = 0.0

    # run detection
    file_count = 0
    while (run < args.runs):
        for root, dirs, files in os.walk(args.input):
            for file_name in files:
                if not file_name.endswith("avi"):
                   continue
                file_count += 1

                # open camera
                print('Processing: ' + file_name)
                args.video = os.path.join(root, file_name)
                cam = Camera(args)
                if not cam.isOpened():
                    raise SystemExit('ERROR: failed to open camera!')
                open_window(WINDOW_NAME, 'Camera TensorRT YOLO Demo', cam.img_width, cam.img_height)
                   
                # loop
                loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)

                # average benchmarks
                if args.benchmark:
                    avg_avg_fps += avg_fps
                    avg_avg_latency += avg_latency
                    avg_avg_loss += avg_loss
                    avg_avg_iou += avg_iou
                    avg_avg_cpu_percent += avg_cpu_percent
                    avg_avg_cpu_raw += avg_cpu_raw
                    avg_avg_cpu_temp += avg_cpu_temp
                    avg_avg_gpu_percent += avg_gpu_percent
                    avg_avg_gpu_temp += avg_gpu_temp
                   
                    avg_fps = 0.0
                    avg_latency = 0.0
                    avg_loss = 0.0
                    avg_iou = 0.0
                    avg_cpu_percent = 0.0
                    avg_cpu_raw = 0.0
                    avg_cpu_temp = 0.0
                    avg_gpu_percent = 0.0
                    avg_gpu_temp = 0.0
            
                # reset run
                cam.release()
        run += 1
    
    cv2.destroyAllWindows()

    # print benchmark stats
    if args.benchmark:
        print("\nBenchmark Complete\n----------------\n")
        print("[TEST INFO]  %i Files annotated" %(file_count))
        print("[TEST INFO]  Runs: %i" %(args.runs))
        print("[TEST INFO]  Total Time: %.2f\n" %(time.time() - start))
        print("[INFO] Batch size: %i" %(batch_size))
        print("[INFO] AVG FPS: %.2f" %(avg_avg_fps / args.runs / file_count))
        print("[INFO] AVG Inference Time: %.3f" %((1 / avg_avg_fps) / args.runs/ file_count))
        print("[INFO] AVG Latency: %.2f" %(avg_avg_latency / args.runs/ file_count))
        print("[INFO] AVG Loss Rate: %.2f" %(avg_avg_loss / args.runs/ file_count))
        print("[INFO] AVG IOU:%.2f\n" %(avg_avg_iou / args.runs/ file_count))
        print("[HARDWARE] AVG CPU Usage: %.1f | %.0f MB, Peak CPU Usage: %.1f | %.0f MB" %(avg_avg_cpu_percent / args.runs/ file_count, avg_avg_cpu_raw / args.runs/ file_count, cpu_peak, cpu_raw_peak))
        print("[HARDWARE] AVG GPU Usage: %.1f, Peak GPU Usage: %.1f | %.0f MB" %(avg_avg_gpu_percent /  args.runs/ file_count, gpu_peak, gpu_raw_peak))
        print("[HARDWARE] AVG CPU Temperature: %.1f, Peak CPU Temperature: %.1f" %(avg_avg_cpu_temp / args.runs/ file_count, cpu_temp_peak))
        print("[HARDWARE] AVG GPU Temperature: %.1f, Peak GPU Temperature: %.1f\n" %(avg_gpu_temp /  args.runs/ file_count, gpu_temp_peak))

        f = open(args.output+'benchmark_results.txt', 'w')
        f.write("[TEST INFO]  %i Files annotated\n" %(file_count))
        f.write("[TEST INFO]  Runs: %i\n" %(args.runs))
        f.write("[TEST INFO]  Total Time: %.2f\n\n" %(time.time() - start))
        f.write("[INFO] Batch size: %i\n" %(batch_size))
        f.write("[INFO] AVG FPS: %.2f\n" %(avg_avg_fps / args.runs/ file_count))
        f.write("[INFO] AVG Inference Time: %.3f\n" %((1 / avg_avg_fps) / args.runs/ file_count))
        f.write("[INFO] AVG Latency: %.2f\n" %(avg_avg_latency / args.runs/ file_count))
        f.write("[INFO] AVG Loss Rate: %.2f \n" %(avg_avg_loss / args.runs/ file_count))
        f.write("[INFO] AVG IOU:%.2f\n\n" %(avg_avg_iou / args.runs/ file_count))
        f.write("[HARDWARE] AVG CPU Usage: %.1f | %.0f MB, Peak CPU Usage: %.1f | %.0f MB\n" %(avg_avg_cpu_percent / args.runs/ file_count, avg_avg_cpu_raw / args.runs/ file_count, cpu_peak, cpu_raw_peak))
        f.write("[HARDWARE] AVG GPU Usage: %.1f, Peak GPU Usage: %.1f | %.0f MB\n" %(avg_avg_gpu_percent /  args.runs/ file_count, gpu_peak, gpu_raw_peak))
        f.write("[HARDWARE] AVG CPU Temperature: %.1f, Peak CPU Temperature: %.1f\n" %(avg_avg_cpu_temp / args.runs/ file_count, cpu_temp_peak))
        f.write("[HARDWARE] AVG GPU Temperature: %.1f, Peak GPU Temperature: %.1f\n" %(avg_avg_gpu_temp /  args.runs/ file_count, gpu_temp_peak))
        f.close()


if __name__ == '__main__':
    main()
