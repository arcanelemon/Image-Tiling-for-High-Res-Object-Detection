# Image Tiling (tensorrt_demos)

This is a custom fork of the [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) repository provided by user @jkjung. The purpose of this repository is to benchmark the performance of specialized image tiling algorithms to achieve peak performance and object detection accuracy on high resolution images and videos.

Supported hardware:

* NVIDIA Jetson
   - All NVIDIA Jetson Developer Kits, e.g. [Jetson AGX Orin DevKit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/#advanced-features), [Jetson AGX Xavier DevKit](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit), [Jetson Xavier NX DevKit](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit), Jetson TX2 DevKit, [Jetson Nano DevKit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit).
   - Seeed [reComputer J1010](https://www.seeedstudio.com/Jetson-10-1-A0-p-5336.html) with Jetson Nano and [reComputer J2021](https://www.seeedstudio.com/reComputer-J2021-p-5438.html) with Jetson Xavier NX, which are built with NVIDIA Jetson production module and pre-installed with NVIDIA [JetPack SDK](https://developer.nvidia.com/embedded/jetpack).
* x86_64 PC with modern NVIDIA GPU(s).  Refer to [README_x86.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/README_x86.md) for more information.

Table of contents
-----------------

* [Setup](#setup)
* [Usage](#usage)
* [Algorithms](#algo)
* [Testing](#test)

<a name="setup"></a>
Setup
------------

The code in this repository is a fork from the tensorrt_demos repository. For more in depth instructions, please refer to the tensorrt_demos repo or  [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/) as provided by @jkjung. Step-by-step details have been provided by Nitish Patil.

Step-by-step:

1. Installation

   ```shell
   Download and install the Jetson Nano 2GB SD Card image from NVIDIA https://developer.nvidia.com/embedded/downloads

   I used the JetPack 4.6 as TensorRT 8.0.1 is bundled with it
   ```

2. Initial Setup

   ```shell
   You can follow NVIDIA’s guide for the initial setup https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

   During the initial setup, make sure that you enable the 4GB swap file. The Jetson Nano only has 2GB of RAM and is bound to run out of RAM if you do not enable swap.

   If you are doing a headless setup (without display), you will run into issues with getting the VNC server running as NVIDIA’s guide does not take into count the headless VNC setup. My recommendation is that you use a display for the first setup, set the user to auto login and then use NVIDIA’s guide for VNC. https://developer.nvidia.com/embedded/learn/tutorials/vnc-setup

   If you don’t have a display for the initial setup, you can use xrdp instead, run sudo apt-get install xrdp and connect to the Jetson’s IP address using Windows’ Remote Desktop Connection application. Once in, you can move forward with the VNC Setup or keep using xrdp. (I would prefer VNC).

   Once you set the default user to autologin, you can VNC into the Jetson Nano at any time even without the display, provided the computer you want to VNC from and the Jetson Nano are in the same network.

   Important Note: On the first login screen, make sure to choose LXDE as the desktop environment as it’s light on RAM usage.
   ```

3. Setting up the requirements
   ```shell
   JetPack 4.6 comes with YOLOv3 demos installed but they have a lot of prerequisites such as protobuf missing which need to be built from source.

   Luckily, JK Jung has written a script to install all the prerequisites and build protobuf from source and we can use his repository.

   Note: Some commands may require sudo

   Edit the .bashrc file:

   sudo nano ~/.bashrc

   Add the following lines so that bash knows the CUDA path:

   $ export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

   $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64\

   ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

   Restart the terminal after the above

   Clone this repository: git clone https://github.com/jkjung-avt/tensorrt_demos.git

   Change to the directory where you cloned it:

   cd <your path here>/tensorrt_demos/yolo

   Install PyCuda:

   sudo ./install_pycuda.sh

   Download and build Protobuf from source:

   wget https://raw.githubusercontent.com/jkjung-avt/jetson_nano/master/install_protobuf-3.8.0.sh

   chmod +x install_protobuf-3.8.0.sh

   sudo ./install_protobuf-3.8.0.sh

   Now you can install ONNX. Make sure you get version 1.4.1.

   sudo pip3 install onnx==1.4.1

   Change directory to the plugins folder:

   cd <your path here>/tensorrt_demos/plugins

   Run the makefile

   make

   Change directory to the yolo folder:

   cd <your path here>/tensorrt_demos/yolo

   Download the yolo weights manually or you can run this script to download them all:

   ./download_yolo.sh

   Convert your desired weights from darknet to ONNX:

   python3 yolo_to_onnx.py -m <model name>

   Convert the ONNX weights to TensorRT:

   python3 onnx_to_tensorrt.py -m <model name>
   ```

4. Run it
   ```shell
   You will need to connect to the Jetson via VNC and run the below commands on the terminal through VNC.

   Change directory to the project folder:

   cd <your path here>/tensorrt_demos/

   Download an image and run your model with:

   python3 trt_yolo.py --image ${HOME}/Pictures/yourimage.jpg -m <model name
   ```

<a name="usage"></a>
Usage
------------
For image tiling, all execution is handled in <b><i>trt_yolo.py</i></b> Standard operating usage can be found in the [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) repository. However, several new arguments have been added. Instructions on these new parameters are outlined below. It should be noted that users now must include target images/video under the input directory instead of referencing it directly as an argument.

Example Usage
```shell
python trt_yolo.py --m yolov3_tiny --benchmark 1 --vis 1 --pattern 2 --tile_size 4 -- tile_padding 15
```

### <b>--input</b> (string)
The relative file path of input images/videos.
> Default: /input

### <b>--output</b> (string)
The relative file path to save benchmark results and annotated videos to.
> Default: /output/

### <b>--ground</b> (string)
The relative file path containing ground truth data to perform benchmark analysis with.
> Default: /ground/

### <b>--benchmark</b> (bool)
Sets whether or not to run performance and efficiency benchmarks when active.
> True - run benchmarks<br>
> False - do not run benchmarks<br>
><br> Default: False

### <b>--vis</b> (bool)
Sets whether or not to visualize tiling methods by annotating images/videos. 
> True - visualize tiling pattern<br>
> False - do not visualize tiling pattern<br>
><br> Default: False

### <b>--pattern</b> (int)
Sets the tiling method utilized while running.
> 0 - None<br>
> 1 - Static Tiling Pattern<br>
> 2 - Attentive Tiling Pattern<br>
> 3 - Dynamic Tiling Pattern <br>
><br> Default: 0

### <b>--tile_size</b> (int)
The number of crops to tile video/image with.
> Default: 4

### <b>--tile_padding</b> (int)
The number of pixels that overlap between tiles.
> Default: 0

<a name="algo"></a>
Algorithms
------------

Below is a breakdown of the algorithms used for the different image tiling patterns used

* [Static](#static)
* [Attentive](#attentive)
* [Predictive](#predictive)

<a name="static"></a>
## Static

Static image tiling allows a high-resolution image to
be processed as several lower resolution image crops that are overlayed onto the base image. Typically,
crops contain a padding value that specifies the overlap between crops to prevent misses between
intersections. The resulting tiles contribute to a general increase in detection accuracy, at the expense of a
device’s performance. This process can cause lengthy delays and will always have a time complexity
proportional to the crop size.  
   ```python
   # tile image
   imgs = crop(img, tile_width, tile_height, args.tile_padding) 

   # loop and recorded active boxes
   col, row = 0, 0
   boxes = [] # active detection bounds evaluated by yolo
   confs = [] # confidence of detections 
   clss = [] # class of detections (i.e. person)
   for i in imgs:

         x = col*tile_width
         y = row*tile_height        
         x1 = max(x - args.tile_padding, 0)
         y1 = max(y - args.tile_padding, 0)

         # detect in images
         bb, cf, cl = trt_yolo.detect(i, conf_th)
         for j in range(len(bb)):
            bounds = bb[j]
               box = (bounds[0]+x1, bounds[1]+y1, bounds[2]+x1, bounds[3]+y1) # add cummulative position to evaulated bounds
            boxes.append(box)
            confs.append(cf[j])
            clss.append(cl[j])

         col+=1
         if col > cols:
            col = 0
            row += 1
   ```

<a name="attentive"></a>
## Attentive

The Attentive tiling process begins by
performing a low-resolution crop over the base image. These crops are then scaled to the network
dimensions within the model. After processing, the active bounding boxes are collected and crops
containing bounding boxes are split into smaller high-resolution crops. These new crops are then passed
into the network. The result is more precise image classification and object detection while maintaining
greater performance than the static approach on the device.
   ```python
   # dimensions of high res crops (this is configurable)
   high_res_width = tile_width // 2
   high_res_height = tile_height // 2

   # low res tile
   low_res = crop(img, tile_width, tile_height, args.tile_padding) 

   # loop and recorded active boxes
   col, row = 0, 0
   boxes = [] # active detection bounds evaluated by yolo
   confs = [] # confidence of detections 
   clss = [] # class of detections (i.e. person)

   # evaluate low res crops
   for i in low_res:

         x = col*tile_width
         y = row*tile_height  

         # get detections
         bb_low, cf_low, cl_low = trt_yolo.detect(i, conf_th)

         # begin high res evaluation
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

               coord[0] = coord[0] + high_res_width
               if coord[0] >= crop_end[0]:
                     coord[0] = crop_start[0]
                     coord[1] = coord[1] + high_res_height
                     if coord[1] > crop_end[1]:
                        coord[1] = crop_end[1]

         col+=1
         if col > cols:
            col = 0
            row += 1
   ```
<a name="predictive"></a>
## Predictive

Predictive tiling utilizes a simple predictive algorithm to create crops based off previous detections. First,
this model creates a border of crops equal to the desired dimension size. These crops are passed into the
network for detection. If a target is detected, its bounding box is recorded for the next pass. During
subsequent passes, new crops are predicted based on their proximity to the bounding box. Parameters may
be added to the algorithm to predict the pixels per frame (PPS) for more precise crop activation. This
effectively creates a perimeter around the image, only processing the information that is needed at any
given instance.
```python
# tile image
imgs = crop(img, tile_width, tile_height, args.tile_padding) 

col, row = 0, 0
boxes = [] # active detection bounds evaluated by yolo
confs = [] # confidence of detections 
clss = [] # class of detections (i.e. person)
evaluated_tiles = [] # tiles that have already been evaluated 

for i in imgs:
      
      x = col*tile_width
      y = row*tile_height 
      x1 = max(x-args.tile_padding,0)
      y1 = max(y-args.tile_padding,0)
      x2 = min(x + tile_width + args.tile_padding, img.shape[1])
      y2 = min(y + tile_height + args.tile_padding, img.shape[0]) 
      
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

      # dynamically check crops from perimeter sweep and expected dets
      # also perform worst-case scenario for first frame
      if frame == 1 or len(expected) > 0 or col == 0 or row == 0 or col == cols or row == rows:

         names = []
         bbs, cf, cl = trt_yolo.detect(i, conf_th)
         if len(bbs) > 0:
            for j in range(len(bbs)):
                  bb = bbs[j]
                  boxes.append((bb[0]+x1, bb[1]+y1, bb[2]+x1, bb[3]+y1))
                  confs.append(cf[j])
                  clss.append(cl[j])
                  names.append(cls_dict.get(cl[j], 'CLS{}'.format(int(cl[j]))))
                  evaluated_tiles.append((col,row))

         # missing detections from expected
         for name in expected:
            if expected[name] > names.count(name):
                  eval_tiles = []

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
                        # img is marked to be evaluated
                        if temp_col == tile[0] and temp_row == tile[1]:
                              
                              tile_x = tile[0] * tile_width
                              tile_y = tile[1] * tile_height
                              tile_x1 = max(tile_x-args.tile_padding, 0)
                              tile_y1 = max(tile_y-args.tile_padding, 0)
                              tile_x2 = min(tile_x + tile_width + args.tile_padding, img.shape[1])
                              tile_y2 = min(tile_y + tile_height+args.tile_padding, img.shape[0]) 

                              # pass img for evaluation
                              bb, cf, cl = trt_yolo.detect(eval_img, conf_th)
                              evaluated_tiles.append(tile)

                              # append new detections
                              if len(bb) > 0:
                                 for k in range(len(bb)):
                                       bounds = bb[k]
                                       boxes.append((bounds[0]+tile_x1, bounds[1]+tile_y1, bounds[2]+tile_x1, bounds[3]+tile_y1))
                                       confs.append(cf[k])
                                       clss.append(cl[k])
                           
                        temp_col += 1
                        if temp_col > cols:
                              temp_col = 0
                              temp_row += 1
                        
      col+=1
      if col > cols:
         col = 0
         row += 1
```

<a name="test"></a>
Testing
------------

A full breakdown of test results can be viewed in the report [Image_Tiling_Methodologies_and_Effectiveness](/Image%20Tiling%20Methodologies%20and%20Effectiveness.pdf). Previews of each algorithm in operation are also included under the [Preview](/preview/) folder. 