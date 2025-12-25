"""

LIST OF FUNCTIONS

LoadAndCrop
cropframe
Reference
Locate
TrackLocation
LocationThresh_View
ROI_plot
ROI_Location
Batch_LoadFiles
Batch_Process
PlayVideo
showtrace
Heatmap
DistanceTool
ScaleDistance

"""





########################################################################################

import os
import sys
import cv2
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import time
import warnings
import functools as fct

import bokeh
from scipy import ndimage
from tqdm import tqdm
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
from io import BytesIO
from IPython.display import clear_output, Image, display
hv.notebook_extension('bokeh')
warnings.filterwarnings("ignore")

from scipy.ndimage import minimum_filter


CLIP_BRIGHT_OBJECTS = True

########################################################################################    


def CountFrames(video_dict):
    """ 
    Returns:
        Count of actual frames in video (use this if we suspect metadata is not accurate)
        
    """
    
    #Upoad file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
            file=video_dict['fpath']))

    #Print video information. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    fps=cap.get(cv2.CAP_PROP_FPS)
    print('total frames from metadata: {frames}'.format(frames=cap_max))
    print(f'nominal fps: {fps}')
    print(f'Duration: {cap_max / fps / 60 / 60:.2f} hours')
    print('dimensions (h x w): {h},{w}'.format(
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

    video_dict['nominal_fps'] = fps
    
    #check for video p-frames
    check_p_frames(cap)
    
    #Set first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
    ret, frame = cap.read()

    if frame is None:
        print("Can't find starting frame.")
        return None, None
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    video_dict['f0'] = frame

    # If we don't trust the metadata, we can count actual frames. This will take a while, and usually is not necessary.
    frame_count = 0
    print('Frame count: ', end="")
    while ret:
        frame_count += 1
        if frame_count % 2500 == 0:
            print(f' {frame_count}', end="")
        ret, _ = cap.read()
            
    cap.release()
    print(f'\n\nActual frames: {frame_count}')
    video_dict['end'] = frame_count
    return frame_count


def LoadAndCrop(video_dict,cropmethod=None,fstfile=False,accept_p_frames=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Loads video and creates interactive cropping tool (video_dict['crop'] from first frame. In the 
    case of batch processing, the first frame of the first video is used. Additionally, 
    when batch processing, the same cropping parameters will be appplied to every video.  
    Care should therefore be taken that the region of interest is in the same position across 
    videos.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        cropmethod:: [str]
            Method of cropping video.  cropmethod takes the following values:
                None : No cropping 
                'Box' : Create box selection tool for cropping video
                
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        image:: [holoviews.Image]
            Holoviews hv.Image displaying first frame
            
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
    
    -------------------------------------------------------------------------------------
    Notes:
        - in the case of batch processing, video_dict['file'] is set to first 
          video in file 
        - prior cropping method HLine has been removed
    
    """
    
    #if batch processing, set file to first file to be processed
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    
    #Upoad file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
            file=video_dict['fpath']))

    #Print video information. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    fps=cap.get(cv2.CAP_PROP_FPS)
    print('total frames: {frames}'.format(frames=cap_max))
    print(f'nominal fps: {fps}')
    print(f'Duration: {cap_max / fps / 60 / 60:.2f} hours')
    print('dimensions (h x w): {h},{w}'.format(
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

    video_dict['nominal_fps'] = fps
    
    #check for video p-frames
    if accept_p_frames is False:
        check_p_frames(cap)
    
    #Set first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
    ret, frame = cap.read()

    if frame is None:
        print("Can't find starting frame.")
        return None, None
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    video_dict['f0'] = frame

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))

    print(f"Using bokeh version {bokeh.__version__}")
    
    if bokeh.__version__ < "3.4":
        graph_title = "Frame #1. Select region with mouse double-click, then drag"
    else:
        graph_title = "Frame #1. Select region using SHIFT+click+drag to start rectangle"
        
    image.opts(
        width=int(frame.shape[1]*video_dict['stretch']['width']),
        height=int(frame.shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,
        cmap='gray',
        colorbar=True,
        toolbar='below',
        title=graph_title
    )
    
    def centers_box(data):
        if data is None:
            return hv.Labels(([], [], []))
        try:
            x_ls = [(a + b)/2 for a, b in zip(data['x0'], data['x1'])]
            y_ls = [(a + b)/2 for a, b in zip(data['y0'], data['y1'])]
        except Exception as e:
            print(f'centers_box error: {e}')
            print(data)
            x_ls, y_ls = [], []
        rois = video_dict['crop_names'][:len(x_ls)]
        return hv.Labels((x_ls, y_ls, rois)).opts(text_color='black', background_fill_color='white')  # , background_fill_alpha=0)
    
    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod==None:
        image.opts(title="First Frame")
        video_dict['crop'] = None
        return image, video_dict
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box_stream = streams.BoxEdit(source=box,num_objects=len(video_dict['crop_names']))
        box.opts(alpha=.5, active_tools=['box_edit'])
        dmap = hv.DynamicMap(centers_box, streams=[box_stream])
        video_dict['crop'] = box_stream
        return (image*box*dmap), video_dict
    
    

########################################################################################

def cropframe(frame, crop=None, crop_num=0):
    """ 
    -------------------------------------------------------------------------------------
    
    Crops passed frame with `crop` specification
    
    -------------------------------------------------------------------------------------
    Args:
        frame:: [numpy.ndarray]
            2d numpy array 
        crop:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.
    
    -------------------------------------------------------------------------------------
    Returns:
        frame:: [numpy.ndarray]
            2d numpy array
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    try:
        Xs=[crop.data['x0'][crop_num],crop.data['x1'][crop_num]]
        Ys=[crop.data['y0'][crop_num],crop.data['y1'][crop_num]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
        return frame[fymin:fymax,fxmin:fxmax]
    except:
        return frame
 
    
    
    

########################################################################################

def Reference(video_dict,num_frames=100,
              altfile=False,fstfile=False,frames=None,
             crop_num=0):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates reference frame by taking median of random subset of frames.  This has the 
    effect of removing animal from frame provided animal is not inactive for >=50% of
    the video segment.  
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        num_frames:: [uint]
            Number of frames to base reference frame on.
            
        altfile:: [bool]
            Specify whether alternative file than video to be processed will be
            used to generate reference frame. If `altfile=True`, it is expected
            that `video_dict` contains `altfile` key.
        
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        frames:: [np array]
            User defined selection of frames to use for generating reference
    
    -------------------------------------------------------------------------------------
    Returns:
        reference:: [numpy.array]
            Reference image. Median of random subset of frames.
        image:: [holoviews.image]
            Holoviews Image of reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
        - If `altfile` is specified, it will be used to generate reference.
    
    """
    
    #set file to use for reference
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    vname = video_dict.get("altfile","") if altfile else video_dict['file']    
    fpath = os.path.join(os.path.normpath(video_dict['dpath']), vname)
    if os.path.isfile(fpath):
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    #Get video dimensions with any cropping applied
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)

    frame = cropframe(
        frame, 
        video_dict.get('crop'),
        crop_num
    )
    h,w = frame.shape[0], frame.shape[1]
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    print(f"Will sample {num_frames} frames between {video_dict['start']} and {cap_max - 2}")
    
    #Collect subset of frames
    if frames is None:
        #frames = np.random.randint(video_dict['start'],cap_max,num_frames)
        frames = np.linspace(start=video_dict['start'], stop=cap_max - 2, num=num_frames)
        breakpoint()
    else:
        num_frames = len(frames) #make sure num_frames equals length of passed list

    collection = np.zeros((num_frames,h,w))  
    print(f"Using frames: ", end="")
    for (idx,framenum) in enumerate(frames):    
        grabbed = False
        while grabbed == False: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
            ret, frame = cap.read()
            if ret == True:
                print(f"{int(framenum)}, ", end="")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if (video_dict['dsmpl'] < 1):
                    gray = cv2.resize(
                        gray,
                        (
                            int(gray.shape[1]*video_dict['dsmpl']),
                            int(gray.shape[0]*video_dict['dsmpl'])
                        ),
                        cv2.INTER_NEAREST)
                gray = cropframe(
                    gray, 
                    video_dict.get('crop'),
                    crop_num
                )
                collection[idx,:,:]=gray
                grabbed = True
            elif ret == False:
                print(f'\nFailed to grab frame {framenum}, will reset to ', end="")
                framenum = np.random.randint(video_dict['start'],cap_max,1)[0]
                print(f'{framenum} to try again')
                pass
    cap.release() 
    print('\n')

    reference = np.percentile(collection, 90, axis=0)
    print(reference.shape)
    image = hv.Image((np.arange(reference.shape[1]),
                      np.arange(reference.shape[0]), 
                      reference)).opts(#width=int(reference.shape[1]*video_dict['stretch']['width']),
                                       #height=int(reference.shape[0]*video_dict['stretch']['height']),
                                       invert_yaxis=True,
                                       cmap='gray', clim=(0, 255),  # clim forces full 0-255 range
                                       colorbar=True, data_aspect=1,
                                       toolbar='below',
                                       title="Ref: " + video_dict['crop_names'][crop_num])

    _om = np.median(reference)

    if not 'reference' in video_dict:
        video_dict['reference'] = []
    if not 'overall_median' in video_dict:
        video_dict['overall_median'] = []
    if not 'clipped_reference' in video_dict:
        video_dict['clipped_reference'] = []

    while len(video_dict['reference']) <= crop_num:
        video_dict['reference'].append(None)
    while len(video_dict['overall_median']) <= crop_num:
        video_dict['overall_median'].append(None)
    while len(video_dict['clipped_reference']) <= crop_num:
        video_dict['clipped_reference'].append(None)

    video_dict['reference'][crop_num] = reference
    video_dict['overall_median'][crop_num] = _om

    # Remove pixels brighter than the median
    reference[reference > _om] = _om
    video_dict['clipped_reference'][crop_num] = reference

    return reference, image    


########################################################################################


KERNEL_SIZE = 5


def Locate(cap,tracking_params,video_dict,prior=None, clip=False, crop_num=0):
    """ 
    -------------------------------------------------------------------------------------
    
    Return location of animal in frame, in x/y coordinates. 
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.VideoCapture]
            OpenCV VideoCapture class instance for video.
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        prior:: [list]
            If window is being used, list of length 2 is passed, where first index is 
            prior y position, and second index is prior x position.
    
    -------------------------------------------------------------------------------------
    Returns:
        ret:: [bool]
            Specifies whether frame is returned in response to cv2.VideoCapture.read.
        
        dif:: [numpy.array]
            Pixel-wise difference from prior frame, after thresholding and
            applying window weight.
        
        com:: [tuple]
            Indices of center of mass as tuple in the form: (y,x).
        
        frame:: [numpy.array]
            Original video frame after cropping.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #attempt to load frame
    ret, frame = cap.read() 
    
    #set window dimensions
    if prior != None and tracking_params['use_window']==True:
        # This doesn't work with multiple animals
        window_size = tracking_params['window_size']//2
        ymin,ymax = prior[0]-window_size, prior[0]+window_size
        xmin,xmax = prior[1]-window_size, prior[1]+window_size

    if not ret:
        return ret, None, None, frame
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
            frame,
            (
                int(frame.shape[1]*video_dict['dsmpl']),
                int(frame.shape[0]*video_dict['dsmpl'])
            ),
            cv2.INTER_NEAREST)

#        ref1 = video_dict['reference'][crop_num]
#        ref2 = video_dict['clipped_reference'][crop_num]
    
        #find difference from reference
    def find_diff(frame, tracking_params,video_dict,prior, clip, crop_num):
        if tracking_params['method'] == 'abs':
            dif = np.absolute(frame-video_dict['reference'][crop_num])
        elif tracking_params['method'] == 'light':
            dif = frame-video_dict['reference'][crop_num]
        elif tracking_params['method'] == 'dark':
            # Remove pixels brighter than median
            if clip:
                _ref = video_dict['clipped_reference'][crop_num]
                _om = video_dict['overall_median'][crop_num]
                frame[frame > _om] = _om
            else:
                _ref = video_dict['reference'][crop_num]

            dif = _ref-frame
        dif = dif.astype('int16')
        if 'mask' in video_dict.keys():
            if video_dict['mask']['mask'] is not None:
                    dif[video_dict['mask']['mask']] = 0
    
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint16)
        dif = minimum_filter(dif, footprint=kernel)
              
        #apply window
        weight = 1 - tracking_params['window_weight']
        if prior != None and tracking_params['use_window']==True:
            dif = dif + (dif.min() * -1) #scale so lowest value is 0
            dif_weights = np.ones(dif.shape)*weight
            dif_weights[slice(ymin if ymin>0 else 0, ymax),
                        slice(xmin if xmin>0 else 0, xmax)]=1
            dif = dif*dif_weights
            
        #threshold differences and find center of mass for remaining values
        dif[dif<np.percentile(dif,tracking_params['loc_thresh'])]=0
        
        #remove influence of wire
        if tracking_params['rmv_wire'] == True:
            ksize = tracking_params['wire_krn']
            kernel = np.ones((ksize,ksize),np.uint8)
            dif_wirermv = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
            krn_violation =  dif_wirermv.sum()==0
            dif = dif if krn_violation else dif_wirermv
            if krn_violation:
                print("WARNING: wire_krn too large. Reverting to rmv_wire=False for frame {x}".format(
                    x= int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-video_dict['start'])))
            
        com=ndimage.measurements.center_of_mass(dif)
        return dif, com, frame

    if crop_num < 0:
        if 'num_animals' in video_dict:
            n = video_dict['num_animals']
            start_animal = 0
        else:
            n = 1
            start_animal = 0
    else:
        n = 1
        start_animal = crop_num

    end_animal = start_animal + n
        
    # Loop through all animals
    dif = [None] * n
    com = [None] * n

    idx = 0
    for x in range(start_animal, end_animal):
        frame_crop = cropframe(frame, video_dict.get('crop'), crop_num=x)
        dif[idx], com[idx], frame_out = find_diff(frame_crop, tracking_params,video_dict,prior, clip, crop_num=x)
        idx = idx + 1

    return ret, dif, com, frame_out  # Note that only one frame is returned, which is the last one. Currently the only code that uses this return value calls it with n=1, so this is OK for now.

    
########################################################################################        

def TrackLocation(video_dict,tracking_params):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame in video define location of animal, in x/y coordinates, and distance
    travelled from previous frame.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array] 
                              
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]    
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """

    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])#set file
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']) #set starting frame
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max 

    n = video_dict['num_animals']
    n_frames = cap_max - video_dict['start'] + 1
    
    #Initialize vector to store motion values in
    X = np.zeros((n, n_frames))
    Y = np.zeros((n, n_frames))
    D = np.zeros((n, n_frames))

    print(f"Reading {n_frames} frames\n")

    #Loop through frames to detect frame by frame differences
    time.sleep(.2) #allow printing

    for f in tqdm(range(n_frames)):
        
        if f>0: 
            yprior = np.around(Y[:, f-1]).astype(int)
            xprior = np.around(X[:, f-1]).astype(int)
            ret,dif,com, _ = Locate(cap,tracking_params,video_dict,prior=[yprior,xprior], clip=CLIP_BRIGHT_OBJECTS, crop_num=-1)
        else:
            ret,dif,com, _ = Locate(cap,tracking_params,video_dict, clip=CLIP_BRIGHT_OBJECTS, crop_num=-1)

        if not ret:
            #if no frame is detected
            f = f-1
            X = X[:, :f] #Amend length of X vector
            Y = Y[:, :f] #Amend length of Y vector
            D = D[:, :f] #Amend length of D vector
            break
        
        for x in range(n):
            # Use center of mass coordinates. Currently we ignore dif
            Y[x, f] = com[x][0]
            X[x, f] = com[x][1]
            if f>0:
                D[x, f] = np.sqrt((Y[x, f]-Y[x, f-1])**2 + (X[x, f]-X[x, f-1])**2)

    #release video
    cap.release()
    time.sleep(.2) #allow printing

    n_frames_processed = D.shape[1]
    
    print('total frames processed: {f}\n'.format(f=n_frames_processed))

    #create pandas dataframe
    df = pd.DataFrame(
    {'File' : video_dict['file'],
     'Location_Thresh': np.ones(n_frames_processed)*tracking_params['loc_thresh'],
     'Use_Window': str(tracking_params['use_window']),
     'Window_Weight': np.ones(n_frames_processed)*tracking_params['window_weight'],
     'Window_Size': np.ones(n_frames_processed)*tracking_params['window_size'],
     'Start_Frame': np.ones(n_frames_processed)*video_dict['start'],
     'Frame': np.arange(n_frames_processed)
    })

    for x in range(n):
        xname = 'X' + str(x)
        yname = 'Y' + str(x)

#        breakpoint()
        
        df[xname] = X[x,:]
        df[yname] = Y[x,:]
        
    for x in range(n):
        Dname = 'Dist_px' + str(x)
        df[Dname] = D[x,:]

    #add region of interest info
    df = ROI_Location(video_dict, df) 
    if video_dict['region_names'] is not None:
        print('Defining transitions...')
        rnames = video_dict['region_names']
        if any(x not in df for x in rnames):
            # Remove any region names that were not defined, otherwise we will get error
            # later at the df[rnames] step
            rnames = list(x for x in rnames if x in df)
            print(f'Some ROI(s) are not defined. After removing from analysis, will analyze ROIs with names: {rnames}')

        if len(rnames) > 0:
            # Add column that has region name
            df['ROI_location'] = ROI_linearize(df[rnames])
            # Add column of True/False for transitions from one to another
            df['ROI_transition'] = ROI_transitions(df['ROI_location'])
    
    #update scale, if known
    df = ScaleDistance(video_dict, df=df, column='Dist_px0')
       
    return df





########################################################################################

def LocationThresh_View(video_dict,tracking_params,examples=4, crop_num=0):
    """ 
    -------------------------------------------------------------------------------------
    
    Display example tracking with selected parameters for a random subset of frames. 
    NOTE that because individual frames are analyzed independently, weighting 
    based upon prior location is not implemented.
    
    -------------------------------------------------------------------------------------
    Args:
  
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
            
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int] 
                           
        examples:: [uint]
            The number of frames for location tracking to be tested on.
        
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [holoviews.Layout]
            Returns Holoviews Layout with original images on left and heat plots with 
            animal's estimated position marked on right.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence display and not
          calculation
    
    """
    
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #examine random frames
    images = []
    for example in range (examples):
        
        #analyze frame
        ret = False
        while ret is False:     
            frm=np.random.randint(video_dict['start'],cap_max) #select random frame
            cap.set(cv2.CAP_PROP_POS_FRAMES,frm) #sets frame to be next to be grabbed
            ret,dif,com,frame = Locate(cap, tracking_params, video_dict, clip=example > 0, crop_num=crop_num)

        # Since we only tracked one animal, convert list to a single item
        dif = dif[0]

        #plot original frame
        image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
        image_orig.opts(
            #width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
            #height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True,cmap='gray',toolbar='below',
            data_aspect=1,  # make sure pixels are square
            title=video_dict['crop_names'][crop_num] + ": " + str(frm))
        orig_overlay = image_orig * hv.Points(([com[0][1]],[com[0][0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        #plot heatmap
        dif = dif*(255//dif.max())
        image_heat = hv.Image((
            np.arange(dif.shape[1]), 
            np.arange(dif.shape[0]), 
            dif))
        image_heat.opts(
            #width=int(dif.shape[1]*video_dict['stretch']['width']),
            #height=int(dif.shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True,cmap='jet',toolbar='below',
            data_aspect=1,
            title=video_dict['crop_names'][crop_num] + ": " + str(frm - video_dict['start']))
        heat_overlay = image_heat * hv.Points(([com[0][1]],[com[0][0]])).opts(
            color='red',size=20,marker='+',line_width=3) 
        
        images.extend([orig_overlay,heat_overlay])
    
    cap.release()
    layout = hv.Layout(images)
    return layout





########################################################################################    
    
def ROI_plot(video_dict, roi_method="Poly"):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                                      
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define regions of interest.
        
        poly_stream:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            selection tool. `poly_stream.data` contains x and y coordinates of roi 
            vertices.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #get number of objects to be drawn
    nobjects = len(video_dict['region_names']) if video_dict['region_names'] else 0 

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(video_dict['reference'].shape[1]),
        np.arange(video_dict['reference'].shape[0]),
        video_dict['reference']))
    image.opts(
        width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
        height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,cmap='gray', colorbar=True,toolbar='below',
        title="No Regions to Draw" if nobjects == 0 else "Draw Regions: "+', '.join(video_dict['region_names']))

    def centers(data):
        try:
            x_ls, y_ls = data['xs'], data['ys']
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        rois = video_dict['region_names'][:len(xs)]
        return hv.Labels((xs, ys, rois))

    def centers_box(data):
        try:
            x_ls = [(a + b)/2 for a, b in zip(data['x0'], data['x1'])]
            y_ls = [(a + b)/2 for a, b in zip(data['y0'], data['y1'])]
        except Exception as e:
            print(f'error: {e}')
            x_ls, y_ls = [], []
        rois = video_dict['region_names'][:len(x_ls)]
        return hv.Labels((x_ls, y_ls, rois))
    

    if nobjects > 0:
        #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
        poly = hv.Polygons([])
        if roi_method == "Poly":
            poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
            poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])
            dmap = hv.DynamicMap(centers, streams=[poly_stream])
            return (image * poly * dmap), poly_stream
        elif roi_method == "Box":
            poly_stream = streams.BoxEdit(source=poly,num_objects=3)
            poly.opts(alpha=.5, active_tools=['box_edit'])
            dmap = hv.DynamicMap(centers_box, streams=[poly_stream])
            return (image * poly * dmap), poly_stream

    else:
        return (image),None
    

    
    
    
########################################################################################    

def ROI_Location(video_dict, df):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame, determine which regions of interest the animal is in.  For each
    region of interest, boolean array is added to `df` dataframe passed, with 
    column name being the region name.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        df:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Must contain column names 'X' and 'Y'.

    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            For each region of interest, boolean array is added to `df` dataframe 
            passed, with column name being the region name. Additionally, under column
            `ROI_coordinates`, coordinates of vertices of each region of interest are
            printed. This takes the form of a dictionary of x and y coordinates, e.g.:any
                'xs' : [[region 1 x coords], [region 2 x coords]],
                'ys' : [[region 1 y coords], [region 2 y coords]]
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """

    region_name_list = video_dict['region_names']
    
    if region_name_list == None:
        return df

    if 'roi_stream' not in video_dict or video_dict['roi_stream'] is None:
        # No ROIs were defined
        return df

    rsd = video_dict['roi_stream'].data

    #Create ROI Mask dictionary, key is ROI name
    ROI_mask_dict = {}

    num_polygons = len(rsd['xs'])

    IS_BOX = 'x0' in video_dict['roi_stream'].data   # Test whether we defined ROIs using box instead of polygon

    if IS_BOX:
        for idx_poly in range(num_polygons):
            # Loop through each polygon. Note that the number of polygons might be less than the number of region_names
            # if the user didn't bother to define all regions.
            x0 = rsd['x0'][idx_poly]
            y0 = rsd['y0'][idx_poly]
            x1 = rsd['x1'][idx_poly]
            y1 = rsd['y1'][idx_poly]
            mask = np.zeros(video_dict['reference'].shape) # create empty mask
            cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1) #fill rectangle
            ROI_mask_dict[region_name_list[idx_poly]] = mask==255 # convert to boolean, and save to ROI_mask_dict
    else:
        for idx_poly in range(num_polygons):
            # Loop through each polygon. Note that the number of polygons might be less than the number of region_names
            # if the user didn't bother to define all regions.
            x = np.array(rsd['xs'][idx_poly]) #x coordinates
            y = np.array(rsd['ys'][idx_poly]) #y coordinates
            xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
            mask = np.zeros(video_dict['reference'].shape) # create empty mask
            cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon  
            ROI_mask_dict[region_name_list[idx_poly]] = mask==255 # convert to boolean, and save to ROI_mask_dict

    if num_polygons < len(region_name_list):
        # Show warning
        print(f'Warning: specified {len(region_name_list)} region names, but only defined {num_polygons} polygons.')

    #Create arrays to store whether animal is within given ROI
    ROI_bool = {}
    for region_name in ROI_mask_dict:
        # Create dictionary of vectors.
        # key = region_name, value = vector of False, one per frame
        ROI_bool[region_name]=np.full(len(df['Frame']),False,dtype=bool)

    #For each frame assess truth of animal being in each ROI
    for f in df['Frame']:
        # Loop through frames
        y,x = df['Y'][f], df['X'][f]
        for region_name in ROI_mask_dict:
            # Loop through ROIs, assign True/False
            ROI_bool[region_name][f] = ROI_mask_dict[region_name][int(y),int(x)]
    
    #Add data to df data frame
    for region_name in ROI_bool:
        df[region_name]=ROI_bool[region_name]
    
    #Add ROI coordinates
    df['ROI_coordinates']=str(video_dict['roi_stream'].data)
    
    return df





########################################################################################        

def ROI_linearize(rois, null_name = 'non_roi'):
    
    """ 
    -------------------------------------------------------------------------------------
    
    Creates array defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Args:
        rois:: [pd.DataFrame]
            Pandas dataframe where each column corresponds to an ROI, with boolean values
            defining if animal is in said roi.
        null_name:: [string]
            Name used when animals is not in any defined roi.
    
    -------------------------------------------------------------------------------------
    Returns:
        rois['ROI_location']:: [pd.Series]
            pd.Series defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    region_names = rois.columns.values
    rois['ROI_location'] = null_name
    for region in region_names:
        # Loop through each region
        bool_vals = rois[region]
        # Use logical indexing to change only the rows for which animal is in current region
        rois['ROI_location'][bool_vals] = rois['ROI_location'][bool_vals].apply(
            lambda x: '_'.join([x, region]) if x!=null_name else region
        )
    return rois['ROI_location']






########################################################################################        

def ROI_transitions(regions, include_first=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates boolean array defining where transitions between each ROI occur.
    
    -------------------------------------------------------------------------------------
    Args:
        regions:: [Pandas Series]
            Pandas Series defining ROI as string for each frame
        include_first:: [string]
            Whether to count first frame as transition
    
    -------------------------------------------------------------------------------------
    Returns:
        transitions:: [Boolean array]
            pd.Series defining where transitions between ROIs occur.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    regions_offset = np.append(regions[0], regions[0:-1])
    transitions = regions!=regions_offset
    if include_first:
        transitions[0] = True
    return transitions





########################################################################################        
    
def Summarize_Location(location, video_dict, bin_dict=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates summary of distance travelled and proportional time spent in each region
    of interest according to user defined time bins.  If bins are not provided 
    (`bin_dict=None`), average of entire video segment will be provided.
    
    -------------------------------------------------------------------------------------
    Args:
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame.
      
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                              
        bin_dict:: [dict]
            Dictionary specifying bins.  Dictionary keys should be names of the bins.  
            Dictionary value for each bin should be a tuple, with the start and end of 
            the bin, in seconds, relative to the start of the analysis period 
            (i.e. if start frame is 100, it will be relative to that). If no bins are to 
            be specified, set bin_dict = None.
            example: bin_dict = {1:(0,100), 2:(100,200)}                             

    
    -------------------------------------------------------------------------------------
    Returns:
        bins:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #define bins
    avg_dict = {'all': (location['Frame'].min(), location['Frame'].max())}   
    bin_dict = bin_dict if bin_dict is not None else avg_dict
    
    #get summary info
    bins = (pd.Series(bin_dict).rename('range(f)')
            .reset_index().rename(columns=dict(index='bin')))    
    bins['Distance_px'] = bins['range(f)'].apply(
        lambda r: location[location['Frame'].between(*r)]['Distance_px'].sum())
    if video_dict['region_names'] is not None:
        bins_reg = bins['range(f)'].apply(
            lambda r: location[location['Frame'].between(*r)][video_dict['region_names']].mean())
        bins = bins.join(bins_reg)
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y'] + video_dict['region_names']
    else:
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y']
    bins = pd.merge(
        location.drop(drp_cols, axis='columns'),
        bins,
        left_index=True,
        right_index=True)
    
    #scale distance
    bins = ScaleDistance(video_dict,df=bins,column='Distance_px') 
    
    return bins





######################################################################################## 

def Batch_LoadFiles(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Populates list of files in directory (`dpath`) that are of the specified file type
    (`ftype`).  List is held in `video_dict['FileNames']`.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

    
    -------------------------------------------------------------------------------------
    Returns:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """

    #Get list of video files of designated type
    if os.path.isdir(video_dict['dpath']):
        video_dict['FileNames'] = sorted(os.listdir(video_dict['dpath']))
        video_dict['FileNames'] = fnmatch.filter(video_dict['FileNames'], ('*.' + video_dict['ftype'])) 
        return video_dict
    else:
        raise FileNotFoundError('{path} not found. Check that directory is correct'.format(
            path=video_dict['dpath']))

        
        
        
        
######################################################################################## 

def Batch_Process(video_dict,tracking_params,bin_dict,accept_p_frames=False):   
    """ 
    -------------------------------------------------------------------------------------
    
    Run LocationTracking on folder of videos of specified filetype. 
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
         accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        summary_all:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
            
        layout:: [hv.Layout]
            Holoviews layout wherein for each session the reference frame is returned
            with the regions of interest highlightted and the animals location across
            the session overlaid atop the reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    images = []
    for file in video_dict['FileNames']:
        
        print ('Processing File: {f}'.format(f=file))  
        video_dict['file'] = file 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), file)
        
        #Print video information. Note that max frame is updated later if fewer frames detected
        cap = cv2.VideoCapture(video_dict['fpath'])
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        print('total frames: {frames}'.format(frames=cap_max))
        print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
        print('dimensions (h x w): {h},{w}'.format(
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        
        #check for video p-frames
        if accept_p_frames is False:
            check_p_frames(cap)
        
        video_dict['reference'], image = Reference(video_dict,num_frames=50) 
        location = TrackLocation(video_dict,tracking_params)
        location.to_csv(os.path.splitext(video_dict['fpath'])[0] + '_LocationOutput.csv', index=False)
        file_summary = Summarize_Location(location, video_dict, bin_dict=bin_dict)
               
        try: 
            summary_all = pd.concat([summary_all,file_summary],sort=False)
        except NameError: 
            summary_all = file_summary
        
        trace = showtrace(video_dict,location)
        heatmap = Heatmap(video_dict, location, sigma=None)
        images = images + [(trace.opts(title=file)), (heatmap.opts(title=file))]

    #Write summary data to csv file
    sum_pathout = os.path.join(os.path.normpath(video_dict['dpath']), 'BatchSummary.csv')
    summary_all.to_csv(sum_pathout, index=False)
    
    layout = hv.Layout(images)
    return summary_all, layout


def GetFileBase(video_dict, full_path=True):
    output_file_base = os.path.splitext(video_dict['file'])[0]
    if full_path:
        return os.path.join(os.path.normpath(video_dict['dpath']), output_file_base)
    else:
        return output_file_base


########################################################################################        

def PlayVideo(video_dict,display_dict,location):
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying animal's estimated location. Video is played
    in notebook

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'resize' : Default is None, in which original size is retained.
                           Alternatively, set to tuple as follows: (width,height).
                           Because this is in pixel units, must be integer values.
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame. 
            
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """

    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
            
#        frame = cropframe(frame, video_dict['crop'])
        height, width = int(frame.shape[0]), int(frame.shape[1])

        output_file_base = GetFileBase(video_dict, full_path=False) + "_tracked.avi"
        output_file_base2 = GetFileBase(video_dict, full_path=False) + "_tracked_fast.avi"
        fps = video_dict['nominal_fps']
        fpath = os.path.join(os.path.normpath(video_dict['dpath']), output_file_base)
        fpath2 = os.path.join(os.path.normpath(video_dict['dpath']), output_file_base2)

        if fps == 0:
            print(f'Warning: unable to determine original frame rate, substituting 30 fps.')
            fps = 30
        elif fps < 5:
            # For very low frame rates, impose a minimum of 5fps
            fps = 5
        else:
            print(f'Writing video file, input frame rate {fps} fps: {fpath}')
            
        # fourcc = cv2.VideoWriter_fourcc(*'FFV1')   # Lossless. Not compatible with ImageJ, but readable by most other programs. Not very space-efficient
        # fourcc = cv2.VideoWriter_fourcc(*'jpeg')   # only writes up to 20 fps, though video read can be 30.
        # fourcc = cv2.VideoWriter_fourcc(*'FMP4')   # fragmented MP4. Uses H264 under the hood.
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')     # H264. 15x more efficient than FFV1. About 2kB per frame.
        writer = cv2.VideoWriter(fpath,
                                 fourcc, fps,
                                 (width, height),
                                 isColor=False)

        writer2 = cv2.VideoWriter(fpath2,
                                 fourcc, 30,
                                 (width, height),
                                 isColor=False)


    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    param_start = display_dict['start']
    param_stop = display_dict['stop']

    analysis_start = video_dict['start']
    analysis_end = video_dict['end']

    if analysis_start is None:
        analysis_start = 0
    if analysis_end is None:
        analysis_end = frame_count

    if (param_start is None) or (param_start < analysis_start):
        param_start = analysis_start
        
    if (param_stop is None) or (param_stop > analysis_end):
        param_stop = analysis_end
    
    #Initialize video play options   
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start'])
    print(f'Reading video frames {param_start}-{param_stop}')

    if analysis_start != 0:
        print(f'   (corresponding to analysis frames {param_start-analysis_start}-{param_stop-analysis_start})')

    #Play Video
    percent_reported = 0

    USE_EXT_VIEWER = True

    n = video_dict['num_animals']

    height = None
    
    for f in range(param_start, param_stop):
        ret, frame = cap.read() #read frame

        if ret == False:
            print(f'warning. failed to get video frame{f}')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
#        frame = cropframe(frame, video_dict['crop'])

        if height is None:
            height, width = int(frame.shape[0]), int(frame.shape[1])

        f2 = f - analysis_start   # Frame in analysis arrays

        if f2 >= len(location['X0']):
            # Reached end of location table
            break
        
        for x in range(n):
            x0 = video_dict['crop'].data['x0'][x] + location['X' + str(x)][f2]
            y0 = video_dict['crop'].data['y1'][x] + location['Y' + str(x)][f2]
            try:
                markposition = (int(x0),int(y0))
                cv2.drawMarker(img=frame,position=markposition,color=255)
            except:
                pass

        #Save video (if desired). 
        if display_dict['save_video']:
            writer.write(frame)

        if f % fps == 0:
            # Save every 15th frame to a second file
            writer2.write(frame)

        if f % 200 == 0:
            # Display every nth frame
            percent_done = (f - param_start) * 100.0 / (param_stop - param_start)
            if USE_EXT_VIEWER:
                cv2.putText(frame, f"frame {f}/{param_stop}", (5, 25), fontFace=2, fontScale=0.5, color=255)
                cv2.imshow("preview", frame)
                cv2.waitKey(1)
                if np.ceil(percent_done) > percent_reported:
                    print(".", end="")
                    percent_reported = np.ceil(percent_done)
            else:
                cv2.putText(frame, f"{percent_done:0.1f}% done", (5, 25), fontFace=2, fontScale=0.5, color=255)
                display_image(frame, display_dict['fps'],display_dict['resize'])

    if USE_EXT_VIEWER:
        print('\n')

    #Close video window and video writer if open
    print('Done playing segment')

    if USE_EXT_VIEWER:
        #Close video window and video writer if open
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    if display_dict['save_video']:
        print('Closing cv2.writer object')
        writer.release()
        writer2.release()


def display_image(frame,fps,resize):
    # Shows image in Jupyter notebook. This seems slower than cv2, which shows in external viewer
    img = PIL.Image.fromarray(frame, "L")
    img = img.resize(size=resize) if resize else img
    buffer = BytesIO()
    img.save(buffer,format="JPEG")    
    display(Image(data=buffer.getvalue()))
    time.sleep(1/fps)
    clear_output(wait=True)
        
    
    
########################################################################################

def showtrace(video_dict, location, color="red",alpha=.8,size=3, animal_num=0):
    """ 
    -------------------------------------------------------------------------------------
    
    Create image where animal location across session is displayed atop reference frame

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 

            
        color:: [str]
            Color of trace.  See Holoviews documentation for color options
                               
        alpha:: [float]
            Alpha of trace.  See Holoviews documentation for details
        
        size:: [float]
            Size of trace.  See Holoviews documentation for details.     
    
    -------------------------------------------------------------------------------------
    Returns:
        holoviews.Overlay
            Location of animal superimposed upon reference. If poly_stream is passed
            than regions of interest will also be outlined.
    
    -------------------------------------------------------------------------------------
    Notes:

    """

    video_dict['roi_stream'] = video_dict['roi_stream'] if 'roi_stream' in video_dict else None
    if video_dict['roi_stream'] != None:
        lst = []
        for poly in range(len(video_dict['roi_stream'].data['xs'])):
            x = np.array(video_dict['roi_stream'].data['xs'][poly]) #x coordinates
            y = np.array(video_dict['roi_stream'].data['ys'][poly]) #y coordinates
            lst.append( [ (x[vert],y[vert]) for vert in range(len(x)) ] )
        poly = hv.Polygons(lst).opts(fill_alpha=0.1,line_dash='dashed')
        
    image = hv.Image((np.arange(video_dict['reference'][animal_num].shape[1]),
                      np.arange(video_dict['reference'][animal_num].shape[0]),
                      video_dict['reference'][animal_num])
                    ).opts(#width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
                           #height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
                           invert_yaxis=True,cmap='gray',toolbar='below', data_aspect=1,
                           title="Motion Trace")
    
    points = hv.Scatter(np.array([location['X' + str(animal_num)],location['Y' + str(animal_num)]]).T).opts(color='red',alpha=alpha,size=size)
    
    return (image*poly*points) if video_dict['roi_stream']!=None else (image*points)





########################################################################################    

def Heatmap (video_dict, location, sigma=None, animal_num=0):
    """ 
    -------------------------------------------------------------------------------------
    
    Create heatmap of relative time in each location. Max value is set to maxiumum
    in any one location.

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
                
        sigma:: [numeric]
            Optional number specifying sigma of guassian filter
  
    
    -------------------------------------------------------------------------------------
    Returns:
        map_i:: [holoviews.Image]
            Heatmap image
    
    -------------------------------------------------------------------------------------
    Notes:
        stretch only affects display

    """    
    heatmap = np.zeros(video_dict['reference'][animal_num].shape)
    for frame in range(len(location)):
        Y,X = int(location['Y'  + str(animal_num)][frame]), int(location['X' + str(animal_num)][frame])
        heatmap[Y,X]+=1
    
    sigma = np.mean(heatmap.shape)*.05 if sigma == None else sigma
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma)
    heatmap = (heatmap / heatmap.max())*255
    
    map_i = hv.Image((np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), heatmap))
    map_i.opts( #width=int(heatmap.shape[1]*video_dict['stretch']['width']),
#           height=int(heatmap.shape[0]*video_dict['stretch']['height']),
           invert_yaxis=True, cmap='jet', alpha=1,
           data_aspect=1, # aspect=1,
           colorbar=False, toolbar='below', title="Heatmap")
    
    return map_i





########################################################################################    

def DistanceTool(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for measuring length between two points, in pixel units, in 
    order to ease process of converting pixel distance measurements to some other scale.
    Use point drawing tool to calculate distance beteen any two popints.
    
    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
    
    -------------------------------------------------------------------------------------
    Returns:
        image * points * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define 2 points, the distance 
            between which will be measured and displayed.
        
        distance:: [dict]
            Dictionary with the following keys:
                'd' : Euclidean distance between two reference points, in pixel units, 
                      rounded to thousandth. Returns None if no less than 2 points have 
                      been selected.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(video_dict['reference'].shape[1]), 
        np.arange(video_dict['reference'].shape[0]), 
        video_dict['reference']))
    image.opts(width=int(video_dict['reference'].shape[1]*video_dict['stretch']['width']),
               height=int(video_dict['reference'].shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Select Points")

    #Create Point instance on which to draw and connect via stream to pointDraw drawing tool 
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def markers(data, distance):
        try:
            x_ls, y_ls = data['x'], data['y']
        except TypeError:
            x_ls, y_ls = [], []
        
        x_ctr, y_ctr = np.mean(x_ls), np.mean(y_ls)
        if len(x_ls) > 1:
            x_dist = (x_ls[0] - x_ls[1])
            y_dist = (y_ls[0] - y_ls[1])
            distance['px_distance'] = np.around( (x_dist**2 + y_dist**2)**(1/2), 3)
            text = "{dist} px".format(dist=distance['px_distance'])
        return hv.Labels((x_ctr, y_ctr, text if len(x_ls) > 1 else "")).opts(
            text_color='blue',text_font_size='14pt')
    
    distance = dict(px_distance=None)
    markers_ptl = fct.partial(markers, distance=distance)
    dmap = hv.DynamicMap(markers_ptl, streams=[pointDraw_stream])
    return (image * points * dmap), distance


########################################################################################

def setScale(distance, scale, scale_dict):

    """ 
    -------------------------------------------------------------------------------------
    
    Updates dictionary with scale information, given the true distance between points 
    (e.g. 100), and the scale unit (e.g. 'cm')
    
    -------------------------------------------------------------------------------------
    Args:
    
        distance :: [numeric]
            The real-world distance between the selected points
        
        scale :: [string]
            The scale used for defining the real world distance.  Can be any string
            (e.g. 'cm', 'in', 'inch', 'stone')

        scale_dict :: [dict]
            Dictionary with the following keys:
                'px_distance' : distance between reference points, in pixels [numeric]
                'true_distance' : distance between reference points, in desired scale 
                                   (e.g. cm) [numeric]
                'true_scale' : string containing name of scale (e.g. 'cm') [str]
                'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]

    -------------------------------------------------------------------------------------
    Returns:
        scale_dict :: [dict]
                Dictionary with the following keys:
                    'px_distance' : distance between reference points, in pixels [numeric]
                    'true_distance' : distance between reference points, in desired scale 
                                       (e.g. cm) [numeric]
                    'true_scale' : string containing name of scale (e.g. 'cm') [str]
                    'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
    -------------------------------------------------------------------------------------
    Notes:

    """

    scale_dict['true_distance'] = distance
    scale_dict['true_scale'] = scale
    return scale_dict
    


########################################################################################    

def ScaleDistance(video_dict, df=None, column=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Adds column to dataframe by multiplying existing column by scaling factor to change
    scale. Used in order to convert distance from pixel scale to desired real world 
    distance scale.
    
    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        df:: [pandas.dataframe]
            Pandas dataframe with column to be scaled.
        
        column:: [str]
            Name of column in df to be scaled
        
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with column of scaled distance values.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    if 'scale' not in video_dict.keys():
        return df

    if video_dict['scale']['px_distance']!= None:
        video_dict['scale']['factor'] = video_dict['scale']['true_distance']/video_dict['scale']['px_distance']
        new_column = "_".join(['Distance', video_dict['scale']['true_scale']])
        df[new_column] = df[column]*video_dict['scale']['factor']
        order = [col for col in df if col not in [column,new_column]]
        order = order + [column,new_column]
        df = df[order]
    else:
        print('Distance between reference points undefined. Cannot scale column: {c}.\
        Returning original dataframe'.format(c=column))
    return df



########################################################################################    
    
def Mask_select(video_dict, fstfile=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            First frame of video that can be drawn upon to define regions of interest.
            
        mask:: [dict]
            Dictionary with the following keys:
                'mask' : boolean numpy array identifying regions to exlude
                         from analysis.  If no such regions, equal to
                         None. [bool numpy array)   
                'mask_stream' : Holoviews stream object enabling dynamic selection 
                        in response to selection tool. `mask_stream.data` contains 
                        x and y coordinates of region vertices. [holoviews polystream]
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #Load first file if batch processing
    if fstfile:
        video_dict['file'] = video_dict['FileNames'][0] 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
        if os.path.isfile(video_dict['fpath']):
            print('file: {file}'.format(file=video_dict['fpath']))
            cap = cv2.VideoCapture(video_dict['fpath'])
        else:
            raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
                file=video_dict['fpath']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        video_dict['f0'] = frame
    
    #Make first image the base image on which to draw
    f0 = cropframe(
        video_dict['f0'],
        video_dict.get('crop')
    )
    image = hv.Image((np.arange(f0.shape[1]), np.arange(f0.shape[0]), f0))
    image.opts(width=int(f0.shape[1]*video_dict['stretch']['width']),
               height=int(f0.shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Draw Regions to be Exluded")

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    mask = dict(mask=None)
    poly = hv.Polygons([])
    mask['stream'] = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    #poly_stream = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def make_mask(data, mask):
        try:
            x_ls, y_ls = data['xs'], data['ys'] 
        except TypeError:
            x_ls, y_ls = [], []
        
        if len(x_ls)>0:
            mask['mask'] = np.zeros(f0.shape) 
            for submask in range(len(x_ls)):
                x = np.array(mask['stream'].data['xs'][submask]) #x coordinates
                y = np.array(mask['stream'].data['ys'][submask]) #y coordinates
                xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
                cv2.fillPoly(mask['mask'], pts =[xy], color=1) #fill polygon  
            mask['mask'] = mask['mask'].astype('bool')
        return hv.Labels((0,0,""))
    
    
    make_mask_ptl = fct.partial(make_mask, mask=mask)        
    dmap = hv.DynamicMap(make_mask_ptl, streams=[mask['stream']])
    return image*poly*dmap, mask



def check_p_frames(cap, p_prop_allowed=.01, frames_checked=300):
    """ 
    -------------------------------------------------------------------------------------
    
    Checks whether video contains substantial portion of p/blank frames
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.videocapture]
            OpenCV video capture object.
        p_prop_allowed:: [numeric]
            Proportion of putative p-frames permitted.  Alternatively, proportion of 
            frames permitted to return False when grabbed.
        frames_checked:: [numeric]
            Number of frames to scan for p/blank frames.  If video is shorter
            than number of frames specified, will use number of frames in video.
    
    -------------------------------------------------------------------------------------
    Returns:
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    p_allowed = int(frames_checked*p_prop_allowed)
    
    p_frms = 0
    for i in range(frames_checked):
        ret, frame = cap.read()
        p_frms = p_frms+1 if ret==False else p_frms
    if p_frms>p_allowed:
        raise RuntimeError(
            'Video compression method not supported. ' + \
            'Approximately {p}% frames are p frames or blank. '.format(
                p=(p_frms/frames_checked)*100) + \
            'Consider video conversion.')


########################################################################################        
#Code to export svg
#conda install -c conda-forge selenium phantomjs

#import os
#from bokeh import models
#from bokeh.io import export_svgs

#bokeh_obj = hv.renderer('bokeh').get_plot(image).state
#bokeh_obj.output_backend = 'svg'
#export_svgs(bokeh_obj, dpath + '/' + 'Calibration_Frame.svg')

    