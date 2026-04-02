
print("Importing libraries...")
import os
import sys
import holoviews as hv
import numpy as np
import LocationTracking_Functions_loop as lt
from bokeh.io import output_notebook, show
import inspect
import time
import importlib
import socket
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import selenium  # This is needed by bokeh.io image export functions. Make sure you have latest version by running pip install -U selenium
from webdriver_manager.firefox import GeckoDriverManager      # Need to use pip install webdriver-manager to get this.
from selenium import webdriver

# Get the hostname of the local machine
hostname = socket.gethostname()
print(f"The machine's hostname is: {hostname}")

if hostname == "TJHomeOffice":
    TeamsFolder = "D:\\University of Maryland School of Medicine\\JhouLab Overflow1 - General\\"
    RaidFolder = r"\\TOWER\share_zfs48_home/"
elif hostname == "TomOffice2025":
    TeamsFolder = "D:/University of Maryland School of Medicine/JhouLab Overflow1 - General/"
    RaidFolder = r"\\LABUNRAID\zfs48_share/"
elif hostname == "TJ_gram":
    TeamsFolder = "C:/Users/tomjh/University of Maryland School of Medicine/JhouLab Overflow1 - General/"
    RaidFolder = r"\\LABUNRAID\zfs48_share/"
elif hostname == "DESKTOP-DIQ9828":
    RaidFolder = r"\\LABUNRAID\zfs48_share/"
else:
    print("Unrecognized machine, please choose top level video folder from dialog.")
    RaidFolder = r"\\LABUNRAID\zfs48_share/"

# Set gecko_driver environment variables
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager      # Need to use pip install webdriver-manager to get this
from pathlib import Path

# Automatically manage the geckodriver executable
service = FirefoxService(GeckoDriverManager().install())
gecko_driver_path = service.path

def find_firefox_windows():
    # Common default installation paths for Firefox on Windows
    # 64-bit Firefox on 64-bit Windows or 32-bit Firefox on 32-bit Windows
    paths = [
        Path("C:\\Program Files\\Mozilla Firefox\\firefox.exe"),
        # 32-bit Firefox on 64-bit Windows
        Path("C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe")
    ]
    # Check for user-only installation via Microsoft Store or specific installer
    appdata_local = os.getenv("LOCALAPPDATA")
    if appdata_local:
        # The exact folder name for the App Store version can vary (e.g., Mozilla.Firefox_xxxxxxxxxxxxx)
        # This is a common location pattern for user-only installs
        paths.append(Path(appdata_local) / "Mozilla Firefox" / "firefox.exe")
    for path in paths:
        if path.is_file():
            print(f'Found Firefox installation at: "{path}"')
            return str(path)
    print('Unable to find firefox installation. Some graph generating code may not work.')
    return None

# Append firefox.exe to path
current_path = os.environ.get('PATH')
firefox_dir = os.path.dirname(find_firefox_windows())  # r"C:\Program Files (x86)\Mozilla Firefox/"   # Use "r" in front to prevent backslashes from being interpreted as escape symbols
if not current_path.endswith(os.pathsep):
    # Append semicolon if not already present. Usually it will already be there, so this is skipped
    current_path = current_path + os.pathsep
gecko_path = os.path.dirname(gecko_driver_path)
print(f'Found gecko installation at: "{gecko_path}"')

# Add firefox and gecko paths to PATH variable
os.environ['PATH'] = f"{current_path}{firefox_dir}{os.pathsep}{gecko_path}"
print("Initialization completed.")


#  %%output size = 200

## Prompt user to choose file from dialog box
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', True)

print('Please select AVI file from dialog box. Note: dialog might be behind the Python window, or on another screen.')

relative_paths = ["behavior_videos", "behavior_videos_copy"]

for x in relative_paths:
    tmp = RaidFolder + x
    if Path(tmp).is_dir():
        SrcDir = filedialog.askdirectory(
            parent=root,
            initialdir = tmp)
        break
root.destroy()
if SrcDir == '':
    print('No folder selected')
    sys.exit(1)
else:
    print(f'You selected folder "{SrcDir}"')


files_recursive = list(Path(SrcDir).rglob('*.avi'))

for f in files_recursive:

    # Remove .avi suffix
    tmp = Path(f).with_suffix("")

    if "tracked" in f.name:
        continue

    video_dict = {
        'dpath'        : f.parent,   # Path to parent folder
        'file'         : f.name,     # File name only, without directories
        'fpath'        : [],         # Full path, including all directories
        'start'        : 0,
        'end'          : None,
        'num_animals'  : 1,        # How many animals are in this frame?
        'crop_names'   : ['animal1', 'animal2', 'animal3', 'animal4'],     # Name of each subject
        'region_names' : ['region1', 'region2', 'region3', 'region4'],
        'dsmpl'        : 0.5,           # Downsample proportion
        'stretch'      : dict(width=1, height=1)
    }

    # Generate output target filename with downsample ratio in filename
    out_path = Path(f"{tmp}_{str(video_dict['dsmpl'])}_Location.csv")

    if out_path.is_file():
        # If previous analysis is present, then skip
        print(f'{f}: Already have analysis file {out_path}. Will skip.')
        # continue

    CROP_NAME = ""

    print(f)

#    continue

    img_crp, video_dict = lt.LoadAndCrop(video_dict, cropmethod='Box')
#    display(img_crp)

    video_dict['reference'] = []

    for idx in range(video_dict['num_animals']):
        fpath = lt.GetFileBase(video_dict) + "_" + video_dict['crop_names'][idx] + "_" + str(video_dict['dsmpl']) + "_reference.png"

        ref, img_ref = lt.Reference(video_dict, num_frames=50, frames=None, crop_num=idx)
        video_dict['reference'].append(ref)
        print(f'Saving to: {fpath}')
        hv.save(img_ref, fpath)

    tracking_params = {
        'loc_thresh'    : 95,   # Default percentile, can be overridden later if needed (but usually isn't)
        'use_window'    : False,
        'window_size'   : 150,
        'window_weight' : .9,
        'method'        : 'dark',
        'rmv_wire'      : True,
        'wire_krn'      : 5
    }

    for idx in range(video_dict['num_animals']):
        img_exmpls = lt.LocationThresh_View(video_dict, tracking_params, examples=6, crop_num=idx)
        img_exmpls.cols(6)

        fpath = lt.GetFileBase(video_dict) + "_" + video_dict['crop_names'][idx] + "_" + str(video_dict['dsmpl']) + "_track_examples.png"
        print(f'Saving to file: {fpath}')
        hv.save(img_exmpls, fpath)
        print('Done saving')


    location = lt.TrackLocation(video_dict, tracking_params)
    out_path = os.path.splitext(video_dict['fpath'])[0] + "_" + str(video_dict['dsmpl']) + '_Location.csv'

    # Print stats
    Dist = location['Dist_px0']
    print(f"Mean distance per frame is: {Dist.mean()}, min is {Dist.min()}, max is {Dist.max()}")

    print(f"Saving to file: {out_path}")
    location.to_csv(out_path, index=False)
    location.head()

    #
    # Show graphics and plots of movement
    #

    w, h = 600,200

    fps = video_dict['nominal_fps']

    for x in range(video_dict['num_animals']):
        Dist = location['Dist_px' + str(x)]
        plt_dist = hv.Curve((location['Frame'] / fps / 60, Dist), 'Time (minutes)', 'Pixel Distance').opts(
#            height=h, width=w,   # Bokeh
            fig_size=300, aspect=1.2,  # For matplotlib.
            color='red', title=f"Distance Across Session, mean={Dist.mean():0.3f}, SD={np.std(Dist):0.3f}")
            #toolbar="below")
        plt_trks = lt.showtrace(video_dict, location, color="red", alpha=.05, size=2)
        plt_hmap = lt.Heatmap(video_dict, location, sigma=None)
        p = (plt_trks + plt_hmap + plt_dist).cols(3)
        # (plt + plt_dist)

        fpath = lt.GetFileBase(video_dict) + "_" + video_dict['crop_names'][x] + "_" + str(video_dict['dsmpl']) + "_movement.png"
        print(f'Saving to file: {fpath}')
        hv.save(p, fpath)

    display_dict = {
        'start'      : 0,   # If < video_dict['start'], will be coerced to that value
        'stop'       : None,    # If > video_dict['end'], will be coerced to that value
        'resize'     : None,
        'save_video' : True
    }

    start_time = time.time()

    lt.PlayVideo(video_dict, display_dict, location)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

