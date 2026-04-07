
print("Importing libraries...")
import os
import sys
print('Importing holoviews...')
import holoviews as hv
from holoviews import streams
import numpy as np
print('Importing LocationTracking_Functions_loop...')
import LocationTracking_Functions_loop as lt
import time
import socket
print('Importing tkinter...')
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from webdriver_manager.firefox import GeckoDriverManager      # Need to use pip install webdriver-manager to get this.
from selenium.webdriver.firefox.service import Service as FirefoxService

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


## Prompt user to choose file from dialog box
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', True)

print('\nPlease select folder from dialog box. Note that it might be behind the Python window, or on another screen.')

relative_paths = ["behavior_videos", "behavior_videos_copy"]

SrcDir = ''
for x in relative_paths:
    tmp = RaidFolder + x
    if Path(tmp).is_dir():
        SrcDir = filedialog.askdirectory(
            parent=root,
            initialdir=tmp)
        break
root.destroy()
if SrcDir == '':
    print('No folder selected')
    sys.exit(1)

DOWNSAMPLE_FACTOR = 0.5


def has_analysis(filepath):
    filepath_without_ext = Path(filepath).with_suffix("")
    out_path_old = Path(f"{filepath_without_ext}_{str(DOWNSAMPLE_FACTOR)}_Location.csv")
    out_path = Path(f"{filepath_without_ext}_Location.csv")
    return out_path_old.is_file() or out_path.is_file()


print(f'You selected folder "{SrcDir}"\nFiles to analyze are:')
files_recursive = list(Path(SrcDir).rglob('*.avi'))

# Remove some files based on heuristics
files_recursive = [x for x in files_recursive if "exclude" not in str(x)]
files_recursive = [x for x in files_recursive if "tracked" not in str(x)]
files_recursive = [x for x in files_recursive if not has_analysis(x)]

for idx, f in enumerate(files_recursive):
    print(f"{idx+1}: {f}")

import tkinter as tk
from tkinter import messagebox

# Standard setup to hide the main background window
root = tk.Tk()
root.withdraw()
result = messagebox.askokcancel("", "Please check file list in console, and press OK to continue (note that files already analyzed are excluded from thie list)")
root.destroy()

if not result:
    print("Cancelled.")
    sys.exit(1)


for progress_count, f in enumerate(files_recursive):

    # Remove .avi suffix
    filepath_without_ext = Path(f).with_suffix("")

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
    out_path_old = Path(f"{filepath_without_ext}_{str(video_dict['dsmpl'])}_Location.csv")
    out_path = Path(f"{filepath_without_ext}_Location.csv")

    if has_analysis(f):
        # If previous analysis is present, then skip
        print(f'File {progress_count+1} of {len(files_recursive)}, already have _Location.csv, skipping: {f}')
        continue

    animal_id_suffix = f.name[16:]
    animal_id_suffix = animal_id_suffix.split("_")[1]
    try:
        animal_id_suffix = int(animal_id_suffix)
    except ValueError:
        print(f'\n    Unable to find animal ID for "{animal_id_suffix}", skipping.')
        continue

    #
    #   Passed all preliminary checks. Now starting real video analysis
    #

    print(f'\n**** FILE {progress_count+1} OF {len(files_recursive)}: {f}')

    start_time = time.time()
    img_crp, video_dict = lt.LoadAndCrop(video_dict)

    # Hard-coded crop rectangle corresponding to bottom middle of window
    # left-right edges 38 to 276
    # bottom to top edges 238 to 39
    initial_box = {'x0': [38], 'x1': [276], 'y0': [238], 'y1': [39]}
    box_stream = streams.BoxEdit(data=initial_box)  # source=box, num_objects=video_dict['num_animals'])
    video_dict['crop'] = box_stream
    
    video_dict['reference'] = []
    ref = None

    # Calculate reference image for each animal
    for idx in range(video_dict['num_animals']):
        fpath = lt.GetFileBase(video_dict) + "_" + video_dict['crop_names'][idx] + "_reference.png"

        ref, img_ref = lt.Reference(video_dict, num_frames=50, frames=None, crop_num=idx)
        if ref is None:
            break
        video_dict['reference'].append(ref)
        print(f'Saving ref image to     : {fpath}')
        hv.save(img_ref, fpath)

    if ref is None:
        print('    Unable to grab enough frames to compute reference image, skipping.')
        continue

    tracking_params = {
        'loc_thresh'    : 95,   # Default percentile, can be overridden later if needed (but usually isn't)
        'use_window'    : False,
        'window_size'   : 150,
        'window_weight' : .9,
        'method'        : 'dark',
        'rmv_wire'      : True,
        'wire_krn'      : 5
    }

    # Generate track examples for checking
    for idx in range(video_dict['num_animals']):
        img_exmpls = lt.LocationThresh_View(video_dict, tracking_params, examples=6, crop_num=idx)
        img_exmpls.cols(6)

        fpath = lt.GetFileBase(video_dict) + "_" + video_dict['crop_names'][idx] + "_track_examples.png"
        print(f'Saving track examples to: {fpath}')
        hv.save(img_exmpls, fpath)

    # Track animal
    location = lt.TrackLocation(video_dict, tracking_params)
    out_path = os.path.splitext(video_dict['fpath'])[0] + '_Location.csv'

    # Print stats
    Dist = location['Dist_px0']
    print(f"Mean distance per frame is: {Dist.mean():0.5f}, min is {Dist.min():0.5f}, max is {Dist.max():0.5f}")

    print(f"Saving tracking data to file   : {out_path}")
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
            fig_size=300, aspect=1.2,  # For matplotlib.
            color='red', title=f"Distance Across Session, mean={Dist.mean():0.3f}, SD={np.std(Dist):0.3f}")
        plt_trks = lt.showtrace(video_dict, location, color="red", alpha=.05, size=2)
        plt_hmap = lt.Heatmap(video_dict, location, sigma=None)
        p = (plt_trks + plt_hmap + plt_dist).cols(3)

        fpath = lt.GetFileBase(video_dict) + "_" + video_dict['crop_names'][x] + "_movement.png"
        print(f'Saving to movement summary file: {fpath}')
        hv.save(p, fpath)

    display_dict = {
        'start'      : 0,       # If < video_dict['start'], will be coerced to that value
        'stop'       : None,    # If > video_dict['end'], will be coerced to that value
        'resize'     : None,
        'save_video' : True
    }

    # Generate tracked vide
    lt.PlayVideo(video_dict, display_dict, location)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds\n")

