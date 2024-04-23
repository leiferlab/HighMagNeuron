import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import unmix as um
import mistofrutta as mf
import wormbrain as wormb
import wormneuronsegmentation as wormns

#####################
# PARSE CONFIGURATION
#####################

if "--help" in sys.argv: print(um.label_neurons_instructions); quit()

folder = sys.argv[1]
if folder[-1] != "/": folder += "/"
if not os.path.isdir(folder): print("Invalid folder."); quit()

side_views = "--side-views" in sys.argv
mc1 = "--mc1" in sys.argv
mc2 = "--mc2" in sys.argv
atlas = "--atlas" in sys.argv
save = "--no-save" not in sys.argv
verbose = "-v" in sys.argv or "--verbose" in sys.argv
show_ch = "--no-show-ch" not in sys.argv
crop = "--no-crop" not in sys.argv
from_visualize_light = "--from-visualize-light" in sys.argv
to_visualize_light = "--to-visualize-light" in sys.argv
ignore_cache = "--ignore-cache" in sys.argv
if ignore_cache:
    confirm = input("Are you sure you want to ignore the cache? " + \
                    "This will overwrite the saved brain and labels. (y/n):")
    if confirm[0] != "y": quit()
gamma = [1]
dil_size = 7
resize_ratio = 1 if not mc2 else 2
segm_threshold = 0.02
pixel_offset = np.array([430, 430, 430, 430])
shift_blue = None if not mc2 else -2
scale = 1.0
progressive_gamma = 0.04
comment_fields = "Color-Differences-Location" if atlas else ""
for s in sys.argv:
    sa = s.split(":")
    if "--gamma" in sa:
        gamma = [float(num) for num in sa[1].split("-")]
    elif sa[0] == "--progressive-gamma":
        progressive_gamma = float(sa[1])
    elif sa[0] == "--dil-size":
        dil_size = int(sa[1])
    elif sa[0] == "--resize-ratio":
        resize_ratio = int(sa[1])
    elif sa[0] == "--segm-threshold":
        segm_threshold = float(sa[1])
    elif sa[0] == "--pixel-offset":
        pixel_offset = [int(a) for a in sa[1].split("-")]
        if len(pixel_offset) == 1:
            pixel_offset = [pixel_offset[0] for i in range(4)]
        elif len(pixel_offset) == 3:
            pixel_offset.insert(2, 0)
        pixel_offset = np.array(pixel_offset)
    elif sa[0] == "--shift-blue":
        shift_blue = int(sa[1])
    elif sa[0] == "--scale":
        scale = int(sa[1])
    elif sa[0] == "--comment-fields":
        comment_fields = sa[1]

# Simple log of the command used
logbook_f = open(folder + "analysis.log", "a")
now = datetime.now()
dt = now.strftime("%Y-%m-%d %H:%M:%S:\t")
logbook_f.write(dt + " ".join(sys.argv) + "\n")
logbook_f.close()

#####################################################
# LOAD IMAGE, PREPROCESS IT, AND FIND OR LOAD NEURONS
#####################################################

if not from_visualize_light:
    ##################
    # STANDARD LOADING
    ##################

    # Load the multicolor image
    im, n_z, n_ch = um.load_spinningdisk_uint16(folder, return_all=True)
    if shift_blue is not None:
        im_tmp = im.copy()
        for iz in np.arange(n_z): im[iz, 0] = im_tmp[(iz - shift_blue) % n_z, 0]

    # If mc1 remove the GFP channel
    if mc1:
        im = im[:, [0, 2, 3, 4], ...].copy()
        n_ch -= 1

    z_of_frame = um.load_spinningdisk_z(folder)

    # THIS NEEDS TO BE FINISHED
    # rot_im = mf.geometry.rotations.rotate_hyperstack(im)
    # mf.plt.hyperstack2(rot_im)
    # quit()

    # Resize the image
    if resize_ratio != 1:
        r_im = np.zeros((im.shape[0], im.shape[1], im.shape[-2] // resize_ratio,
                         im.shape[-1] // resize_ratio), dtype=np.uint16)
        for rr1 in np.arange(resize_ratio):
            for rr2 in np.arange(resize_ratio):
                r_im += im[:, :, rr1::resize_ratio, rr2::resize_ratio]
        im = r_im
    else:
        pass

    # Brains object: Load it if it exists, otherwise find the neurons and
    # create it.
    if os.path.isfile(folder + "brains.json") and not ignore_cache:
        cervelli = wormb.Brains.from_file(folder)
        manual_labels = [cervelli.get_labels(0)]
    else:
        manual_labels = None
        neuron_yx, neuron_props = wormns.findNeurons(
            im[:, 2].copy(), 0, 1, 1, np.array([0, n_z]),
            threshold=segm_threshold, checkPlanesN=7,
            dil_size=dil_size)
        cervelli = wormb.Brains.from_find_neurons(
            neuron_yx, np.array([0, n_z]),
            [z_of_frame],
            properties=neuron_props,
            stabilize_z=False, stabilize_xy=False)

else:

    ##############################################################
    # LOAD DATA PREPROCESSED WITH MATT'S AND PANINSKI LAB SOFTWARE
    ##############################################################
    # Load image data
    mc_adj = um.load_multicolor_adjustment(folder)

    # Adjust axes order
    im = np.swapaxes(mc_adj["data"], 0, 2)
    im = np.swapaxes(im, 1, 3)

    # Remove GFP
    im = im[:, [0, 2, 3, 4], ...].copy()
    n_z = im.shape[0]

    # Force some parameter values in this case
    pixel_offset = np.zeros(4)
    gamma = np.ones(3)
    gamma[0] = mc_adj["gamma_val"][0]
    gamma[1] = mc_adj["gamma_val"][2]
    gamma[2] = mc_adj["gamma_val"][4]
    crop = False
    progressive_gamma = False

    # Load Brains object either directly from the visualize_light weird
    # format, or from the "cached" json version.
    if not os.path.isfile(folder + "brains-from_visualize_light.json"):
        cervelli = wormb.Brains.from_visualize_light(folder)
    else:
        cervelli = wormb.Brains.from_file(folder,
                                          "brains-from_visualize_light.json")

    manual_labels = [cervelli.get_labels(0)]

# Number of neurons in this brain
n_neurons = cervelli.nInVolume[0]

# Obtain the overlay for the hyperstack GUI from the Brains object
ovrl, ovrl_labs = cervelli.get_overlay2(vol=0, return_labels=True,
                                        label_size=15, scale=scale)

# If verbose, print the number of labeled neurons and the current labels
if verbose:
    print(np.where(np.array(manual_labels[0]) != "")[0].shape, "labeled neurons")
    print(manual_labels)

# Crop the image
if crop:
    im, r_c = mf.geometry.draw.crop_image(im, folder, return_all=True, scale=scale)
    ovrl[:, 1] -= r_c[0, 0]
    ovrl[:, 2] -= r_c[0, 1]

# Make the RGB image from the multicolor hyperstack
im_rgb = um.im_to_rgb(
    np.clip(im.astype(float) - pixel_offset[None, :, None, None], 0, None),
    gamma=gamma)

# Apply the progressive gamma correction in the first planes of the far half of
# the image
if progressive_gamma is not None:
    for iq in np.arange(n_z):
        if iq < n_z / 2 and iq > n_z * 0.3:
            im_rgb[iq] = np.power(im_rgb[iq], 1. - progressive_gamma * (n_z / 2 - iq))

# Create an instance of the Labels object, which manages the external callbacks
# set in the hyperstack GUI
labels_obj = um.Labels(n_neurons, cervelli, comment_fields=comment_fields)
callbacks = [labels_obj.set_from_action]

#################
# HYPERSTACK GUIs
#################

# If requested, create the hyperstack GUI with the individual channels. For this
# hyperstack GUI and the main RGB hyperstack GUI to be sychronized, set them in
# live mode. Also add the update() method of this hypestack to the callbacks
# that are going to be set in the RGB hyperstack.
if show_ch:
    ipp_r = mf.plt.hyperstack2(im, overlay=ovrl, overlay_labels=ovrl_labs,
                               manual_labels=manual_labels,
                               side_views=side_views, live=True)
    callbacks.append(ipp_r.update)
    live = True
    ipp_r.fig.canvas.set_window_title("Individual channels")
    ipp_r.channel_titles = ["BFP", "CyOFP", "tagRFP", "mNeptune"]
    if from_visualize_light: ipp_r.z_descr = "R (smaller z) -> L (larger z)"
else:
    live = False

# Create the RGB hyperstack GUI
ipp = mf.plt.hyperstack2(im_rgb, overlay=ovrl, overlay_labels=ovrl_labs, rgb=True,
                         manual_labels=manual_labels, side_views=side_views,
                         ext_event_callback=callbacks, live=live)
if from_visualize_light: ipp.z_descr = "L (smaller z) -> R (larger z)"

ipp.fig.canvas.set_window_title("RGB")

# Cycle for synchronization of the z positions of the two hyperstack GUIS
z_old = ipp.z
if live:
    while True:
        if ipp.z != z_old:
            ipp_r.z = ipp.z
            z_old = ipp.z
        if ipp.has_been_closed: plt.close('all');break
        ipp.update(live=True, refresh_time=0.01)

##################################
# EXTRACT RESULTS FROM HYPERSTACKS
##################################

# Once the GUIs are closed, from the hyperstack GUI get the labels
labels = ipp.get_manual_labels()[0].copy()
# and the added points
_sel_pts = ipp.get_selected_points()
sel_pts = np.zeros((_sel_pts.shape[0], 3), dtype=int)
sel_pts[:, 0] = _sel_pts[:, 0]
sel_pts[:, 1:] = _sel_pts[:, 2:]
labels += ipp.selected_points_labels
# If the image was cropped, translate the added points accordingly
if crop: sel_pts[:, 1] += r_c[0, 0]; sel_pts[:, 2] += r_c[0, 1];

######
# SAVE
######

# Store the labels and the comments in the Brains object and save
if save:
    if scale != 1: sel_pts[-2:] = (sel_pts[-2:] / scale).astype(int)
    cervelli.add_points(sel_pts)
    cervelli.set_labels(0, labels, labels_obj.confidences)
    cervelli.labels_comments[0] = labels_obj.comments

    if not from_visualize_light:
        fname = ""
    else:
        fname = "brains-from_visualize_light.json"
    cervelli.to_file(folder, fname)
    if to_visualize_light: cervelli.to_visualize_light(folder)
