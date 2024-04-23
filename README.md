# Analysis Pipeline For High-mag recording.

install bcrypt and nacl
python3 -m pip install --upgrade bcrypt --user
python3 -m pip install --upgrade py-bcrypt --user
python3 -m pip install --upgrade pynacl --user

This folder contains script for volume stabilization, segementation and straightening of highMag neural signal.

0. Do the beads alignment and time alighment using old pipeline.

1. open a command line on tigressdata(call it window1).
a. module load anaconda
b. cd /projects/LEIFER/communalCode/HighMagNeuron/PythonSubmitDella
c. python submit_segmentation.py
d. select folder and run

This takes usually 2-3 hours and does volume stabilization and segmentation. Check folder_name/output_segment, and see if there are errors(not necessary).

Generates:
output_segment
aligned_volume
flow_folder
neuron_pt
startWorkspace.mat

2. After above job finish(also wait for email), again on tigressdata, open a new command line(window2):
a. cd /projects/LEIFER/communalCode/HighMagNeuron/src
b. source ../venv/bin/activate
c. python Himag_cline.py --folder folder_name(same folder as in 1)
     
     It runs through every volumes first(about 10 min), then you are asked whether you want to annotate more volumes(I only do more volumes for moving-immobile recording when squeezing happens or you can press 0 here and do it in step d), press 0 if no obvious deformation of head happens. Next a image of worm pops up. You need to annotate centerline by hand(from anterior to posterior). left click to add points and right click to delete points. You can press j and k to navigate through z. Close the window after you click points. Then a same image pops up and this time you are asked to plot a polygon of the area containing neurons(only include all head neurons including those in ventral cord). Bright points outside the polygon will not be segemented. Before you close this window, pay attention to the ventral cord of the worm and remember it is on the left hand side or right hand side(when face the same direction as the worm's head, or if you rotate the worm head pointing to left, then if ventral cord is on top, it is right. If ventral cord is on the bottom, it is left.). Close the window and answer the question in command line which side the ventral cord is. 

d. (optional) if there is big deformation of head(for example, in moving-immobile recording), we need to add extra template where deformation happens. go to                       data_folder/neuron_cline_result/neuron_img 
and find the volume index where the deformation happens. use python Himag_cline.py --folder folder_name --add_temp            to add a template of deformed head and volumes after it will use new template.

Generates:
neuron_cline_init.pt
neuron_cline_result (partially empty)
CLstraight folder (empty)

3. Back to the command line in step 1(window1), (already in PythonSubmitDella folder and load python2)
a. python submit_HimagCline.py
b. select folder and run

Generates:
neuron_cline_result
CLstraight folder
output_cline
PointsStats.mat

if the worm is immobilized, unclick Automatic Flip      ########### This can be more stable.
This takes about 2 hours, Check folder_name/output_cline/, and see if there are errors(not necessary). And check folder_name/neuron_cline_result/cline_img to quality check the centerline(blue line with + at head). Sometimes the centerline is flipped. In window2(python3 venv), use command e.g. python Himag_cline.py --folder /projects/LEIFER/PanNeuronal/20210729/BrainScanner20210729_153442 --start_idx <example_ind> --end_idx <example_ind> --method 2 --date_mode last --flip_mode human    This flips frame [43-100] inclusive back . And after you correct for flipping, remember to run combinePointStatsFiles_v2(folder_name) in matlab(script is in 3dbrain folder) for updating the pointStats.mat

4. In window1, and
a. cd  /projects/LEIFER/communalCode/3dbrain/PythonSubmissionScripts
b.run python submitWormAnalysisPipelineFull.py  (unclick the first option:straighten)

Generates:
trackMatrix
pointsStats2.mat

# Matching whole-brain imaging recording with multicolor recording.

STEP 1:

Label multicolor recording with a multicolor gui(recommend https://github.com/leiferlab/multicolor/tree/master/scripts/label_neurons.py)

STEP 2:

a. cd /projects/LEIFER/communalCode/HighMagNeuron/src
b. source ../venv/bin/activate
c. python label_wholebrain2mcw.py --folder whole-brain_foldername
press h in the gui for help. Choose a volume and rotate it to compare with the RFP channel in multicolor recording(single channel window in label_neurons.py).


# Example of tracking with fDNC

fDNC_tracking.py 










