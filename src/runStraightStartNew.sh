#!/bin/sh
#
# File:   runWormCellTracking.sh
# Author: benbratton
#
# Created on May 26, 2015, 11:56:37 AM
# tell the scheduler which version of matlab to use, put it before $PATH so that it takes precedence
export PATH=/usr/local/matlab-R2013a/bin/:$PATH
# This path is for Della
export PATH=/tigress/LICENSED/matlab-R2017a/bin/:$PATH
export CODE_HOME=/tigress/LEIFER/communalCode
# parse matlab paths
FILES=$CODE_HOME/3dbrain/PythonSubmissionScripts/*.path
#echo $FILES
for input in $FILES
do
  #echo "Processing $input file..."  
  while read file
  do
      export MATLABPATH="$MATLABPATH:$CODE_HOME/$file"
  done  < "$input"
done

export MATLABPATH="$MATLABPATH;"

# make sure that the matlab path is full/correct
echo $MATLABPATH
echo $1
# run the job
# if 3rd input is 1, run on with slurm task array ID
# REMOVED SGE capabilities
# if [ "$4" ge "1" ]; then
   matlab -nosplash -nodesktop -nodisplay -singleCompThread -r "clusterStraightenStart_v3('$1');exit;"
   # else
   # matlab -nosplash -nodesktop -nodisplay -singleCompThread -r "clusterWormTracker('$1',$SGE_TASK_ID,$2,$3);exit;"
   # fi
