#!/usr/bin/python
# 
#code for submitting jobs onto Princeton Slurm cluster, mainly using Della now.
#Running this opens a GUI for selecting input files and parameters for analyzing
#imaging data after centerline and straightening steps. Neuron tracking algorithm
#is outlined in Nguyen et all (2016). This is the Centerline portion of the code. 
#The folder requires an alignment. mat file
#
# Requirements:
#            CLworkspace.mat file must be in the LowMag folder. This file is made locally by initializeCLworkspace.m 
#            alignments.mat file must be in the BrainScanner folder, this file is made by alignmnets_gui.m 


import slurmInput as slurm
import socket
import guiHelper as gu
import os


def make_gui():
    # load pickle file and get default values
    prevUser=gu.pickle_load()
    defaultName = prevUser['username']
    defaultDate = prevUser['date']
    defaultFolder = prevUser['folderName']
    
    #set up layout, up to 12 rows, 2 columns
    master = gu.submitTK(rows=12,cols=2)
    #variables will be stored in a dict in master.e
    master.addGuiField("User Name",'user_name',defaultName)
    master.addGuiField("Parent Path",'parent_path','/tigress/LEIFER/PanNeuronal')
    master.addGuiField("Date of data",'date',defaultDate)
    master.addGuiField("DataFolderName",'folder_name',defaultFolder)

    # add check box inputs
    master.addGuiCheck("Automatic Flip", 'flip_flag')

    master.addGuiButton("Enter",b_command=lambda:callback1(master=master))
    if  socket.gethostname()=='tigressdata.princeton.edu' or socket.gethostname()=='tigressdata2.princeton.edu':
        master.addGuiButton("Select Folder",b_command=lambda:gu.selectFolder(master=master))
    return master
    

def submitScript(master=None):
    # runs on Enter button press, parses data from gui and submits the job
    username = master.e['user_name'].get()
    beginOfPath=master.e['parent_path'].get()
    date=master.e['date'].get()
    folderName=master.e['folder_name'].get()

    flipFlag = master.e['flip_flag'].var.get()
    if flipFlag:
        flip_mode = 'auto'
    else:
        flip_mode = 'no'
    # which folder to process, must add paths linux style
    fullPath = beginOfPath + "/" + date
    fullPath = fullPath + "/" + folderName

    print("Username: " + username)
    print("full path: " + fullPath)
    # deal with folder names that have spaces
    
    #connect with paramiko to the server, use a password if it was needed
    if 'password' in master.e.keys():
        password =master.e['password'].get
        client=gu.dellaConnect(username,password)
    else:
        client=gu.dellaConnect(username)

    #construct command list for entering into della. 

    #save defaults using pickle dump
    commandList = ["pwd"]
    commandList.append("cd /projects/LEIFER/communalCode/HighMagNeuron/src")
    commandList.append("module load anaconda3/2020.11")
    commandList.append("python submit_cline.py --folder " + fullPath + " --flip_mode " + flip_mode)
    commandList.append("pwd")
    master.pickleDump()
    
    #submit the command list and all of the jobs. 
    commands = "\n".join(commandList)
    #print('commands:\n', commands)
    stdin, stdout, stderr = client.exec_command(commands)

    print('stdOutput:')
    returnedOutput = stdout.readlines()
    print(' '.join(returnedOutput))
    print('stdError:')
    print(stderr.readlines())
    print('Done submitting job.\n\n')

    print('''
        Output files will be saved in 
        '''
        + '''
        neuron_cline_result/
        ''')
    # close window at the end
    client.close()
    master.destroy()
        

def callback1(event=None,master=None):
    
    #Check for password and continue to submit job
    print(master.e['user_name'] .get())
    username = master.e['user_name'].get()
    isNeedsPassword=gu.passwordReqCheck(username)
    if isNeedsPassword:
        # use the same window as before, just add an additional password field
        # password
        master.addGuiField('password','password',default='******',show='*')
        master.addGuiButton("Enter",lambda:submitScript(master=master))
    else:
        print("No password needed")
        submitScript(master)
        


if __name__ == '__main__':
# bind enter key and button
    print('''
        This is the submission script for running centerline fitting.
        
        For a quick test, run this code as follows:
        User Name: <your username>
        Parent Path:/tigress/LEIFER/PanNeuronal
        Date of Data: testing_sets
        Data Folder Name: Brain_working_dataset
        Start Workspace: <Check box> 
        Email: <Check box>
        
        ''')
    master=make_gui()
    

    master.e['user_name'].focus_set()
    
    master.run()
