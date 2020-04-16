
import dlc_tools as dt
import yaml
import os
import sys
import cv2
import time
import pandas as pd
import numpy as np
import os.path
import argparse
import tensorflow as tf

from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
from pathlib import Path
from tqdm import tqdm
from deeplabcut.utils import auxiliaryfunctions
from skimage.util import img_as_ubyte
from threading import Thread
from queue import Queue
from collections import OrderedDict as ordDict


class StreamHandler:
    def __init__(self, camIdx=0, dropFrames = False):

        #Storage structure for frames, Thinking can sequentially add frames for multiple cameras
        self.queue = Queue()
        self.stopped = False
        self.curFrame = None #creates pointer to current frame
        self.dropFrames = dropFrames
        self.meta = ordDict({
            'Frame Idx':[],
            'Frame Read':[],
            'Time Retrieved':[],
            'Time Read':[],
            'Time Processed':[],
            'Time Displayed':[],
            'Processing Time':[],
            'Displaying Time':[],
            'Total Time':[]
        })
        self.frames = []
        self.frameCounter = 0
        self.curIdx = 0

        #Will prevent file from being passed in
        if isinstance(camIdx, int):
            self.stream = cv2.VideoCapture(camIdx)#Initializing camera stream
            (self.grabbed, self.frame) = self.stream.read()
            self.curFrame = self.frame
            self.startTime = time.time() #Reference time to calculate elapsed time
            self.updateMeta()

        elif isinstance(camIdx, []):
            print("Multiple Camera handling to be implemented at a later date")
        else:
            raise Exception('Invalid input for camIdx')

    #starts thread to read frames
    def startThread(self):
        self.t = Thread(target=self.update)
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while True:            #If thread is stopped, the loop will stop
            #grabs next frame
            self.frameCounter += 1
            (self.grabbed, self.frame) = self.stream.read()

            #handling meta data
            self.updateMeta()

            #Puts data in structures
            self.queue.put_nowait(self.frame)
            if (self.dropFrames):
                self.curIdx += 1
                self.curFrame = self.frame

            if self.stopped:
                print("Stopping")
                self.t.join() #I think
                break


    def read(self):
        #only returns once a frame is available and returns a single frame
        while True:
            if self.dropFrames:
                if len(self.curFrame) > 0:
                    frame = self.curFrame
                    self.curFrame = []
                    self.meta['Frame Read'][self.curIdx] = True #might need to move meta update into main while loop
                    return frame

            elif not self.queue.empty():
                self.meta['Frame Read'][self.curIdx] = True
                self.curIdx += 1
                return self.queue.get_nowait()


    def stop(self):
        self.stopped = True

    def beginCapture(self, maxFrames = None):
        sess, inputs, outputs, dlc_cfg = self.initializeDLC()
        self.startThread()

        loopcount = 0
        timeArr = np.zeros((100))
        self.fpsArr = []
        while True:
            if isinstance(maxFrames, int):
                if self.frameCounter > maxFrames:
                    break

            timeIdx = loopcount % 100

#            if loopcount%100 == 0:
#                start = time.time()
#                loopcount = 0

            if cv2.waitKey(1) == 27:
                break  # esc to quit

            curIdx = self.curIdx #ensures all idx are consistent for current loop iterations
            frame = self.read()
            self.meta['Time Read'][curIdx] = time.time() - self.startTime

            pose = self.analyzeFrame(frame, sess, inputs, outputs, dlc_cfg)
            procTime =  time.time() - self.startTime
            self.meta['Time Processed'][curIdx] = procTime

            timeArr[timeIdx] = procTime
            elapsedTime = timeArr[timeIdx] - timeArr[(timeIdx+1)%100] #%100 should account for edge case of 99
            fps = 100 / elapsedTime
            self.fpsArr.append(fps)

            labeledFrame = self.labelFrame(frame, pose)

            end = time.time()
#            cv2.putText(labeledFrame, ("FPS: " + str(loopcount / (end-start))), (0,30), cv2.FONT_HERSHEY_SIMPLEX, .69, (0,0,0), thickness = 2, lineType=cv2.LINE_AA)

            cv2.imshow('my webcam', labeledFrame)
            self.meta['Time Displayed'][curIdx] = time.time() - self.startTime

#            sys.stdout.write('\r')
#            sys.stdout.write(str(loopcount / (end-start)))
#            sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.write(str(fps))
            sys.stdout.flush()
            loopcount  += 1

        cv2.destroyAllWindows()
        self.stopped = True



    def updateMeta(self):
        self.meta['Frame Idx'].append(self.frameCounter)
        self.meta['Frame Read'].append(False)
        self.meta['Time Retrieved'].append(time.time() - self.startTime)
        self.meta['Time Read'].append(np.nan)
        self.meta['Time Processed'].append(np.nan)
        self.meta['Time Displayed'].append(np.nan)
        self.meta['Processing Time'].append(np.nan)
        self.meta['Displaying Time'].append(np.nan)
        self.meta['Total Time'].append(np.nan)
        self.frames.append(self.frame)

    def getMetaData(self):
        timeproc = np.array(self.meta['Time Processed'])
        timedisp = np.array(self.meta['Time Displayed'])
        timeread = np.array(self.meta['Time Read'])
        self.meta['Processing Time'] = timeproc - timeread
        self.meta['Displaying Time'] = timedisp - timeproc
        self.meta['Total Time'] = timedisp - timeread
        return pd.DataFrame.from_dict(self.meta)

    def analyzeFrame(self, frame, sess, inputs, outputs, dlc_cfg):
        frame = img_as_ubyte(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
        return pose

    def labelFrame(self, frame, pose, threshold = .1):
        poseBool = pose[:, 2] > threshold
        labelCoords = pose[poseBool, 0:2]
        curBodyparts = self.bodyparts[poseBool]
        for i, bodypart in enumerate(curBodyparts):
            #cv2.circle(frame, (int(points[i,0]), int(points[i,1])), 2, color = (255,255,255), thickness =2)
             cv2.putText(frame, bodypart, (int(labelCoords[i,0]), int(labelCoords[i,1])), cv2.FONT_HERSHEY_SIMPLEX, .69, (255,255,255), thickness = 2, lineType=cv2.LINE_AA)
        return frame




####################################################################################################
    def initializeDLC(self, shuffle=0, config = None):
        videotype='avi';
        shuffle=0
        trainingsetindex=0;
        gputouse=0;
        save_as_csv=False;
        destfolder=None;
        batchsize=1;
        crop=None;
        TFGPUinference=True;
        dynamic=(False, .5, 10)

        #Temporary hardcoded file path
        if config == None:
            config = '/home/nickt/DeepLabCut/Trial 8-Nick_T-2020-04-08/config.yaml'

        #########################################################################################
        #Taken from Deeplabcut

        if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
            del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

        tf.reset_default_graph()
        start_path=os.getcwd() #record cwd to return to this directory in the end
        cfg = auxiliaryfunctions.read_config(config)
        trainFraction = cfg['TrainingFraction'][trainingsetindex]

        modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
        path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
        try:
            dlc_cfg = load_config(str(path_test_config))
        except FileNotFoundError:
            raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

        # Check which snapshots are available and sort them by # iterations
        try:
          Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
        except FileNotFoundError:
          raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

        if cfg['snapshotindex'] == 'all':
            print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex=cfg['snapshotindex']

        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

        ##################################################
        # Load and setup CNN part detector
        ##################################################

        # Check if data already was generated:
        dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
        trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
        # Update number of output and batchsize
        dlc_cfg['num_outputs'] = cfg.get('num_outputs', dlc_cfg.get('num_outputs', 1))
        batchsize = 1
        if dynamic[0]: #state=true
            #(state,detectiontreshold,margin)=dynamic
            print("Starting analysis in dynamic cropping mode with parameters:", dynamic)
            dlc_cfg['num_outputs']=1
            TFGPUinference=False
            dlc_cfg['batch_size']=1
            print("Switching batchsize to 1, num_outputs (per animal) to 1 and TFGPUinference to False (all these features are not supported in this mode).")

        # Name for scorer:
        if dlc_cfg['num_outputs']>1:
            if  TFGPUinference:
                print("Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently.")
                TFGPUinference=False
            print("Extracting ", dlc_cfg['num_outputs'], "instances per bodypart")
            xyz_labs_orig = ['x', 'y', 'likelihood']
            suffix = [str(s+1) for s in range(dlc_cfg['num_outputs'])]
            suffix[0] = '' # first one has empty suffix for backwards compatibility
            xyz_labs = [x+s for s in suffix for x in xyz_labs_orig]
        else:
            xyz_labs = ['x', 'y', 'likelihood']

        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
        DLCscorer = 'nickt'
        pdindex = pd.MultiIndex.from_product([dlc_cfg['all_joints_names'],
                                              xyz_labs],
                                             names=['bodyparts', 'coords'])
        #####################################################################################
        tmpcfg = yaml.load(config)
        stream = open(config, 'r')
        tmpcfg = yaml.load(stream)
        bodyparts = tmpcfg['bodyparts']
        self.bodyparts = np.array(bodyparts)
        return sess, inputs, outputs, dlc_cfg
        #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
