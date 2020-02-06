from dlc_tools import quantifyErrors, getFrameCount, getLabelNames, testThresholds
import numpy as np
import pandas as pd
import os
import math
import time
import sys
from pathlib import Path

#Matplotlib Configuration
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('WxAgg')

import sklearn.metrics as sk

#True Positive, True Negative, False Positive, False Negative
errorTypes = ['TP', 'TN', 'FP', 'FN']
def score(trueLabels, testLabels, filepath, thresholds, func = None):
    thresh = dict(zip(thresholds,range(len(thresholds))))

    results, SnS = quantifyErrors(trueLabels, testLabels, filepath)
    occ, distances, snapshots = testThresholds(trueLabels, testLabels, filepath,
                                              thresholds=thresholds, filter=filter)

    labels = getLabelNames(testLabels)
    scores = pd.DataFrame(index = thresholds, columns = labels )
    numberTestFrames = getFrameCount(filepath)['Test']

    for i, s in enumerate(snapshots):
        S = s['Test'].values
        print(S)
        TP = S[:,0]
        TN = S[:,1]
        FP = S[:,2]
        FN = S[:,3]
        dist = np.asarray(distances['Test'].iloc[i, 2::2], dtype = 'float64')
        std = np.asarray(distances['Test'].iloc[i, 3::2], dtype = 'float64')
        if func == None:
            func = "np.multiply( (3 / dist)**2, (TP+TN)) / numberTestFrames"
        score  = eval(func)

        scores.iloc[i] = np.asarray(score)
    variableThresholds = np.asarray(scores.astype("float").idxmax())
    labelScores = {}
    scoreSum = 0
    i = 0
    for i,t in enumerate(variableThresholds):
        score = scores.loc[t,labels[i]]
        labelScores[labels[i]] = {"Threshold":np.round(t, 2), "Score":score}
        scoreSum += score
    finalScore = scoreSum / len(variableThresholds)
    return scores, labelScores, finalScore
"""
def score(trueLabels, testLabels, filepath, thresholds=[], filter=True):
    occ, distances, snapshots = testThresholds(trueLabels, testLabels, filepath,
                                              thresholds=thresholds, filter=filter)

    label = getLabelNames(testLabels)
    scores = pd.DataFrame(index = thresholds, columns = label )
    numTestFrames = 0

    err1 = []

    numTestFrames = getFrameCount(filepath)['Test']

    for i,s in enumerate(snapshots):
        numOcc = s['Test'].values
        errSum = numOcc[:,1] # - numOcc[:,2] - numOcc[:,2]
        err1.append(errSum)
        tp = numOcc[:,0]
        dists = np.asarray(distances['Test'].iloc[i, 2::2], dtype = 'float64')
        std = np.asarray(distances['Test'].iloc[i, 3::2], dtype = 'float64')

        weights = (3 / dists)**2

        score = np.multiply(weights, (tp+errSum))/numTestFrames
        scores.iloc[i] = np.asarray(score)

    variableThresholds = np.asarray(scores.astype("float").idxmax())
    labelScores = {}
    scoreSum = 0
    for i,t in enumerate(variableThresholds):
        score = scores.loc[t,label[i]]
        labelScores[label[i]] = {"Threshold":np.round(t, 2), "Score":score}
        scoreSum += score
    finalScore = scoreSum / len(variableThresholds)
    return scores, labelScores, finalScore
"""
