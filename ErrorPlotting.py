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

def plotDistanceDistribution(trueLabels, testLabels, filepath, threshold = .1, filter=True,
                            type='Test'):

    results, SnS, params = quantifyErrors(trueLabels, testLabels, filepath, getParams=True,
                                            filter = filter)
    sb.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    bins = np.linspace(0,100,201) #.5 width bins

    for i,label in enumerate(params['labels']):
        tempResults = results[results['Type']==type].iloc[:,(i+1)*3 ].values
        x = tempResults[~pd.isnull(tempResults)]

        sb.boxplot(x, ax=ax_box)
        sb.distplot(x, ax=ax_hist, bins=bins)

        ax_box.set(yticks=[])
        sb.despine(ax=ax_hist)
        sb.despine(ax=ax_box, left = True)

        plt.title(label)
        plt.xlim(0,50)
        plt.xlabel('Distance (pixels)')
        plt.waitforbuttonpress()
        ax_box.clear()
        ax_hist.clear()




def violinplot(trueLabels, testLabels, filepath, filter = True, threshold=.1,type='Test' ):
    """
    if filter != "compare" and isinstance(filter, (str)):
        raise Exception('filter is an invalid string. It must be either a boolean or the string'
                        ' "compare". You passed in filter: {}'.format(filter))
    elif not isinstance(filter, boolean):
        raise exception('filter is of the incorrect type. It must be either a boolean or the string'
                        ' "compare". You passed in filter of type {}'.format(type(filter))
    """
    #Compares filter and filtered distance distrutions
    if (filter == 'compare'):
        resultsFilt, SnS, params = quantifyErrors(trueLabels,testLabels, filepath, getParams=True,
                                                filter=True, threshold = threshold)
        resultsRaw, SnSRaw = quantifyErrors(trueLabels,testLabels, filepath, getParams=False,
                                            filter=False, threshold = threshold)

        XFilt = resultsFilt[resultsFilt['Type']==type].iloc[:,3::3]
        labels = XFilt.columns.get_level_values(0)
        XFilt.columns = XFilt.columns.droplevel(1)

        XRaw = resultsRaw[resultsRaw['Type']==type].iloc[:,3::3]
        XRaw.columns = XRaw.columns.droplevel(1)

        nR = pd.isnull(XRaw).sum().sum()
        nF = pd.isnull(XFilt).sum().sum()
        ndf2 = np.empty(((nR+nF), 3), dtype = object)
        index = 0

        #Reorganizing data so that it can be plotted using violin plots
        for label in labels:
            xR = XRaw[label].values
            xR = xR[~pd.isnull(xR)]
            xF = XFilt[label].values
            xF = xF[~pd.isnull(xF)]

            lenR = len(xR)
            ndf2[index:index+lenR,0] = label
            ndf2[index:index+lenR,1] = xR
            ndf2[index:index+lenR,2] = "Unfiltered"
            index += lenR

            lenF = len(xF)
            ndf2[index:index+lenF,0] = label
            ndf2[index:index+lenF,1] = xF
            ndf2[index:index+lenF,2] = "Filtered"
            index += lenF

        df = pd.DataFrame(columns = ['label', 'Distance', 'Filt'], data = ndf2)
        sb.set(style="ticks")
        df['Distance'] = df['Distance'].astype(float)
        ax = sb.violinplot(y=df['Distance'], x=df['label'], data=df,split=True, hue='Filt', cut=0,
                        inner="quartile", palette={"Filtered":'azure',"Unfiltered":'lightcoral'})
        title = "Comparison of filtered and Unfiltered"

    #No comparison of data, can be filtered or unfiltered distributions
    else:
        results, SnS, params = quantifyErrors(trueLabels,testLabels, filepath, getParams=True,
                                                    filter=filter, threshold = threshold)
        X = results[results['Type']==type].iloc[:,3::3]
        labels = X.columns.get_level_values(0)
        X.columns = X.columns.droplevel(1)
        sb.violinplot(data=X, cut=0, inner = 'quartile')
        plt.xlabel('Distance')
        if filter:
            title = "Filtered"
        else:
            title = "Unfiltered"


    plt.title("{} Distance Distribution for each Label at threshold: {}".format(title, threshold))
    plt.ylim(0,50)

    plt.show()


    def plotSnS(trueLabels, testLabels, thresholds, filepath, snapshots=[] , save=True,
                   normalize=True, manual = True):
        """Short summary.

        Parameters
        ----------
        trueLabels : type
            Description of parameter `trueLabels`.
        testLabels : type
            Description of parameter `testLabels`.
        thresholds : type
            Description of parameter `thresholds`.
        filepath : type
            Description of parameter `filepath`.
        snapshots : type
            Description of parameter `snapshots`.
        save : type
            Description of parameter `save`.
        normalize : type
            Description of parameter `normalize`.

        Returns
        -------
        type
            Description of returned object.

        """

        matplotlib.use('WXAgg')

        #If you do not pass in the snapshots
        if not snapshots:
            avgOccurences, distances, snapshots = testThresholds(trueLabels, testLabels, filepath, thresholds)

        plt.style.use('fivethirtyeight')

        #ani = FuncAnimation(plt.gcf(), __animate2, fargs=(snapshots, normalize, thresholds,), interval=2000)

        fig = plt.figure(1)

        for i in range(len(thresholds)):
            if manual:
                keypress = False
                while not keypress:
                    keypress = plt.waitforbuttonpress()
            else:
                plt.pause(1)
            __animate(i,snapshots, normalize, thresholds)

        plt.close()

        #Need to fix writer
        #Writer = matplotlib.animation.writers['ffmpeg']
        #writer = Writer(fps=1)

        #plt.show()

def __animate(i, snapshots, normalize, thresholds):
    print(i) #Should be printed. Helps ensure plotting is occuring
    fontSize = [14,10,20]
    barWidth = .4
    b = snapshots[i]
    n_groups = b.shape[0]
    if(normalize):
        normalizer = [b['Training'].iloc[0,:].sum(), b['Test'].iloc[0,:].sum()]
    else:
        normalizer = [1,1]
    fig = plt.figure(1)
    plt.suptitle("Threshold: " + str(round(thresholds[i],4)), fontsize = fontSize[0])

    for j, errorType in enumerate(errorTypes):
        #fig, ax = plt.subplots()
        plt.subplot(2,2,j+1)
        plt.cla()
        index = np.arange(n_groups)

        r1 = plt.bar(index, (b['Training'][errorType]) / normalizer[0],
                     barWidth, label = 'Training')

        r2 = plt.bar(index + barWidth, (b['Test'][errorType]) / normalizer[1],
                     barWidth, label = 'Test')

        plt.xlabel('Label', fontsize = fontSize[0])
        plt.ylabel('% of Labels', fontsize = fontSize[0])
        plt.ylim(0, 1)
        plt.title('Training vs Test Comparison: ' + errorType, fontsize = fontSize[0])
        plt.xticks(index + barWidth/2, (b.index), fontsize = fontSize[0])
        plt.yticks(fontsize = fontSize[0])
        plt.legend(fontsize = fontSize[0])
        #fig.canvas.draw()
        plt.tight_layout()
    #ani.save('Error Data.mp4', writer=writer, dpi=1200)

#WIP  in notebooks in different forms. Need more data to continue working on it.
#def Score(snapshots, thresholds, ):


#Private Methods and helper functions (Do Not Call)
#-------------------------------------------------------------------------------------------------#
def __animate(i, snapshots, normalize, thresholds):
    print(i) #Should be printed. Helps ensure plotting is occuring
    fontSize = [14,10,20]
    barWidth = .4
    b = snapshots[i]
    n_groups = b.shape[0]
    if(normalize):
        normalizer = [b['Training'].iloc[0,:].sum(), b['Test'].iloc[0,:].sum()]
    else:
        normalizer = [1,1]
    fig = plt.figure(1)
    plt.suptitle("Threshold: " + str(round(thresholds[i],4)), fontsize = fontSize[0])

    for j, errorType in enumerate(errorTypes):
        #fig, ax = plt.subplots()
        plt.subplot(2,2,j+1)
        plt.cla()
        index = np.arange(n_groups)

        r1 = plt.bar(index, (b['Training'][errorType]) / normalizer[0],
                     barWidth, label = 'Training')

        r2 = plt.bar(index + barWidth, (b['Test'][errorType]) / normalizer[1],
                     barWidth, label = 'Test')

        plt.xlabel('Label', fontsize = fontSize[0])
        plt.ylabel('% of Labels', fontsize = fontSize[0])
        plt.ylim(0, 1)
        plt.title('Training vs Test Comparison: ' + errorType, fontsize = fontSize[0])
        plt.xticks(index + barWidth/2, (b.index), fontsize = fontSize[0])
        plt.yticks(fontsize = fontSize[0])
        plt.legend(fontsize = fontSize[0])
        #fig.canvas.draw()
        plt.tight_layout()
