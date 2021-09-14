#Rewriting the selection method, have done the sorting idx adn getting adjusted
#frame idx, need to get file paths

#Was accidently using the complex movements test video in the iterative
#training framework


import shutil as sh
import deeplabcut as dlc
import os
import pandas as pd
import numpy as np
import dlc_tools as dt
import pickle
import random

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

import sys
import ruamel.yaml
import deeplabcut.utils.auxfun_videos as aux
from ruamel.yaml.comments import CommentedMap

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
import plotly.figure_factory as ff

from IPython.display import HTML

def View(df):
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))




def evaluate_frames(error_data, min_points = 18, total_points = 24, dist_comparator = 12, return_scores = False):
    """Short summary.

    Evaluates the quality of frames assigning frames either a boolean tag (good=True and bad=False)
    or a scores out of 24

    Parameters
    ----------
    error_data : type = np.arrary
        The error data that contains the distance, error type, and confidence for each prediction.
        Each row corresponds to one frame. Note: The error data is the output of the quantifyErrors
        function converted into a np array
    min_points : type = int
        The minimum number of points that must be correct for a frame to be considered "good". Only
        applicable if you are doing binary frame evaluation. Default is 18
    total_points : dtype = int
        The total number of labels/bodyparts you are tracking
    dist_comparator : dtype = int
        The maximum allowable distance error. If the distance for a point is greater than this value
        the point will be reclassified as a False Positive (FP). Default is 12 pixels
    return_scores : dtype = boolean
        A boolean flag that when set to true will return scores. When set to false will return
        binary evaluations of the frames

    Returns
    -------
    frame_eval : dtype = np.array
        An array of boolean flags that say whether a frame is good (well labeled) or bad
    scores : dtype = np.array
        An array of scores correspoding to each frame being evaluated

    """

    threshold = min_points/total_points
    frame_eval = np.zeros(len(error_data), dtype = bool)
    scores = []
    for idx in range(len(error_data)):
        score = 0
        row = error_data[idx, :]
        error_type = row[1::3]
        dist = row[3::3]

        score += len([err for err in error_type if err=='TN'])

        for j in range(len(error_type)):

            if error_type[j]== 'TP' and dist[j] < dist_comparator:
                score+=1

        scores.append(score)
        if (score/total_points >= threshold):
            frame_eval[idx] = True

    if return_scores:
        return np.asarray(scores)
    else:
        return frame_eval

def calc_vel(data, missing_value = -1, threshold = .9):
    """Short summary.


    Parameters
    ----------


    Returns
    -------


    """
    x_coords = data.iloc[:, 0::3].values
    y_coords = data.iloc[:, 1::3].values
    likelihood = data.iloc[:,2::3].values

    dx = x_coords[1:] - x_coords[0:-1]
    dy = y_coords[1:] - y_coords[0:-1]

    vel = np.zeros(np.shape(likelihood))
    vel[1:,:] = np.power( np.power(dx, 2) + np.power(dy, 2), .5)
    idx = likelihood<threshold
    vel[idx] = missing_value

    return vel

def getInputOutput(vid_names, DATA, min_points, dist_comparator):
    """Short summary.


    Parameters
    ----------


    Returns
    -------


        """
    frame_evals=[]
    vels = []
    vid_idx = {}

    num_videos = len(vid_names)
    cur_idx = 0
    loopcount = 0
    for vid_name in vid_names:
        data = DATA[vid_name]
        if len(data['True'] != len(data['Test'])):
            count = min(len(data['True']), len(data['Test']))
            data['True'] = data['True'].iloc[0:count, :]
            data['Test'] = data['Test'].iloc[0:count, :]

        vid_idx[vid_name] = [loopcount, (cur_idx, cur_idx + len(data['True']))]
        cur_idx += len(data['True'])
        results, sns = dt.quantifyErrors(data['True'], data['Test'], filter = False, threshold =.9)

        result_data = results.values

        frame_evals.append(evaluate_frames(result_data, min_points = min_points, dist_comparator=dist_comparator))
        vels.append(calc_vel(data['Test'], missing_value=-1))
        loopcount += 1
    output = np.concatenate((frame_evals), axis=0)
    input_data = np.array(np.concatenate((vels), axis=0), dtype=int)
    return input_data, output, vid_idx

def test_model(input_data, output, count = 3, print_loop=True, print_mean = True, get_meta = False):
    """Short summary.


    Parameters
    ----------


    Returns
    -------


    """    scores = []
    meta = {}
    for i in range(count):


        #x_train, x_test, y_train, y_test = train_test_split(input_data, output, test_size = .3)

        count = len(output)
        random = np.random.rand(count)
        tmpidx = np.linspace(0,count-1,count, dtype = int)
        idx = tmpidx[np.argsort(random)]
        split_prop = .3 #test set percentage
        split_idx = int(len(idx) * split_prop)

        test_idx = idx[0:split_idx]
        train_idx = idx[split_idx:]
        x_test = input_data[test_idx]
        x_train = input_data[train_idx]
        y_test = output[test_idx]
        y_train = output[train_idx]


        clf = svm.SVC(kernel='linear')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

        y = clf.decision_function(x_test)
        w_norm = np.linalg.norm(clf.coef_)
        dist = y / w_norm

        if print_loop:
            print("Accuracy {}:".format(i), metrics.accuracy_score(y_test, y_pred))
        tmp_meta = {}
        if get_meta:
            tmp_meta['x_train'] = x_train
            tmp_meta['y_train'] = y_train
            tmp_meta['x_test'] = x_test
            tmp_meta['y_test'] = y_test
            tmp_meta['y_pred'] = y_pred
            tmp_meta['score'] = np.mean(scores)
            tmp_meta['dist'] = dist
            tmp_meta['split_idx'] = split_idx
            tmp_meta['test_idx'] = test_idx
            tmp_meta['train_idx'] = train_idx
        meta[str(i)] = tmp_meta
    if print_mean:
        print("Mean Accuracy", np.mean(scores))
    if get_meta:
        return scores, meta
    else:
        return scores

#Retrieves the labeling files in the vid dir, defaults to fully labeled data folder
def getData(shuffle, keyword = 'Iterative', manual_only = False, username = 'beastmode', vid_dir = None):
    if vid_dir == None:
        videos_dir = '/home/{}/DeepLabCut/Fully Labeled Data/Model 1 Videos/Test/'.format(username)
        base_dir = '/home/{}/DeepLabCut/'.format(username)
        vid_dir = base_dir + 'Fully Labeled Data/Model 1 Videos/Test/'

    vid_names = []
    for file in os.listdir(vid_dir):
        vid_names.append(file)
    DATA = {}

    if manual_only:
        for vid in vid_names:
            files = [file for file in os.listdir(vid_dir + vid) if file.endswith('h5')]
            DATA[vid] = {}
            true_data_path = [vid_dir + vid + '/' + file for file in files if 'manual' in file.lower()][0]
            DATA[vid]['True'] = pd.read_hdf(true_data_path)
    else:
        for vid in vid_names:
            files = [file for file in os.listdir(vid_dir + vid) if file.endswith('h5')]
            DATA[vid] = {}
            true_data_path = [vid_dir + vid + '/' + file for file in files if 'manual' in file.lower()][0]
            test_data_path = [vid_dir + vid + '/' + file for file in files if 'manual' not in file.lower() and keyword in file and 'shuffle{}'.format(shuffle) in file][0]
            DATA[vid]['True'] = pd.read_hdf(true_data_path)
            DATA[vid]['Test'] = pd.read_hdf(test_data_path).iloc[0:len( DATA[vid]['True'])]
    return DATA

def getPredictions(train_vid_names, DATA, model):
    data = []
    Y = {}
    frame_count = {}
    cutoffs = {}
    count = 0

    for vid in train_vid_names:
        data = DATA[vid]['Test']
        frame_count[vid] = len(data.index)
        vels = calc_vel(data, threshold = .5)
        y_pred = model.predict(vels)
        Y[vid] = y_pred

        numel = len(Y[vid])
        cutoffs[vid] = (count, count + numel)
        count = count + numel

    predictions = np.concatenate([Y[key] for key in Y.keys()], axis=0)

    return predictions, cutoffs, frame_count

'''
def getFrames(cutoffs, predictions, svm_train_data, train_vid_names, frame_count, num2pick = 50):
    with open('/home/beastmode/DeepLabCut/Iterative_Training-Nick_T-2021-05-31/past_frame_idx.pickle', 'rb') as file:
        past_frame_idx = pickle.load(file)

    adjusted_past_frame_idx = {}

    prior_frames = []
    train_frames = []
    for vid in train_vid_names:
        cutoff = cutoffs[vid]
        adjusted_past_frame_idx[vid] = np.asarray(past_frame_idx[vid]) + cutoff[0]
        prior_frames.append(adjusted_past_frame_idx[vid])
        train_frames.append(svm_train_data[vid] + cutoff[0])

    prior_frames = np.concatenate(prior_frames)
    train_frames = np.concatenate(train_frames)

    idx = np.linspace(0,len(predictions)-1,len(predictions), dtype = int)

    if len(idx[predictions == False]) < num2pick:
           selected_frames = random.sample(list(idx), num2pick)
    else:
           selected_frames = random.sample(list(idx[predictions == False]), num2pick)

    selected_frames = [frame for frame in selected_frames if (frame not in prior_frames) and (frame not in train_frames)]

    while len(selected_frames) < num2pick:
        if len(idx[predictions == False]) < num2pick:
            tmp_idx = random.choice(list(idx))
        else:
            tmp_idx = random.choice(list(idx[predictions == False]))
        if (tmp_idx not in selected_frames) and (tmp_idx not in prior_frames) and (tmp_idx not in train_frames):
            selected_frames.append(tmp_idx)

    selected_frames = np.asarray(selected_frames)

    sorted_frames = {}
    for vid in cutoffs:
        cutoff = cutoffs[vid]
        check_lower = selected_frames >= cutoff[0]
        check_higher = selected_frames < cutoff[1]
        check = check_lower * check_higher
        sorted_frames[vid] = selected_frames[check] - cutoff[0]
        past_frame_idx[vid] = np.concatenate([past_frame_idx[vid], sorted_frames[vid]])

    with open('/home/nickt/DeepLabCut/Iterative_Training-Nick_T-2021-05-31/past_frame_idx.pickle', 'wb') as file:
        pickle.dump(past_frame_idx, file)
        file.close()

    for folder in iter_folder:
        vid = [file for file in os.listdir(folder) if file.endswith('avi')]
        vid = vid[0][:-4]
        prfx_len =  len(str(frame_count[vid] - 1))

        #There is an error where the DLC file did not have labeling from frames 977-1026 because they are blank
        #This means the labeling convention is four digits 0000 - 0977 for frame names
        #This has to be manually adjusted for this video only
        if vid == 'Simple_Movements':
            prfx_len += 1

        frames = []
        for frame in sorted_frames[vid]:

            prfx =  folder + 'images/img{}'.format('0' * (prfx_len - len(str(frame))))
            sfx = '.png'
            frames.append(prfx+str(frame)+sfx)
        sorted_frames[vid] = frames
    return sorted_frames
'''
def updateConfig(train_vids, dlc_dir):

    os.chdir(dlc_dir)
    cfg = dlc_dir + '/config.yaml'
    yaml = ruamel.yaml.YAML()
    with open(cfg, 'r') as file:
        data = yaml.load(file)
        scorer = data['scorer']

        video_sets = {}
        for d in data['video_sets']:
            video_sets[d] = {'crop' : data['video_sets'][d]['crop']}
            crop =  data['video_sets'][d]['crop']

        for vid in train_vids:
            video_sets[vid] = CommentedMap({'crop': crop})

        data['video_sets'] = video_sets

    with open(cfg, 'w') as file:
        yaml.dump(data, file)

    return scorer

def updateTrainingData(shuffle, pcf, DATA, train_vid_names, sorted_frames, dlc_dir, trainingset_dir,
                        scorer = 'Nick_T'):
    curdir = dlc_dir + '/labeled-data/'
    frame_meta_files = [file for file in os.listdir(trainingset_dir) if 'CollectedData' in file]

    frame_meta_files
    if 'Backup' not in os.listdir(trainingset_dir):
        os.mkdir(os.path.join(trainingset_dir, 'Backup'))
    for file in frame_meta_files:
        src = os.path.join(trainingset_dir, file)
        backup = os.path.join(trainingset_dir, 'Backup')
        dest = os.path.join(backup, 'Shuffle{}_'.format(shuffle) + file)
        sh.copyfile(src, dest)

    trainingset = pd.read_hdf(trainingset_dir + file)
    num_frames= 0
    for vid in train_vid_names:
        curdir + vid
        tmp_dir = os.path.join(curdir, vid)
        idx = [s.split('/')[-1][3:-4] for s in sorted_frames[vid]]

        if vid not in os.listdir(curdir):
            os.mkdir(tmp_dir)
         #   tmp_scorer = tmpdata.columns.get_level_values(0)[0]
            tmpdata = DATA[vid]['True'].iloc[idx]

            tmpdata.to_hdf(tmp_dir + '/' + 'CollectedData_{}.h5'.format(scorer), key = 'df_with_missing')
            tmpdata.to_csv(tmp_dir + '/' + 'CollectedData_{}.csv'.format(scorer))
        else:
            tmpdata = DATA[vid]['True'].iloc[idx]
            df = pd.read_hdf(tmp_dir + '/' + 'CollectedData_{}.h5'.format(scorer))
            tmpdata = pd.concat([df, tmpdata])
            tmpdata = tmpdata.sort_index()


            tmpdata.to_hdf(tmp_dir + '/' + 'CollectedData_{}.h5'.format(scorer), key = 'df_with_missing')
            tmpdata.to_csv(tmp_dir + '/' + 'CollectedData_{}.csv'.format(scorer))


        tmp_df = pd.DataFrame(data = tmpdata.values, index = tmpdata.index, columns = trainingset.columns)
        trainingset = pd.concat([trainingset, tmp_df])
        for frame in sorted_frames[vid]:
            sh.copyfile(frame, tmp_dir + '/' + frame.split('/')[-1])
        num_frames += len(tmpdata)


    trainingset.to_hdf(trainingset_dir + 'CollectedData_{}.h5'.format(scorer), key = 'df_with_missing')
    trainingset.to_csv(trainingset_dir + 'CollectedData_{}.csv'.format(scorer))
    dlc.create_training_dataset(pcf, Shuffles=[shuffle], net_type = 'mobilenet_v2_1.0')#, trainIndexes=[train_indices], testIndexes=[test_indices])

def getCutoffs(train_vid_names, DATA):
    cutoffs = {}
    count = 0

    for vid in train_vid_names:
        numel = len(DATA[vid]['True'])
        cutoffs[vid] = (count, count + numel)
        count = count + numel


    return cutoffs

def adjust_idx(vid, cutoffs, idx):
    cutoff = cutoffs[vid]
    idx = np.asarray(idx) + cutoff[0]
    return idx


def idx_to_vid(idx, cutoffs, scores = []):
    sorted_frames = {}
    sorted_scores = {}
    for vid in cutoffs.keys():
        cutoff = cutoffs[vid]
        check_lower = idx >= cutoff[0]
        check_higher = idx < cutoff[1]
        check = check_lower * check_higher
        if len(scores) > 0:
            sorted_scores[vid] = []
            tmp_scores = scores[check]
            tmp_idx = idx[check]
            for n in range(sum(check)):
                sorted_scores[vid].append((tmp_idx[n] - cutoff[0], tmp_scores[n]))
        sorted_frames[vid] = idx[check] - cutoff[0]
    if len(scores) > 0:
        return sorted_frames, sorted_scores
    else:
        return sorted_frames
