import hashlib
from functools import partial
from time import strptime, localtime, mktime
import sys
import os
from subprocess import call, check_output
from tqdm import tqdm
import json
import pandas as pd
import numpy as np

def average_metrics_in_two_dicts(val_1_metrics, val_2_metrics):
    '''
        both dicts must have the same keys
    '''
    val_metrics_avg = {}

    for key in val_1_metrics.keys():
        val_metrics_avg[key] = {}

        for metric_name in val_1_metrics[key].keys():
            val_1_metric = val_1_metrics[key][metric_name]
            val_2_metric = val_2_metrics[key][metric_name]
            val_metrics_avg[key][metric_name] = (val_1_metric + val_2_metric) / 2

    return val_metrics_avg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def timer(timer_started_at):
    # 20190911133748 (YmdHMS) -> struct time
    timer_started_at = strptime(timer_started_at, '%y%m%d%H%M%S')
    # struct time -> secs from 1900 01 01 etc
    timer_started_at = mktime(timer_started_at)
    
    now = mktime(localtime())
    timer_in_hours = (now - timer_started_at) / 3600
    
    return round(timer_in_hours, 2)


def md5sum(filename):
    '''from https://stackoverflow.com/a/7829658/3671347'''
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def make_metafile(available_mp4s_path, json_path, save_meta_path):
    AVAILABLE_MP4_FILE_HASH = 'bd38fb7d72b5d3ebff7d201e2938616a'
    # if os.path.exists(save_meta_path): #and md5sum(save_meta_path) == '':
    #     raise NotImplementedError

    # load the list of available vid ids
    if md5sum(available_mp4s_path) == AVAILABLE_MP4_FILE_HASH:
        available_mp4 = open(available_mp4s_path, 'r').readlines()
        available_mp4 = {filename.replace('\n', '') for filename in available_mp4}
    else:
        raise Exception(f'available_mp4.txt hash does not match the expected')

    video_ids = []
    captions = []
    starts = []
    ends = []
    durations = []

    with open(json_path, 'r') as read_f:
        json_dict = json.load(read_f)

    for video_id, info_dict in json_dict.items():
        # duration of the video
        duration = info_dict['duration']
        # extract a list of captions; each video may have multiple captions
        captions_list = info_dict['sentences']
        # and timestamps
        timestamps_list = info_dict['timestamps']

        # some videos became missing on YT
        if video_id not in available_mp4:
            continue

        for i, caption in enumerate(captions_list):
            # start and end timestamps
            start, end = timestamps_list[i]

            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            durations.append(duration)

    meta = pd.DataFrame({
        'video_id': video_ids,
        'caption': captions,
        'start': starts,
        'end': ends,
        'duration': durations,
    })

    # remove: change the apostrophe (removed` by nonascii); dots; \n; multiple spaces
    replace_dict = {
        "’": "'",
        '\.(?!\d)': '',
        '\n': ' ',
        '\s{2,}': ' ',
    }

    for pattern, val in replace_dict.items():
        meta['caption'] = meta['caption'].str.replace(pattern, val, regex=True)

    meta['caption'] = meta['caption'].str.strip()

    # split path into folder path and filename and select the file name
    # w/o an extension
    meta['phase'] = os.path.split(json_path)[1].replace('.json', '')
    meta['idx'] = meta.index

    # save to .csv file
    meta.to_csv(save_meta_path, index=None, sep='\t')
    

class HiddenPrints:
    '''
    Used in 1by1 validation in order to block printing of the enviroment 
    which is surrounded by this class 
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout