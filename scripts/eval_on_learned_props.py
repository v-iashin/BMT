import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.captioning_dataset import ActivityNetCaptionsDataset
from epoch_loops.captioning_epoch_loops import greedy_decoder, validation_1by1_loop
from model.captioning_module import BiModalTransformer, Transformer

def convert_props_in_json_to_csv(prop_pred_path, val_1_json_path, avail_mp4_path):
    '''
    To convert the produced proposals
    val_json only used for the information about the original duration.
    Note: val_1_json_path is val_1 and can be val_2 but val_1 is ~ 30 videos longer.
    '''
    assert 'val_1' in val_1_json_path, f'Is it the val_1 json: {val_1_json_path}'
    pred_csv_path = prop_pred_path.replace('.json', '.csv')

    if os.path.exists(pred_csv_path):
        print(f'File {pred_csv_path} already exists. I will use it.')
        return pred_csv_path

    # Format of .json
    # {'details': ..., 'version': ..., 'results': {'id': [{'sentence': str, 'timestamp': [s, e]}]}}
    pred_json = json.load(open(prop_pred_path))['results']

    vid2duration = {video: v['duration'] for video, v in json.load(open(val_1_json_path)).items()}

    video_ids = []
    starts = []
    ends = []
    durations = []

    with open(avail_mp4_path) as in_f:
        avail_vid_ids = {fname.replace('.mp4', '').replace('\n', '') for fname in in_f.readlines()}

    for i, (video_id, props_info_list) in enumerate(tqdm(pred_json.items(), desc='Preds to .csv')):
        
        if (video_id not in avail_vid_ids) or (video_id not in vid2duration):
            # some videos are missing from val_1 and val_2 datasets but are present in val_ids:
            # {'v_5F4jcV8dHVs', 'v_4G2jW3hbiO4', 'v_f-Cf16fQTB4', 'v_G_US7iMc6Y4', 'v_jqYzz6YoMEY',
            # 'v__tRAypMWUdc', 'v_UBQfURrVB_Y', 'v_Vre3tO7xV98', 'v_b8pCuIPzb3o'}
            continue

        for prop_info in props_info_list:
            start, end = prop_info['timestamp']
            video_ids.append(video_id)
            starts.append(start)
            ends.append(end)
            durations.append(vid2duration[video_id])

    # mind the order of columns. See getitem in internal Datasets
    dataframe = pd.DataFrame({
        'video_id': video_ids,
        'caption_pred': ['PLACEHOLDER'] * len(video_ids),
        'start': starts,
        'end': ends,
        'duration': durations,
    })

    # extract phase from the gt json path
    dataframe['phase'] = 'val_1'
    dataframe['idx'] = dataframe.index

    # save to .csv file
    dataframe.to_csv(pred_csv_path, index=None, sep='\t')

    return pred_csv_path

def check_args(cfg):
    if 'audio' in cfg.modality:
        assert os.path.exists(cfg.audio_features_path), f'{cfg.audio_features_path}'
    if 'video' in cfg.modality:
        assert os.path.exists(cfg.video_features_path), f'{cfg.video_features_path}'

class Config(object):
    # I need to keep the name defined to loadd the model
    def __init__(self, to_log=True):
        pass

def eval_on_learned_props(args):
    cap_model_cpt = torch.load(args.pretrained_cap_model_path, map_location='cpu')
    cfg = cap_model_cpt['config']
    cfg.max_prop_per_vid = args.max_prop_per_vid
    cfg.device = args.device_ids[0]
    # in case log_path has moved (remove trailing .best_*_model.pt)
    cfg.log_path = os.path.split(args.pretrained_cap_model_path)[0]
    if 'audio' in cfg.modality:
        cfg.audio_features_path = args.audio_features_path
    if 'video' in cfg.modality:
        cfg.video_features_path = args.video_features_path

    check_args(cfg)

    # returns path where .csv was saved, which is prop_pred_path's folder
    # we change the content of cfg only once---here. The motivation is a more clear code for
    # the dataset initialization (caption_iterator)
    cfg.val_prop_meta_path = convert_props_in_json_to_csv(
        args.prop_pred_path, cfg.reference_paths[0], args.avail_mp4_path, 
    )
    print(cfg.log_path)

    TBoard = None
    
    # continue from here
    train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=False)
    pred_prop_dataset = ActivityNetCaptionsDataset(cfg, 'learned_props', get_full_feat=False)

    val_pred_prop_loader = DataLoader(pred_prop_dataset, collate_fn=pred_prop_dataset.dont_collate)
    print(f'Loader will use: {val_pred_prop_loader.dataset.meta_path}')

    if cfg.modality == 'audio_video':
        model = BiModalTransformer(cfg, train_dataset)
    elif cfg.modality in ['audio', 'video']:
        model = Transformer(train_dataset, cfg)

    device = torch.device(args.device_ids[0])
    torch.cuda.set_device(device)
    model = torch.nn.DataParallel(model, [device])
    model.load_state_dict(cap_model_cpt['model_state_dict'])  # if IncompatibleKeys - ignore
    model.eval()
    
    # load the best model
    val_metrics_pred_prop = validation_1by1_loop(
        cfg, model, val_pred_prop_loader, greedy_decoder, cap_model_cpt['epoch'], TBoard
    )

    print(val_metrics_pred_prop)
    
    # If you missed the scores after printing you may find it in the args.pretrained_cap_model_path
    # folder with under the 'results_learned_props_e{epoch}.json' name
