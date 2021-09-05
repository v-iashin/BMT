import argparse
import os
import sys
import subprocess

import numpy as np
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from datasets.captioning_dataset import ActivityNetCaptionsDataset
# from datasets.load_features import load_features_from_npy
from datasets.load_features import crop_a_segment, pad_segment
from epoch_loops.captioning_epoch_loops import make_masks
from model.captioning_module import BiModalTransformer
from model.proposal_generator import MultimodalProposalGenerator
from utilities.proposal_utils import (get_corner_coords,
                                      remove_very_short_segments,
                                      select_topk_predictions, trim_proposals, non_max_suppresion)
from epoch_loops.captioning_epoch_loops import greedy_decoder

from typing import Dict, List, Union

class Config(object):
    # I need this to keep the name defined to load the config objects from model checkpoints.
    def __init__(self, to_log=True):
        pass

def load_features_from_npy(
        feature_paths: Dict[str, str], start: float, end: float, duration: float, pad_idx: int,
        device: int, get_full_feat=False, pad_feats_up_to: Dict[str, int] = None
    ) -> Dict[str, torch.Tensor]:
    '''Loads the pre-extracted features from numpy files.
    This function is conceptually close to `datasets.load_feature.load_features_from_npy` but cleaned up
    for demonstration purpose.

    Args:
        feature_paths (Dict[str, str]): Paths to the numpy files (keys: 'audio', 'rgb', 'flow).
        start (float, None): Start point (in secs) of a proposal, if used for captioning the proposals.
        end (float, None): Ending point (in secs) of a proposal, if used for captioning the proposals.
        duration (float): Duration of the original video in seconds.
        pad_idx (int): The index of the padding token in the training vocabulary.
        device (int): GPU id.
        get_full_feat (bool, optional): Whether to output full, untrimmed, feature stacks. Defaults to False.
        pad_feats_up_to (Dict[str, int], optional): If get_full_feat, pad to this value. Different for audio
                                                    and video modalities. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: A dict holding 'audio', 'rgb' and 'flow' features.
    '''

    # load features. Please see README in the root folder for info on video features extraction
    stack_vggish = np.load(feature_paths['audio'])
    stack_rgb = np.load(feature_paths['rgb'])
    stack_flow = np.load(feature_paths['flow'])

    stack_vggish = torch.from_numpy(stack_vggish).float()
    stack_rgb = torch.from_numpy(stack_rgb).float()
    stack_flow = torch.from_numpy(stack_flow).float()

    # for proposal generation we pad the features
    if get_full_feat:
        stack_vggish = pad_segment(stack_vggish, pad_feats_up_to['audio'], pad_idx)
        stack_rgb = pad_segment(stack_rgb, pad_feats_up_to['video'], pad_idx)
        stack_flow = pad_segment(stack_flow, pad_feats_up_to['video'], pad_idx=0)
    # for captioning use trim the segment corresponding to a prop
    else:
        stack_vggish = crop_a_segment(stack_vggish, start, end, duration)
        stack_rgb = crop_a_segment(stack_rgb, start, end, duration)
        stack_flow = crop_a_segment(stack_flow, start, end, duration)

    # add batch dimension, send to device
    stack_vggish = stack_vggish.to(torch.device(device)).unsqueeze(0)
    stack_rgb = stack_rgb.to(torch.device(device)).unsqueeze(0)
    stack_flow = stack_flow.to(torch.device(device)).unsqueeze(0)

    return {'audio': stack_vggish,'rgb': stack_rgb,'flow': stack_flow}


def load_prop_model(
        device: int, prop_generator_model_path: str, pretrained_cap_model_path: str, max_prop_per_vid: int
    ) -> tuple:
    '''Loading pre-trained proposal generator and config object which was used to train the model.

    Args:
        device (int): GPU id.
        prop_generator_model_path (str): Path to the pre-trained proposal generation model.
        pretrained_cap_model_path (str): Path to the pre-trained captioning module (prop generator uses the
                                         encoder weights).
        max_prop_per_vid (int): Maximum number of proposals per video.

    Returns:
        Config, torch.nn.Module: config, proposal generator
    '''
    # load and patch the config for user-defined arguments
    checkpoint = torch.load(prop_generator_model_path, map_location='cpu')
    cfg = checkpoint['config']
    cfg.device = device
    cfg.max_prop_per_vid = max_prop_per_vid
    cfg.pretrained_cap_model_path = pretrained_cap_model_path
    cfg.train_meta_path = './data/train.csv'  # in the saved config it is named differently

    # load anchors
    anchors = {
        'audio': checkpoint['anchors']['audio'],
        'video': checkpoint['anchors']['video']
    }

    # define model and load the weights
    model = MultimodalProposalGenerator(cfg, anchors)
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # if IncompatibleKeys - ignore
    model = model.to(cfg.device)
    model.eval()

    return cfg, model

def load_cap_model(pretrained_cap_model_path: str, device: int) -> tuple:
    '''Loads captioning model along with the Config used to train it and initiates training dataset
       to build the vocabulary including special tokens.

    Args:
        pretrained_cap_model_path (str): path to pre-trained captioning model.
        device (int): GPU id.

    Returns:
        Config, torch.nn.Module, torch.utils.data.dataset.Dataset: config, captioning module, train dataset.
    '''
    # load and patch the config for user-defined arguments
    cap_model_cpt = torch.load(pretrained_cap_model_path, map_location='cpu')
    cfg = cap_model_cpt['config']
    cfg.device = device
    cfg.pretrained_cap_model_path = pretrained_cap_model_path
    cfg.train_meta_path = './data/train.csv'

    # load train dataset just for special token's indices
    train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=False)

    # define model and load the weights
    model = BiModalTransformer(cfg, train_dataset)
    model = torch.nn.DataParallel(model, [device])
    model.load_state_dict(cap_model_cpt['model_state_dict'])  # if IncompatibleKeys - ignore
    model.eval()

    return cfg, model, train_dataset


def generate_proposals(
        prop_model: torch.nn.Module, feature_paths: Dict[str, str], pad_idx: int, cfg: Config, device: int,
        duration_in_secs: float
    ) -> torch.Tensor:
    '''Generates proposals using the pre-trained proposal model.

    Args:
        prop_model (torch.nn.Module): Pre-trained proposal model
        feature_paths (Dict): dict with paths to features ('audio', 'rgb', 'flow')
        pad_idx (int): A special padding token from train dataset.
        cfg (Config): config object used to train the proposal model
        device (int): GPU id
        duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
            `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

    Returns:
        torch.Tensor: tensor of size (batch=1, num_props, 3) with predicted proposals.
    '''
    # load features
    feature_stacks = load_features_from_npy(
        feature_paths, None, None, duration_in_secs, pad_idx, device, get_full_feat=True,
        pad_feats_up_to=cfg.pad_feats_up_to
    )

    # form input batch
    batch = {
        'feature_stacks': feature_stacks,
        'duration_in_secs': duration_in_secs
    }

    with torch.no_grad():
        # masking out padding in the input features
        masks = make_masks(batch['feature_stacks'], None, cfg.modality, pad_idx)
        # inference call
        predictions, _, _, _ = prop_model(batch['feature_stacks'], None, masks)
        # (center, length) -> (start, end)
        predictions = get_corner_coords(predictions)
        # sanity-preserving clipping of the start & end points of a segment
        predictions = trim_proposals(predictions, batch['duration_in_secs'])
        # fildering out segments which has 0 or too short length (<0.2) to be a proposal
        predictions = remove_very_short_segments(predictions, shortest_segment_prior=0.2)
        # seƒlect top-[max_prop_per_vid] predictions
        predictions = select_topk_predictions(predictions, k=cfg.max_prop_per_vid)

    return predictions

def caption_proposals(
        cap_model: torch.nn.Module, feature_paths: Dict[str, str],
        train_dataset: torch.utils.data.dataset.Dataset, cfg: Config, device: int, proposals: torch.Tensor,
        duration_in_secs: float
    ) -> List[Dict[str, Union[float, str]]]:
    '''Captions the proposals using the pre-trained model. You must specify the duration of the orignal video.

    Args:
        cap_model (torch.nn.Module): pre-trained caption model. Use load_cap_model() functions to obtain it.
        feature_paths (Dict[str, str]): dict with paths to features ('audio', 'rgb' and 'flow').
        train_dataset (torch.utils.data.dataset.Dataset): train dataset which is used as a vocab and for
                                                          specfial tokens.
        cfg (Config): config object which was used to train caption model. pre-trained model checkpoint has it
        device (int): GPU id to calculate on.
        proposals (torch.Tensor): tensor of size (batch=1, num_props, 3) with predicted proposals.
        duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
            `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

    Returns:
        List(Dict(str, Union(float, str))): A list of dicts where the keys are 'start', 'end', and 'sentence'.
    '''

    results = []

    with torch.no_grad():
        for start, end, conf in proposals.squeeze():
            # load features
            feature_stacks = load_features_from_npy(
                feature_paths, start, end, duration_in_secs, train_dataset.pad_idx, device
            )

            # decode a caption for each segment one-by-one caption word
            ints_stack = greedy_decoder(
                cap_model, feature_stacks, cfg.max_len, train_dataset.start_idx, train_dataset.end_idx,
                train_dataset.pad_idx, cfg.modality
            )
            assert len(ints_stack) == 1, 'the func was cleaned to support only batch=1 (validation_1by1_loop)'

            # transform integers into strings
            strings = [train_dataset.train_vocab.itos[i] for i in ints_stack[0].cpu().numpy()]

            # remove starting token
            strings = strings[1:]
            # and remove everything after ending token
            # sometimes it is not in the list (when the caption is intended to be larger than cfg.max_len)
            try:
                first_entry_of_eos = strings.index('</s>')
                strings = strings[:first_entry_of_eos]
            except ValueError:
                pass

            # join everything together
            sentence = ' '.join(strings)
            # Capitalize the sentence
            sentence = sentence.capitalize()

            # add results to the list
            results.append({
                'start': round(start.item(), 1),
                'end': round(end.item(), 1),
                'sentence': sentence
            })

    return results

def which_ffprobe() -> str:
    '''Determines the path to ffprobe library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffprobe_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffprobe_path

def get_video_duration(path):
    '''Determines the duration of the custom video
    Returns:
        float -- duration of the video in seconds'''
    cmd = f'{which_ffprobe()} -hide_banner -loglevel panic' \
          f' -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_duration = float(result.stdout.decode('utf-8').replace('\n', ''))
    print('Video Duration:', video_duration)
    return video_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='One video prediction')
    parser.add_argument('--prop_generator_model_path', required=True)
    parser.add_argument('--pretrained_cap_model_path', required=True)
    parser.add_argument('--vggish_features_path', required=True)
    parser.add_argument('--rgb_features_path', required=True)
    parser.add_argument('--flow_features_path', required=True)
    parser.add_argument('--duration_in_secs', type=float, required=True)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--max_prop_per_vid', type=int, default=5)
    parser.add_argument('--nms_tiou_thresh', type=float, help='removed if tiou > nms_tiou_thresh. In (0, 1)')
    args = parser.parse_args()

    feature_paths = {
        'audio': args.vggish_features_path,
        'rgb': args.rgb_features_path,
        'flow': args.flow_features_path,
    }

    # Loading models and other essential stuff
    cap_cfg, cap_model, train_dataset = load_cap_model(args.pretrained_cap_model_path, args.device_id)
    prop_cfg, prop_model = load_prop_model(
        args.device_id, args.prop_generator_model_path, args.pretrained_cap_model_path, args.max_prop_per_vid
    )
    # Proposal
    proposals = generate_proposals(
        prop_model, feature_paths, train_dataset.pad_idx, prop_cfg, args.device_id, args.duration_in_secs
    )
    # NMS if specified
    if args.nms_tiou_thresh is not None:
        proposals = non_max_suppresion(proposals.squeeze(), args.nms_tiou_thresh)
        proposals = proposals.unsqueeze(0)
    # Captions for each proposal
    captions = caption_proposals(
        cap_model, feature_paths, train_dataset, cap_cfg, args.device_id, proposals, args.duration_in_secs
    )

    print(captions)

    # TODO: save original num of features (0th dim.) and just remove padding from them and trim accordingly
