import os
import json
from time import time
import torch

from utilities.captioning_utils import HiddenPrints
from epoch_loops.captioning_epoch_loops import calculate_metrics
from sklearn.cluster import KMeans


def tiou_vectorized(segments1, segments2, without_center_coords=False, center_length=True):

    def center_length_2_start_end(segments):
        '''there is get_corner_coords(predictions) and has a bit diffrenrent logic. both are kept'''
        start = segments[:, 0] - segments[:, 1] / 2
        end = segments[:, 0] + segments[:, 1] / 2
        return start, end

    # add 'fake' center coordinates. You can use any value, we use zeros
    if without_center_coords:
        segments1 = torch.cat([torch.zeros_like(segments1), segments1], dim=1)
        segments2 = torch.cat([torch.zeros_like(segments2), segments2], dim=1)

    M, D = segments1.shape
    N, D = segments2.shape

    # TODO: replace with get_corner_coords from localization_utils
    if center_length:
        start1, end1 = center_length_2_start_end(segments1)
        start2, end2 = center_length_2_start_end(segments2)
    else:
        start1, end1 = segments1[:, 0], segments1[:, 1]
        start2, end2 = segments2[:, 0], segments2[:, 1]

    # broadcasting
    start1 = start1.view(M, 1)
    end1 = end1.view(M, 1)
    start2 = start2.view(1, N)
    end2 = end2.view(1, N)

    # calculate segments for intersection
    intersection_start = torch.max(start1, start2)
    intersection_end = torch.min(end1, end2)

    # we make sure that the area is 0 if size of a side is negative
    # which means that intersection_start > intersection_end which is not feasible
    # Note: adding one because the coordinates starts at 0 and let's
    intersection = torch.clamp(intersection_end - intersection_start, min=0.0)

    # finally we calculate union for each pair of segments
    union1 = (end1 - start1)
    union2 = (end2 - start2)
    union = union1 + union2 - intersection
    union = torch.min(torch.max(end1, end2) - torch.min(start1, start2), union)

    tious = intersection / (union + 1e-8)
    return tious


def read_segments_from_json(train_json_path):
    train_dict = json.load(open(train_json_path))
    # scaled_segment_lengths = []
    # segments = []
    segment_lengths = []

    for i, (video_id, video_info) in enumerate(train_dict.items()):
        # duration = video_info['duration']
        for start, end in video_info['timestamps']:
            segment_length = float(end) - float(start)
            if segment_length <= 0:
                continue
            # scaled_segment_length = segment_length / duration
            # scaled_segment_lengths.append(scaled_segment_length)
            segment_lengths.append(segment_length)
            # segments.append([float(start), float(end)])

    # scaled_segment_lengths = torch.tensor(scaled_segment_lengths).view(-1, 1)
    segment_lengths = torch.tensor(segment_lengths).view(-1, 1)
    # torch.Size([37421, 1])
    # return scaled_segment_lengths
    return segment_lengths


def calc_anchors_using_kmeans(train_json_path, k):
    # loading data
    # scaled_segment_lengths = read_segments_from_json(train_json_path)
    segment_lengths = read_segments_from_json(train_json_path)
    # kmeans
    kmeans = KMeans(n_clusters=k, random_state=13, init='random', n_init=1)
    # kmeans.fit(scaled_segment_lengths.numpy())
    kmeans.fit(segment_lengths.numpy())
    cluster_centers = kmeans.cluster_centers_.reshape(k)
    cluster_centers.sort()
    cluster_centers = list(cluster_centers)
    return cluster_centers


def calculate_f1(recall, precision):
    f1 = 2*recall*precision / (recall + precision + 1e-16)
    return f1


def filter_meta_for_video_id(meta, video_id, column_name='video_id'):
    return meta[meta[column_name] == video_id]


def get_center_coords(bboxes):
    # todo replace with segments
    return bboxes[:, 0] + (bboxes[:, 1] - bboxes[:, 0]) / 2


def get_corner_coords(predictions):
    '''predictions (B, S*A, num_feats)'''
    starts = predictions[:, :, 0] - predictions[:, :, 1] / 2
    ends = predictions[:, :, 0] + predictions[:, :, 1] / 2
    predictions[:, :, 0] = starts
    predictions[:, :, 1] = ends
    return predictions


def get_segment_lengths(bboxes):
    # todo replace with segments
    return (bboxes[:, 1] - bboxes[:, 0])


def add_dict_to_another_dict(one_dict, another_dict):
    another_dict = {k: another_dict.get(k, 0) + v for k, v in one_dict.items()}
    return another_dict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def select_topk_predictions(model_output, k):
    '''model_output (B, S*A, num_feats)'''
    B, S, num_feats = model_output.shape
    # sort model_output on confidence score (2nd col) within each batch
    # (B, S) <-
    indices = model_output[:, :, 2].argsort(descending=True)
    # (B, S, 1) <- .view()
    # (B, S, num_feats) <- .repeat()
    indices = indices.view(B, S, 1).repeat(1, 1, num_feats)
    model_output = model_output.gather(1, indices)
    # select top k
    # (B, k, num_feats) <-
    model_output = model_output[:, :k, :]
    return model_output


def trim_proposals(model_output, duration_in_secs):
    '''Changes in-place model_output (B, AS, num_feats), starts & ends are in seconds'''
    # for broadcasting it for batches
    duration_in_secs = torch.tensor(duration_in_secs, device=model_output.device).view(-1, 1)
    min_start = torch.tensor([0.0], device=model_output.device)
    # clip start for negative values and if start is longer than the duration
    model_output[:, :, 0] = model_output[:, :, 0].max(min_start).min(duration_in_secs)
    # clip end
    model_output[:, :, 1] = model_output[:, :, 1].min(duration_in_secs)
    return model_output

def remove_very_short_segments(model_output, shortest_segment_prior):
    model_output = model_output
    # (1, A*S) <-
    lengths = model_output[:, :, 1] - model_output[:, :, 0]
    # (A*S) <-
    lengths.squeeze_()
    # (A*S)
    model_output = model_output[:, lengths > shortest_segment_prior, :]

    return model_output


def non_max_suppresion(video_preds, tIoU_threshold):
    '''video_preds (AS, num_features)'''
    # model_output should be sorted according to conf_score, otherwise sort it here
    model_output_after_nms = []
    while len(video_preds) > 0:
        # (1, num_feats) <- (one_vid_pred[0, :].unsqueeze(0))
        model_output_after_nms.append(video_preds[0, :].unsqueeze(0))
        if len(video_preds) == 1:
            break
        # (1, *) <- (1, num_feats) x (*, num_feats)
        tious = tiou_vectorized(video_preds[0, :].unsqueeze(0), video_preds[1:, :], 
                                center_length=False)
        # (*) <- (1, *)
        tious = tious.reshape(-1)
        # (*', num_feats)
        video_preds = video_preds[1:, :][tious < tIoU_threshold]
    # (new_N, D) <- a list of (1, num_feats)
    model_output = torch.cat(model_output_after_nms)
    return model_output

def postprocess_preds(model_output, cfg, batch):
    '''
        model_output (B, AS, num_features) with center & length in grid cells
        1. Takes top-[max_prop_per_vid] predictions
        2. Converts values in grid coords into seconds
        3. Converts center & length into start & end
        4. Trims the segments according to sanity and original duration
    '''
    # select top-[max_prop_per_vid] predictions
    # (B, k, num_feats) <- (B, k, num_feats)
    model_output = select_topk_predictions(model_output, k=cfg.max_prop_per_vid)
    # (B, k, num_feats) <- (B, k, num_feats)
    model_output = get_corner_coords(model_output)
    # clip start & end to duration
    # (B, k, num_feats) <- (B, k, num_feats)
    model_output = trim_proposals(model_output, batch['duration_in_secs'])
    # (B, k, num_feats) <-
    return model_output


class AnetPredictions(object):

    def __init__(self, cfg, phase, epoch):
        self.predictions = {
            'version': 'VERSION 1.0',
            'external_data': {
                'used': True,
                'details': ''
            },
            'results': {}
        }
        self.phase = phase
        self.epoch = epoch
        self.cfg = cfg
        self.segments_used = 0
        self.segments_total = 0
        self.num_vid_w_no_props = 0

    def add_new_predictions(self, model_output, batch):
        '''
        model_output (B, AS, num_features)
        updates anet_prediction dict with the predictions from model_output
        '''
        model_output = postprocess_preds(model_output, self.cfg, batch)

        B, k, D = model_output.shape
        num_of_props_written = 0

        shortest_segment_prior = 0.2  # (sec)
        for b, video_preds in enumerate(model_output):
            vid_id = batch['video_ids'][b]
            vid_id_preds = []

            if self.cfg.nms_tiou_thresh is not None:
                # (nms_N, num_features)<- (AS, num_features)
                video_preds = non_max_suppresion(video_preds, self.cfg.nms_tiou_thresh)

            for pred_start, pred_end, pred_conf in video_preds.tolist():
                segment = {}
                start, end = round(pred_start, 5), round(pred_end, 5)
                if end - start > shortest_segment_prior:
                    segment['sentence'] = ''
                    segment['proposal_score'] = round(pred_conf, 5),
                    segment['timestamp'] = [start, end]
                    vid_id_preds.append(segment)
                    num_of_props_written += 1
            # sometimes all segmets are removed as they are too short. Hence, the preds are saved 
            # only if  at least one segment was added to predictions
            if len(vid_id_preds) > 0:
                self.predictions['results'][vid_id] = vid_id_preds
            else:
                # print(f'{vid_id} has empty proposal list')
                self.num_vid_w_no_props += 1

        self.segments_total += B * k
        self.segments_used += num_of_props_written
        
        num_of_props_written_per_video = num_of_props_written / B
        return num_of_props_written_per_video

    def write_anet_predictions_to_json(self):
        # save only val_1 because the props are the same. 1 not the 2 one because 1st has +30 vids
        if self.phase == 'val_1':
            submission_folder = os.path.join(self.cfg.log_path, 'submissions')
            filename = f'prop_results_{self.phase}_e{self.epoch}_maxprop{self.cfg.max_prop_per_vid}.json'
            self.submission_path = os.path.join(submission_folder, filename)
            os.makedirs(submission_folder, exist_ok=True)
            # if the same file name already exists, append random num to the path
            if os.path.exists(self.submission_path):
                self.submission_path = self.submission_path.replace('.json', f'_{time()}.json')
            with open(self.submission_path, 'w') as outf:
                json.dump(self.predictions, outf)
        else:
            raise NotImplementedError

    def evaluate_predictions(self):
        print(f'{self.cfg.max_prop_per_vid*self.segments_used/self.segments_total:.2f} props/vid')
        # during first epochs we have empty preds because we apply postprocess_preds() on preds
        if self.num_vid_w_no_props > 0:
            print(f'Number of videos with no proposals: {self.num_vid_w_no_props}')
        # blocks the printing
        with HiddenPrints():
            metrics = calculate_metrics(
                self.cfg.reference_paths, self.submission_path, self.cfg.tIoUs,
                self.cfg.max_prop_per_vid, verbose=True, only_proposals=True
            )
        return metrics
            
