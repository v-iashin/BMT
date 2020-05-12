import numpy as np
from torch.utils import tensorboard as tensorboard
# import tensorboardX as tensorboard
import torch
from torch.utils.data import DataLoader

from datasets.captioning_dataset import ActivityNetCaptionsDataset
from datasets.proposal_dataset import ProposalGenerationDataset
from epoch_loops.proposal_epoch_loops import train_loop, train_av_loop, validation_loop
from model.proposal_generator import MultimodalProposalGenerator, ProposalGenerator
from utilities.proposal_utils import calc_anchors_using_kmeans
from utilities.config_constructor import Config
from utilities.captioning_utils import timer

def train_prop(cfg):
    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # preventing PyTorch from allocating memory on the default device (cuda:0) when the desired
    # cuda id for training is not 0.
    torch.cuda.set_device(cfg.device_ids[0])

    # exp_name = cfg.make_experiment_name()
    exp_name = cfg.curr_time[2:]

    anchors = {}
    if 'audio' in cfg.modality:
        anchors['audio'] = calc_anchors_using_kmeans(cfg.train_json_path, cfg.anchors_num_audio)
    if 'video' in cfg.modality:
        anchors['video'] = calc_anchors_using_kmeans(cfg.train_json_path, cfg.anchors_num_video)

    # ActivityNetCaptionsDataset() is used only for pad_idx
    train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=True)

    train_dataset = ProposalGenerationDataset(cfg, 'train', train_dataset.pad_idx)
    # we only need val_1 as the predictions are going to be the same for both val_1 and val_2
    # because videos are the same
    valid_dataset = ProposalGenerationDataset(cfg, 'val_1', train_dataset.pad_idx)

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                              batch_size=cfg.train_batch_size, 
                              collate_fn=train_dataset.collate4proposal_generation)
    valid_loader = DataLoader(valid_dataset, shuffle=False,
                              batch_size=cfg.inference_batch_size,
                              collate_fn=valid_dataset.collate4proposal_generation)
    
    if cfg.modality == 'audio_video':
        model = MultimodalProposalGenerator(cfg, anchors)
    else:
        model = ProposalGenerator(cfg, anchors)

    model = model.to(cfg.device)
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), cfg.lr, cfg.momentum, 
                                    weight_decay=cfg.weight_decay)

    if cfg.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_reduce_factor, patience=cfg.lr_patience
        )
    else:
        scheduler = None

    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')

    if cfg.to_log:
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path)
        print(f'saving log @ {cfg.log_path}')
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    best_metric = -np.inf  # F1
    num_epoch_best_metric_unchanged = 0
    
    for epoch in range(cfg.epoch_num):
        print(f'The best metrict was unchanged for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch+cfg.early_stop_after-num_epoch_best_metric_unchanged}')
        print(f'Started @ {cfg.curr_time}; Current timer: {timer(cfg.curr_time)}')
        # stop training if metric hasn't been changed for cfg.early_stop_after epochs
        if num_epoch_best_metric_unchanged == cfg.early_stop_after:
            break
        
        if cfg.modality == 'audio_video':
            train_av_loop(cfg, model, optimizer, train_loader, epoch, TBoard)
        else:
            train_loop(cfg, model, optimizer, train_loader, epoch, TBoard)
        current_metric = validation_loop(
            cfg, model, optimizer, scheduler, valid_loader, epoch, best_metric, TBoard
        )

        if current_metric > best_metric:
            best_metric = current_metric
            num_epoch_best_metric_unchanged = 0
        else:
            num_epoch_best_metric_unchanged += 1
            

    print(f'Experiment: {exp_name}')
