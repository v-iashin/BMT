import os

import torch
from tqdm import tqdm

from epoch_loops.captioning_epoch_loops import make_masks
from utilities.proposal_utils import AnetPredictions, add_dict_to_another_dict, calculate_f1, get_lr


def save_model(cfg, epoch, model, optimizer, scheduler, anet_metrics, best_metric):
    dict_to_save = {
        'config': cfg,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
        'anchors': model.anchors,
        'val_anet_metrics': anet_metrics,
        'best_metric': best_metric,
    }

    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    os.makedirs(cfg.log_path, exist_ok=True)
    path_to_save = os.path.join(cfg.log_path, f'best_prop_model.pt')
    torch.save(dict_to_save, path_to_save)
    
def train_av_loop(cfg, model, optimizer, loader, epoch, TBoard):
    model.train()
    train_total_loss = 0
    loss_acc_A = {}
    loss_acc_V = {}
    phase = 'train'
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        B = len(batch['video_ids'])
        masks = make_masks(batch['feature_stacks'], None, cfg.modality, loader.dataset.pad_idx)
        predictions, loss, losses_A, losses_V = model(batch['feature_stacks'], batch['targets'], masks)
        loss_acc_A = add_dict_to_another_dict(losses_A, loss_acc_A)
        loss_acc_V = add_dict_to_another_dict(losses_V, loss_acc_V)

        loss.backward()
        
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        train_total_loss += loss.item()

    train_total_loss /= len(loader)
    loss_acc_A = {k: v / len(loader) for k, v in loss_acc_A.items()}
    loss_acc_V = {k: v / len(loader) for k, v in loss_acc_V.items()}

    if TBoard is not None:
        TBoard.add_scalar('debug/loss_epoch', train_total_loss, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)
        for loss_name, value in loss_acc_A.items():
            TBoard.add_scalar(f'debug/train_{loss_name}_A', value, epoch)
        for loss_name, value in loss_acc_V.items():
            TBoard.add_scalar(f'debug/train_{loss_name}_V', value, epoch)
    else:
        print(f'Train Loss @ {epoch} epoch: {train_total_loss}')


def train_loop(cfg, model, optimizer, loader, epoch, TBoard):
    model.train()
    train_total_loss = 0
    loss_acc = {}
    phase = 'train'
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        B = len(batch['video_ids'])
        masks = make_masks(batch['feature_stacks'], None, cfg.modality, loader.dataset.pad_idx)
        predictions, loss, losses_dict = model(batch['feature_stacks'], batch['targets'], masks)
        loss_acc = add_dict_to_another_dict(losses_dict, loss_acc)
        
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
        optimizer.step()
        train_total_loss += loss.item()

    train_total_loss /= len(loader)
    loss_acc = {k: v / len(loader) for k, v in loss_acc.items()}

    if TBoard is not None:
        TBoard.add_scalar('debug/loss_epoch', train_total_loss, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)
        for loss_name, value in loss_acc.items():
            TBoard.add_scalar(f'debug/train_{loss_name}_iter', value, epoch)
    else:
        print(f'Train Loss @ {epoch} epoch: {train_total_loss}')

def validation_loop(cfg, model, optimizer, scheduler, loader, epoch, best_metric, TBoard):
    model.eval()
    phase = loader.dataset.phase
    anet_predictions = AnetPredictions(cfg, phase, epoch)
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        masks = make_masks(batch['feature_stacks'], None, cfg.modality, loader.dataset.pad_idx)

        with torch.no_grad():
            predictions, _, _, _ = model(batch['feature_stacks'], batch['targets'], masks)
            anet_predictions.add_new_predictions(predictions, batch)

    # predictions of the prop gen module are the same for val_1 & val_2.
    # Also, we evaluate preformance againts both of them. Hence,
    # There is no need to repeat it for val_2
    anet_predictions.write_anet_predictions_to_json()
    anet_metrics = anet_predictions.evaluate_predictions()
    if TBoard is not None:
        for tIoU in cfg.tIoUs:
            precision = anet_metrics[tIoU]['Precision']
            recall = anet_metrics[tIoU]['Recall']
            f1 = calculate_f1(recall, precision)
            TBoard.add_scalar(f'densevid_eval_k/precision_{tIoU}', precision, epoch)
            TBoard.add_scalar(f'densevid_eval_k/recall_{tIoU}', recall, epoch)
            TBoard.add_scalar(f'densevid_eval_k/F1_{tIoU}', f1, epoch)
        avg_precision = anet_metrics['Average across tIoUs']['Precision']
        avg_recall = anet_metrics['Average across tIoUs']['Recall']
        avg_f1 = calculate_f1(avg_recall, avg_precision)
        TBoard.add_scalar(f'metrics/avg_precision_at_k', avg_precision, epoch)
        TBoard.add_scalar(f'metrics/avg_recall_at_k', avg_recall, epoch)
        TBoard.add_scalar(f'metrics/avg_F1_at_k', avg_f1, epoch)

    if scheduler is not None:
        scheduler.step(avg_f1)

    if avg_f1 > best_metric and (TBoard is not None):
        best_metric = avg_f1
        save_model(cfg, epoch, model, optimizer, scheduler, anet_metrics, best_metric)
        print(f'Saved model @ {epoch} epoch. Best metric: {best_metric:.5f}')

    return best_metric
