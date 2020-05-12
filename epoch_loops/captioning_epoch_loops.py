import os
import json
from tqdm import tqdm
import torch
import spacy
from time import time

from model.masking import mask
from evaluation.evaluate import ANETcaptions
from datasets.load_features import load_features_from_npy
from utilities.captioning_utils import HiddenPrints, get_lr

def calculate_metrics(
    reference_paths, submission_path, tIoUs, max_prop_per_vid, verbose=True, only_proposals=False
):
    metrics = {}
    PREDICTION_FIELDS = ['results', 'version', 'external_data']
    evaluator = ANETcaptions(
        reference_paths, submission_path, tIoUs, 
        max_prop_per_vid, PREDICTION_FIELDS, verbose, only_proposals)
    evaluator.evaluate()
    
    for i, tiou in enumerate(tIoUs):
        metrics[tiou] = {}

        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            metrics[tiou][metric] = score

    # Print the averages
    
    metrics['Average across tIoUs'] = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        metrics['Average across tIoUs'][metric] = sum(score) / float(len(score))
    
    return metrics

def greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    assert model.training is False, 'call model.eval first'

    with torch.no_grad():
        
        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            masks = make_masks(feature_stacks, trg, modality, pad_idx)
            preds = model(feature_stacks, trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg

    
def save_model(cfg, epoch, model, optimizer, val_1_loss_value, val_2_loss_value, 
               val_1_metrics, val_2_metrics, trg_voc_size):
    
    dict_to_save = {
        'config': cfg,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_1_loss': val_1_loss_value,
        'val_2_loss': val_2_loss_value,
        'val_1_metrics': val_1_metrics,
        'val_2_metrics': val_2_metrics,
        'trg_voc_size': trg_voc_size,
    }
    
    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    os.makedirs(cfg.model_checkpoint_path, exist_ok=True)
    
#     path_to_save = os.path.join(cfg.model_checkpoint_path, f'model_e{epoch}.pt')
    path_to_save = os.path.join(cfg.model_checkpoint_path, f'best_cap_model.pt')
    torch.save(dict_to_save, path_to_save)


def make_masks(feature_stacks, captions, modality, pad_idx):
    masks = {}

    if modality == 'video':
        if captions is None:
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
    elif modality == 'audio':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        else:
            masks['A_mask'], masks['C_mask'] = mask(feature_stacks['audio'][:, :, 0], captions, pad_idx)
    elif modality == 'audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
    elif modality == 'subs_audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
        masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        masks['S_mask'] = mask(feature_stacks['subs'], None, pad_idx)

    return masks


def training_loop(cfg, model, loader, criterion, optimizer, epoch, TBoard):
    model.train()
    train_total_loss = 0
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        caption_idx = batch['caption_data'].caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)
        pred = model(batch['feature_stacks'], caption_idx, masks)
        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        loss = criterion(pred, caption_idx_y) / n_tokens
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)
            

def validation_next_word_loop(cfg, model, loader, decoder, criterion, epoch, TBoard, exp_name):
    model.eval()
    val_total_loss = 0
    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        caption_idx = batch['caption_data'].caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        with torch.no_grad():
            pred = model(batch['feature_stacks'], caption_idx, masks)
            n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            loss = criterion(pred, caption_idx_y) / n_tokens
            val_total_loss += loss.item()
            
    val_total_loss_norm = val_total_loss / len(loader)

    return val_total_loss_norm

def validation_1by1_loop(cfg, model, loader, decoder, epoch, TBoard):
    start_timer = time()
    
    # init the dict with results and other technical info
    predictions = {
        'version': 'VERSION 1.0',
        'external_data': {
            'used': True, 
            'details': ''
        },
        'results': {}
    }
    model.eval()
    loader.dataset.update_iterator()
    
    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    phase = loader.dataset.phase
    # feature_names = loader.dataset.feature_names
    
    if phase == 'val_1':
        reference_paths = [cfg.reference_paths[0]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'val_2':
        reference_paths = [cfg.reference_paths[1]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'learned_props':
        reference_paths = cfg.reference_paths  # here we use all of them
        tIoUs = cfg.tIoUs
        assert len(tIoUs) == 4

    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} 1by1 {epoch} @ {cfg.device}'
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        # caption_idx = batch['caption_data'].caption
        # caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        ### PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE
        ints_stack = decoder(
            model, batch['feature_stacks'], cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality
        )
        ints_stack = ints_stack.cpu().numpy()  # what happens here if I use only cpu?
        # transform integers into strings
        list_of_lists_with_strings = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack]
        ### FILTER PREDICTED TOKENS
        # initialize the list to fill it using indices instead of appending them
        list_of_lists_with_filtered_sentences = [None] * len(list_of_lists_with_strings)

        for b, strings in enumerate(list_of_lists_with_strings):
            # remove starting token
            strings = strings[1:]
            # and remove everything after ending token
            # sometimes it is not in the list
            try:
                first_entry_of_eos = strings.index('</s>')
                strings = strings[:first_entry_of_eos]
            except ValueError:
                pass
            # remove the period at the eos, if it is at the end (safe)
            # if trg_strings[-1] == '.':
            #     trg_strings = trg_strings[:-1]
            # join everything together
            sentence = ' '.join(strings)
            # Capitalize the sentence
            sentence = sentence.capitalize()
            # add the filtered sentense to the list
            list_of_lists_with_filtered_sentences[b] = sentence
            
        ### ADDING RESULTS TO THE DICT WITH RESULTS
        for video_id, start, end, sent in zip(batch['video_ids'], batch['starts'], batch['ends'],
                                              list_of_lists_with_filtered_sentences):
            segment = {
                'sentence': sent,
                'timestamp': [start.item(), end.item()]
            }

            if predictions['results'].get(video_id):
                predictions['results'][video_id].append(segment)

            else:
                predictions['results'][video_id] = [segment]
    
    if cfg.log_path is None:
        return None
    else:
        ## SAVING THE RESULTS IN A JSON FILE
        save_filename = f'captioning_results_{phase}_e{epoch}.json'
        submission_path = os.path.join(cfg.log_path, save_filename)

        # in case TBoard is not defined make logdir
        os.makedirs(cfg.log_path, exist_ok=True)

        # if this is run with another loader and pretrained model
        # it substitutes the previous prediction
        if os.path.exists(submission_path):
            submission_path = submission_path.replace('.json', f'_{time()}.json')

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf)

        ## RUN THE EVALUATION
        # blocks the printing
        with HiddenPrints():
            val_metrics = calculate_metrics(reference_paths, submission_path, tIoUs, cfg.max_prop_per_vid)

        if phase == 'learned_props':
            print(submission_path)

        ## WRITE TBOARD
        if (TBoard is not None) and (phase != 'learned_props'):
            # todo: add info that this metrics are calculated on val_1
            TBoard.add_scalar(f'{phase}/meteor', val_metrics['Average across tIoUs']['METEOR'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu4', val_metrics['Average across tIoUs']['Bleu_4'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu3', val_metrics['Average across tIoUs']['Bleu_3'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/precision', val_metrics['Average across tIoUs']['Precision'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/recall', val_metrics['Average across tIoUs']['Recall'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/duration_of_1by1', (time() - start_timer) / 60, epoch)

        return val_metrics
