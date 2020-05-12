from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import (BridgeConnection, FeatureEmbedder, Identity,
                          PositionalEncoder, VocabularyEmbedder)
from model.decoders import BiModelDecoder, Decoder
from model.encoders import BiModalEncoder, Encoder
from model.generators import Generator



class Transformer(nn.Module):
    
    def __init__(self, train_dataset, cfg):
        super(Transformer, self).__init__()
        self.modality = cfg.modality

        if cfg.modality == 'video':
            self.d_model = cfg.d_model_video
            self.d_feat = cfg.d_vid
            self.d_ff = cfg.d_ff_video
        elif cfg.modality == 'audio':
            self.d_feat = cfg.d_aud
            self.d_model = cfg.d_model_audio
            self.d_ff = cfg.d_ff_audio

        if cfg.use_linear_embedder:
            self.src_emb = FeatureEmbedder(self.d_feat, self.d_model)
        else:
            assert self.d_feat == self.d_model
            self.src_emb = Identity()
            
        self.trg_emb = VocabularyEmbedder(train_dataset.trg_voc_size, self.d_model)
        self.pos_emb = PositionalEncoder(self.d_model, cfg.dout_p)
        self.encoder = Encoder(self.d_model, cfg.dout_p, cfg.H, self.d_ff, cfg.N)
        self.decoder = Decoder(self.d_model, cfg.dout_p, cfg.H, self.d_ff, cfg.N)
        self.generator = Generator(self.d_model, train_dataset.trg_voc_size)
    
        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # initialize embedding after, so it will replace the weights initialized previously
        self.trg_emb.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        # load the pretrained encoder from the proposal (used in ablation studies)
        if cfg.pretrained_prop_model_path is not None:
            print(f'Pretrained prop path: \n {cfg.pretrained_prop_model_path}')
            cap_model_cpt = torch.load(cfg.pretrained_prop_model_path, map_location='cpu')
            encoder_config = cap_model_cpt['config']
            if cfg.modality == 'video':
                self.d_model = encoder_config.d_model_video
                self.d_ff = encoder_config.d_ff_video
            elif cfg.modality == 'audio':
                self.d_model = encoder_config.d_model_audio
                self.d_ff = encoder_config.d_ff_audio
            self.encoder = Encoder(
                self.d_model, encoder_config.dout_p, encoder_config.H, self.d_ff, encoder_config.N
            )
            encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
            encoder_weights = {k.replace('encoder.', ''): v for k, v in encoder_weights.items()}
            self.encoder.load_state_dict(encoder_weights)
            self.encoder = self.encoder.to(cfg.device)
            for param in self.encoder.parameters():
                param.requires_grad = cfg.finetune_prop_encoder
    
    def forward(self, src: dict, trg, masks: dict):
        '''
        In: src (B, Ss, d_feat) trg (B, St) src_mask (B, 1, Ss) trg_mask (B, St, St);
        Out: (B, St, voc_size)
        '''
        if self.modality == 'audio':
            src = src['audio']
            src_mask = masks['A_mask']
        elif self.modality == 'video':
            src = src['rgb'] + src['flow']
            src_mask = masks['V_mask']
        
        trg_mask = masks['C_mask']

        # embed
        src = self.src_emb(src)
        trg = self.trg_emb(trg)
        src = self.pos_emb(src)
        trg = self.pos_emb(trg)
        
        # encode and decode
        memory = self.encoder(src, src_mask)
        out = self.decoder(trg, memory, src_mask, trg_mask)
        
        # generate
        out = self.generator(out)
        
        return out
    

class BiModalTransformer(nn.Module):
    '''
    Forward:
        Inputs:
            src {'rgb'&'flow' (B, Sv, Dv), 'audio': (B, Sa, Da)}
            trg (C): ((B, Sc))
            masks: {'V_mask': (B, 1, Sv), 'A_mask': (B, 1, Sa), 'C_mask' (B, Sc, Sc))}
        Output:
            C: (B, Sc, Vc)
    '''
    def __init__(self, cfg, train_dataset):
        super(BiModalTransformer, self).__init__()

        if cfg.use_linear_embedder:
            self.emb_A = FeatureEmbedder(cfg.d_aud, cfg.d_model_audio)
            self.emb_V = FeatureEmbedder(cfg.d_vid, cfg.d_model_video)
        else:
            self.emb_A = Identity()
            self.emb_V = Identity()

        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model_caps)
        
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)
        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)

        self.encoder = BiModalEncoder(
            cfg.d_model_audio, cfg.d_model_video, cfg.d_model, cfg.dout_p, cfg.H, 
            cfg.d_ff_audio, cfg.d_ff_video, cfg.N
        )
        
        self.decoder = BiModelDecoder(
            cfg.d_model_audio, cfg.d_model_video, cfg.d_model_caps, cfg.d_model, cfg.dout_p, 
            cfg.H, cfg.d_ff_caps, cfg.N
        )

        self.generator = Generator(cfg.d_model_caps, train_dataset.trg_voc_size)

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # initialize embedding after, so it will replace the weights
        # of the prev. initialization
        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        # load the pretrained encoder from the proposal (used in ablation studies)
        if cfg.pretrained_prop_model_path is not None:
            print(f'Pretrained prop path: \n {cfg.pretrained_prop_model_path}')
            cap_model_cpt = torch.load(cfg.pretrained_prop_model_path, map_location='cpu')
            encoder_config = cap_model_cpt['config']
            self.encoder = BiModalEncoder(
                encoder_config.d_model_audio, encoder_config.d_model_video, encoder_config.d_model, 
                encoder_config.dout_p, encoder_config.H, encoder_config.d_ff_audio, 
                encoder_config.d_ff_video, encoder_config.N
            )
            encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
            encoder_weights = {k.replace('encoder.', ''): v for k, v in encoder_weights.items()}
            self.encoder.load_state_dict(encoder_weights)
            self.encoder = self.encoder.to(cfg.device)
            for param in self.encoder.parameters():
                param.requires_grad = cfg.finetune_prop_encoder

    def forward(self, src: dict, trg, masks: dict):
        V, A = src['rgb'] + src['flow'], src['audio']
        C = trg

        # (B, Sm, Dm) <- (B, Sm, Dm), m in [a, v]; 
        A = self.emb_A(A)
        V = self.emb_V(V)
        # (B, Sc, Dc) <- (S, Sc)
        C = self.emb_C(C)
        
        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        C = self.pos_enc_C(C)
        
        # notation: M1m2m2 (B, Sm1, Dm1), M1 is the target modality, m2 is the source modality
        Av, Va = self.encoder((A, V), masks)

        # (B, Sc, Dc)
        C = self.decoder((C, (Av, Va)), masks)
        
        # (B, Sc, Vc) <- (B, Sc, Dc) 
        C = self.generator(C)

        return C
