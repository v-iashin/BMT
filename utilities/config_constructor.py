import os
from time import localtime, strftime
from shutil import copytree, ignore_patterns

class Config(object):
    '''
    Note: don't change the methods of this class later in code.
    '''

    def __init__(self, args):
        '''
        Try not to create anything here: like new forders or something
        '''
        self.curr_time = strftime('%y%m%d%H%M%S', localtime())

        self.procedure = args.procedure
        # dataset
        self.train_meta_path = args.train_meta_path
        self.val_1_meta_path = args.val_1_meta_path
        self.val_2_meta_path = args.val_2_meta_path
        self.modality = args.modality
        self.video_feature_name = args.video_feature_name
        self.audio_feature_name = args.audio_feature_name
        self.video_features_path = args.video_features_path
        self.audio_features_path = args.audio_features_path
        # make them d_video and d_audio
        self.d_vid = args.d_vid
        self.d_aud = args.d_aud
        self.start_token = args.start_token
        self.end_token = args.end_token
        self.pad_token = args.pad_token
        self.max_len = args.max_len
        self.min_freq_caps = args.min_freq_caps

        # model
        if args.procedure == 'train_cap':
            self.word_emb_caps = args.word_emb_caps
            self.unfreeze_word_emb = args.unfreeze_word_emb
            self.model = args.model
            self.pretrained_prop_model_path = args.pretrained_prop_model_path
            self.finetune_prop_encoder = args.finetune_prop_encoder
        elif args.procedure == 'train_prop':
            self.word_emb_caps = args.word_emb_caps  # ActivityNetCaptionsDataset() needs it
            self.pretrained_cap_model_path = args.pretrained_cap_model_path
            self.finetune_cap_encoder = args.finetune_cap_encoder
            self.layer_norm = args.layer_norm
            self.anchors_num_audio = args.anchors_num_audio
            self.anchors_num_video = args.anchors_num_video
            self.noobj_coeff = args.noobj_coeff
            self.obj_coeff = args.obj_coeff
            self.train_json_path = args.train_json_path
            self.nms_tiou_thresh = args.nms_tiou_thresh
            self.strides = {}
            self.pad_feats_up_to = {}
            self.kernel_sizes = {}
            if 'audio' in self.modality:
                self.strides['audio'] = args.audio_feature_timespan
                self.pad_feats_up_to['audio'] = args.pad_audio_feats_up_to
                self.conv_layers_audio = args.conv_layers_audio
                self.kernel_sizes['audio'] = args.kernel_sizes_audio
            if 'video' in self.modality:
                self.feature_timespan_in_fps = args.feature_timespan_in_fps
                self.fps_at_extraction = args.fps_at_extraction
                self.strides['video'] = args.feature_timespan_in_fps / args.fps_at_extraction
                self.pad_feats_up_to['video'] = args.pad_video_feats_up_to
                self.conv_layers_video = args.conv_layers_video
                self.kernel_sizes['video'] = args.kernel_sizes_video
        elif args.procedure == 'evaluate':
            self.pretrained_cap_model_path = args.pretrained_cap_model_path
        else:
            raise NotImplementedError


        self.dout_p = args.dout_p
        self.N = args.N
        self.use_linear_embedder = args.use_linear_embedder
        if args.use_linear_embedder:
            self.d_model_video = args.d_model_video
            self.d_model_audio = args.d_model_audio
        else:
            self.d_model_video = self.d_vid
            self.d_model_audio = self.d_aud
        self.H = args.H
        self.d_model = args.d_model
        self.d_model_caps = args.d_model_caps
        if 'video' in self.modality:
            self.d_ff_video = 4*self.d_model_video if args.d_ff_video is None else args.d_ff_video
        if 'audio' in self.modality:
            self.d_ff_audio = 4*self.d_model_audio if args.d_ff_audio is None else args.d_ff_audio
        self.d_ff_caps = 4*self.d_model_caps if args.d_ff_caps is None else args.d_ff_caps
        # training
        self.device_ids = args.device_ids
        self.device = f'cuda:{self.device_ids[0]}'
        self.train_batch_size = args.B * len(self.device_ids)
        self.inference_batch_size = args.inf_B_coeff * self.train_batch_size
        self.epoch_num = args.epoch_num
        self.one_by_one_starts_at = args.one_by_one_starts_at
        self.early_stop_after = args.early_stop_after
        # criterion
        self.smoothing = args.smoothing  # 0 == cross entropy
        self.grad_clip = args.grad_clip
        # optimizer
        self.optimizer = args.optimizer
        if self.optimizer == 'adam':
            self.beta1, self.beta2 = args.betas
            self.eps = args.eps
            self.weight_decay = args.weight_decay
        elif self.optimizer == 'sgd':
            self.momentum = args.momentum
            self.weight_decay = args.weight_decay
        else:
            raise Exception(f'Undefined optimizer: "{self.optimizer}"')
        # lr scheduler
        self.scheduler = args.scheduler
        if self.scheduler == 'constant':
            self.lr = args.lr
            self.weight_decay = args.weight_decay
        elif self.scheduler == 'reduce_on_plateau':
            self.lr = args.lr
            self.lr_reduce_factor = args.lr_reduce_factor
            self.lr_patience = args.lr_patience
        else:
            raise Exception(f'Undefined scheduler: "{self.scheduler}"')
        # evaluation
        self.reference_paths = args.reference_paths
        self.tIoUs = args.tIoUs
        self.max_prop_per_vid = args.max_prop_per_vid
        self.prop_pred_path = args.prop_pred_path
        self.avail_mp4_path = args.avail_mp4_path
        # logging
        self.to_log = args.to_log
        if args.to_log:
            self.log_dir = os.path.join(args.log_dir, args.procedure)
            self.checkpoint_dir = self.log_dir  # the same yes
            # exper_name = self.make_experiment_name()
            exper_name = self.curr_time[2:]
            self.log_path = os.path.join(self.log_dir, exper_name)
            self.model_checkpoint_path = os.path.join(self.checkpoint_dir, exper_name)
        else:
            self.log_dir = None
            self.log_path = None

