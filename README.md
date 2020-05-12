This is a PyTorch implementation for our paper: A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer.

# Dense Video Captioning with Bi-modal Transformer

![Bi-Modal Transformer with Proposal Generator](https://github.com/v-iashin/v-iashin.github.io/raw/master/images/bmt/bi_modal_transformer_compressed.svg)

## How to Replicate

### Set up the Environment
Clone the repository. Mind the `--recursive` flag to make sure `submodules` are also cloned (evaluation scripts for Python 3).
```bash
git clone --recursive https://github.com/v-iashin/BMT.git
```

Download features (I3D and VGGish) and word embeddings (GloVe). The script will download them (~10 GB) and unpack into `./data` and `./.vector_cache` folders. *Make sure to run it while being in BMT folder*
```bash
bash ./download_data.sh
```

Set up a `conda` environment
```bash
conda env create -f ./conda_env.yml
conda activate bmt
```

### Train

**Train the captioning module**
```bash
python main.py --procedure train_cap --device_ids 1 --B 32
```
or download the pre-trained model [best_cap_model.pt](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/best_cap_model.pt) (`md5 hash 7b4d48cd77ec49a027a4a1abc6867ee7`).

**Train proposal generation module**
```bash
python main.py \
    --procedure train_prop \
    --pretrained_cap_model_path ./log/train_cap/0512080239/best_cap_model.pt \
    --device_ids 2 \
    --B 16
```
If you need a pre-trained proposal generation model, please let me know in Issues.

### Evaluate

Since a part of videos became unavailable over the time, we could only obtain ~91 % of videos in the dataset (see `./data/available_mp4.txt` for ids). To this end, we evaluate the performance of our model against ~91 % of the validation videos. We provide validation sets without such videos in `./data/val_*_no_missings.json`. Please see `Experiments` and `Supplementary Material` sections for details and performance of other models on the same validation sets.

**Ground truth proposals**. The performance of the captioning module on ground truth segments might be obtained from the file with pre-trained captioning module.
```python
import torch
cap_model_cpt = torch.load('./path_to_pre_trained_model/best_cap_model.pt', map_location='cpu')
print(cap_model_cpt['val_1_metrics'])
print(cap_model_cpt['val_2_metrics'])
# To obtain the final results, average values in both dicts
```
Or use the [official evaluation script](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) with `./data/val_*_no_missings.json` as references (`-r` argument).

**Learned proposals**. Create a file with captions for every proposal provided in `--prop_pred_path` using the captioning model specified in `--pretrained_cap_model_path`. The script will automatically evaluate it againts both ground truth validation sets
```bash
python main.py \
    --procedure evaluate \
    --pretrained_cap_model_path ./path_to_best_cap_model.pt \
    --prop_pred_path ./path_to_generated_json_file \
    --device_ids 0
```
Alternatively, use the predictions `prop_results_val_1_e17_maxprop100.json` in `./results` and [official evaluation script](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) with `./data/val_*_no_missings.json` as references (`-r` argument).


### Reproducibility Note

We would like to note that, despite a fixed random seed, some randomness occurs in our experimentation. Therefore, during the training of the captioning module, one might achieve slightly different results. Specifically, the numbers in your case might differ (higher or lower) from ours or the model will saturate in a different number of epochs. At the same time, we observed quite consistent results when training the proposal generation module with the pre-trained captioning module. 

We relate this problem to padding and how it is implemented in PyTorch. (see [PyTorch Reproducibility](https://pytorch.org/docs/1.2.0/notes/randomness.html#pytorch) for details). Also, any suggestions on how to address it are greatly appreciated.

## Comparison with MDVC

|                                    Model | Params (Mill) | BLEU@3 | BLEU@4 | METEOR |
|-----------------------------------------:|--------------:|-------:|-------:|-------:|
| [MDVC](https://arxiv.org/abs/2003.07758) |           149 |   4.52 |   1.98 |  11.07 |
|                                      BMT |            51 |   4.63 |   1.99 |  10.90 |
> Comparison between [MDVC](https://arxiv.org/abs/2003.07758) and Bi-modal Transformer (BMT) on ActivityNet Captions validation set captioning ground truth proposals. BMT performs on par while having three times fewer parameters and using only two modalities.

TODO
- [ ] citation