
# Dense Video Captioning with Bi-modal Transformer
[Project Page](https://v-iashin.github.io/bmt) • [Paper](https://arxiv.org/abs/2005.08271)

This is a PyTorch implementation for our paper: A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer.

## Summary

Dense video captioning aims to localize and describe important events in untrimmed videos. Existing methods mainly tackle this task by exploiting the visual information alone, while completely neglecting the audio track. 

To this end, we present *Bi-modal Transformer with Proposal Generator* (BMT), which efficiently utilizes audio and visual input sequences to select events in a video and, then, use these clips to generate a textual description.

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/bmt/bi_modal_transformer.svg" alt="Bi-Modal Transformer with Proposal Generator" width="900">

Audio and visual features are encoded with [*VGGish*](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish) and [*I3D*](https://github.com/hassony2/kinetics_i3d_pytorch/tree/51240948f9ae92808c390e7217041d6fd89414e9) while caption tokens with [*GloVe*](https://torchtext.readthedocs.io/en/latest/vocab.html#glove). First, VGGish and I3D features are passed through the stack of *N* bi-modal encoder layers where audio and visual sequences are encoded to, what we call, audio-attended visual and video-attended audio features. These features are passed to the bi-modal multi-headed proposal generator, which generates a set of proposals using information from both modalities. 

Then, the input features are trimmed according to the proposed segments and encoded in the bi-modal encoder again. The stack of *N* bi-modal decoder layers inputs both: a) GloVe embeddings of the previously generated caption sequence, b) the internal representation from the last layer of the encoder for both modalities. The decoder produces its internal representation which is, then, used in the generator model the distribution over the vocabulary for the caption next word.

<!-- <img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/bmt/proposal_generator_compressed.svg" alt="Bi-Modal Transformer with Proposal Generator" height="400"> -->

## Getting Started
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
# install spacy language model. Make sure you activated the conda environment
python -m spacy download en
```

## Train

We train our model in two staged: training of the captioning module on ground truth proposals and training of the proposal generator using the pre-trained encoder from the captioning module.

- *Train the captioning module*. You may also download the pre-trained model [best_cap_model.pt](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/best_cap_model.pt) (`md5 hash 7b4d48cd77ec49a027a4a1abc6867ee7`).
```bash
python main.py \
    --procedure train_cap \
    --B 32
```

- *Train proposal generation module*. If you need a pre-trained proposal generation model, please let me know in Issues.
```bash
python main.py \
    --procedure train_prop \
    --pretrained_cap_model_path /your_exp_path/best_cap_model.pt \
    --B 16
```

## Evaluate

Since a part of videos in ActivityNet Captions became unavailable over the time, we could only obtain ~91 % of videos in the dataset (see `./data/available_mp4.txt` for ids). To this end, we evaluate the performance of our model against ~91 % of the validation videos. We provide the validation sets without such videos in `./data/val_*_no_missings.json`. Please see `Experiments` and `Supplementary Material` sections for details and performance of other models on the same validation sets.

- *Ground truth proposals*. The performance of the captioning module on ground truth segments might be obtained from the file with pre-trained captioning module. You may also want to use the [official evaluation script](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) with `./data/val_*_no_missings.json` as references (`-r` argument).
```python
import torch
cap_model_cpt = torch.load('./path_to_pre_trained_model/best_cap_model.pt', map_location='cpu')
print(cap_model_cpt['val_1_metrics'])
print(cap_model_cpt['val_2_metrics'])
# To obtain the final results, average values in both dicts
```

- *Learned proposals*. Create a file with captions for every proposal provided in `--prop_pred_path` using the captioning model specified in `--pretrained_cap_model_path`. The script will automatically evaluate it againts both ground truth validation sets. Alternatively, use the predictions `prop_results_val_1_e17_maxprop100.json` in `./results` and [official evaluation script](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) with `./data/val_*_no_missings.json` as references (`-r` argument).
```bash
python main.py \
    --procedure evaluate \
    --pretrained_cap_model_path /path_to_best_cap_model.pt \
    --prop_pred_path /path_to_generated_json_file \
    --device_ids 0
```


## Details on Feature Extraction
Check out our script for extraction of the I3D features from a set of videos: [i3d_features on GitHub](https://github.com/v-iashin/i3d_features). Also see [#7](https://github.com/v-iashin/MDVC/issues/7) for more details on configuration.


## Reproducibility Note

We would like to note that, despite a fixed random seed, some randomness occurs in our experimentation. Therefore, during the training of the captioning module, one might achieve slightly different results. Specifically, the numbers in your case might differ (higher or lower) from ours or the model will saturate in a different number of epochs. At the same time, we observed quite consistent results when training the proposal generation module with the pre-trained captioning module. 

We relate this problem to padding and how it is implemented in PyTorch. (see [PyTorch Reproducibility](https://pytorch.org/docs/1.2.0/notes/randomness.html#pytorch) for details). Also, any suggestions on how to address this issue are greatly appreciated.

## Comparison with MDVC

Comparison between [MDVC](https://arxiv.org/abs/2003.07758) and Bi-modal Transformer (BMT) on ActivityNet Captions validation set captioning ground truth proposals. BMT performs on par while having three times fewer parameters and using only two modalities.
|                                    Model | Params (Mill) | BLEU@3 | BLEU@4 | METEOR |
|-----------------------------------------:|--------------:|-------:|-------:|-------:|
| [MDVC](https://arxiv.org/abs/2003.07758) |           149 |   4.52 |   1.98 |  11.07 |
|                                      BMT |            51 |   4.63 |   1.99 |  10.90 |

## Citation
Please, use this bibtex if you would like to cite our work
```
@misc{BMT_Iashin_2020,
  title={A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer},
  author={Vladimir Iashin and Esa Rahtu},
  year={2020},
  eprint={2005.08271},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
```
@InProceedings{MDVC_Iashin_2020,
  author = {Iashin, Vladimir and Rahtu, Esa},
  title = {Multi-modal Dense Video Captioning},
  booktitle = {Workshop on Multimodal Learning (CVPR Workshop)},
  year = {2020}
}
```
