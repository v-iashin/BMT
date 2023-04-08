
# Dense Video Captioning with Bi-modal Transformer
[Project Page](https://v-iashin.github.io/bmt)
• [ArXiv](https://arxiv.org/abs/2005.08271)
• [BMVC Page](https://www.bmvc2020-conference.com/conference/papers/paper_0111.html)
• [Presentation](https://www.youtube.com/watch?v=C4zYVIqGDVQ) ([Can't watch YouTube? I gotchu!](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/video_0111.mp4) 🤗)
•

This is a PyTorch implementation for our paper: A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer ([BMVC](https://bmvc2020.github.io/) 2020).

- [Dense Video Captioning with Bi-modal Transformer](#dense-video-captioning-with-bi-modal-transformer)
  - [Summary](#summary)
  - [Getting Started](#getting-started)
  - [Train](#train)
  - [Evaluate](#evaluate)
  - [Details on Feature Extraction](#details-on-feature-extraction)
  - [Reproducibility Note](#reproducibility-note)
  - [Comparison with MDVC](#comparison-with-mdvc)
  - [Single Video Prediction](#single-video-prediction)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)
  - [Media Coverage](#media-coverage)

## Summary

Dense video captioning aims to localize and describe important events in untrimmed videos. Existing methods mainly tackle this task by exploiting the visual information alone, while completely neglecting the audio track.

To this end, we present *Bi-modal Transformer with Proposal Generator* (BMT), which efficiently utilizes audio and visual input sequences to select events in a video and, then, use these clips to generate a textual description.

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/bmt/bi_modal_transformer.svg" alt="Bi-Modal Transformer with Proposal Generator" width="900">

Audio and visual features are encoded with [*VGGish*](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish) and [*I3D*](https://github.com/hassony2/kinetics_i3d_pytorch/tree/51240948f9ae92808c390e7217041d6fd89414e9) while caption tokens with [*GloVe*](https://torchtext.readthedocs.io/en/latest/vocab.html#glove). First, VGGish and I3D features are passed through the stack of *N* bi-modal encoder layers where audio and visual sequences are encoded to, what we call, audio-attended visual and video-attended audio features. These features are passed to the bi-modal multi-headed proposal generator, which generates a set of proposals using information from both modalities.

Then, the input features are trimmed according to the proposed segments and encoded in the bi-modal encoder again. The stack of *N* bi-modal decoder layers inputs both: a) GloVe embeddings of the previously generated caption sequence, b) the internal representation from the last layer of the encoder for both modalities. The decoder produces its internal representation which is, then, used in the generator model the distribution over the vocabulary for the caption next word.

<!-- <img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/bmt/proposal_generator_compressed.svg" alt="Bi-Modal Transformer with Proposal Generator" height="400"> -->

## Getting Started

_The code is tested on `Ubuntu 16.04/18.04` with one `NVIDIA GPU 1080Ti/2080Ti`. If you are planning to use it with other software/hardware, you might need to adapt `conda` environment files or even the code._

Clone the repository. Mind the `--recursive` flag to make sure `submodules` are also cloned (evaluation scripts for Python 3 and scripts for feature extraction).
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

- *Train proposal generation module*. You may also download the pre-trained model [best_prop_model.pt](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/best_prop_model.pt) (`md5 hash 5f8b20826b09eadd41b7a5be662c198b`)
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
Check out our script for extraction of I3D and VGGish features from a set of videos: [video_features on GitHub](https://github.com/v-iashin/video_features/tree/662ec51caf591e76724237f0454bdf7735a8dcb1) (make sure to checkout to `662ec51caf591e76724237f0454bdf7735a8dcb1` commit). Also see [#7](https://github.com/v-iashin/MDVC/issues/7) for more details on configuration.


## Reproducibility Note

We would like to note that, despite a fixed random seed, some randomness occurs in our experimentation. Therefore, during the training of the captioning module, one might achieve slightly different results. Specifically, the numbers in your case might differ (higher or lower) from ours or the model will saturate in a different number of epochs. At the same time, we observed quite consistent results when training the proposal generation module with the pre-trained captioning module.

We relate this problem to padding and how it is implemented in PyTorch. (see [PyTorch Reproducibility](https://pytorch.org/docs/1.2.0/notes/randomness.html#pytorch) for details). Also, any suggestions on how to address this issue are greatly appreciated.

## Comparison with MDVC

Comparison between [MDVC](https://arxiv.org/abs/2003.07758) and Bi-modal Transformer (BMT) on ActivityNet Captions validation set captioning ground truth proposals. BMT performs on par while having three times fewer parameters and using only two modalities.
|                                    Model | Params (Mill) | BLEU@3 | BLEU@4 | METEOR |
|-----------------------------------------:|--------------:|-------:|-------:|-------:|
| [MDVC](https://arxiv.org/abs/2003.07758) |           149 |   4.52 |   1.98 |  11.07 |
|                                      BMT |            51 |   4.63 |   1.99 |  10.90 |

## Single Video Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v-iashin/BMT/blob/master/colab_demo_BMT.ipynb)

_The experience with Google Colab is pretty poor. To ensure a more optimal experience, we recommend following the installation guide and setting up the software locally as described below._

Start by extracting audio and visual features from your video using [video_features](https://github.com/v-iashin/video_features/tree/662ec51caf591e76724237f0454bdf7735a8dcb1) repository. This repo is also included in `./submodules/video_features` (commit `662ec51caf591e76724237f0454bdf7735a8dcb1`).

Extract I3D features
```bash
# run this from the video_features folder:
cd ./submodules/video_features
conda deactivate
conda activate i3d
python main.py \
    --feature_type i3d \
    --on_extraction save_numpy \
    --device_ids 0 \
    --extraction_fps 25 \
    --video_paths ../../sample/women_long_jump.mp4 \
    --output_path ../../sample/
```

Extract VGGish features (if `ValueError`, download the vggish model first--see `README.md` in `./submodules/video_features`)
```bash
conda deactivate
conda activate vggish
python main.py \
    --feature_type vggish \
    --on_extraction save_numpy \
    --device_ids 0 \
    --video_paths ../../sample/women_long_jump.mp4 \
    --output_path ../../sample/
```

Run the inference
```bash
# run this from the BMT main folder:
cd ../../
conda deactivate
conda activate bmt
python ./sample/single_video_prediction.py \
    --prop_generator_model_path ./sample/best_prop_model.pt \
    --pretrained_cap_model_path ./sample/best_cap_model.pt \
    --vggish_features_path ./sample/women_long_jump_vggish.npy \
    --rgb_features_path ./sample/women_long_jump_rgb.npy \
    --flow_features_path ./sample/women_long_jump_flow.npy \
    --duration_in_secs 35.155 \
    --device_id 0 \
    --max_prop_per_vid 100 \
    --nms_tiou_thresh 0.4
```

Expected output
```
[
  {'start': 0.1, 'end': 4.9, 'sentence': 'We see a title screen'},
  {'start': 5.0, 'end': 7.9, 'sentence': 'A large group of people are seen standing around a building'},
  {'start': 0.7, 'end': 11.9, 'sentence': 'A man is seen standing in front of a large crowd'},
  {'start': 19.6, 'end': 33.3, 'sentence': 'The woman runs down a track and jumps into a sand pit'},
  {'start': 7.5, 'end': 10.0, 'sentence': 'A large group of people are seen standing around a building'},
  {'start': 0.6, 'end': 35.1, 'sentence': 'A large group of people are seen running down a track while others watch on the sides'},
  {'start': 8.2, 'end': 13.7, 'sentence': 'A man runs down a track'},
  {'start': 0.1, 'end': 2.0, 'sentence': 'We see a title screen'}
]
```

Note that in our research we avoided non-maximum suppression for computational efficiency and to allow the event prediction to be dense. Feel free to play with `--nms_tiou_thresh` parameter: for example, try to make it `0.4` as in the provided example.

The sample video credits: [Women's long jump historical World record in 1978](https://www.youtube.com/watch?v=nynA-Gmh2r8)

If you are having an error
```
RuntimeError: Vector for token b'<something>' has <some-number> dimensions, but previously read vectors
have 300 dimensions.
```
try to remove `*.txt` and `*.txt.pt` from the hidden folder `./.vector_cache/` and check if you
are not running out of disk space (unpacking of `glove.840B.300d.zip` requires extra ~8.5G).
Then run `single_video_prediction.py` again.

## Citation
Our paper was accepted at BMVC 2020. Please, use this bibtex if you would like to cite our work
```
@InProceedings{BMT_Iashin_2020,
  title={A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer},
  author={Iashin, Vladimir and Rahtu, Esa},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2020}
}
```

```
@InProceedings{MDVC_Iashin_2020,
  title = {Multi-Modal Dense Video Captioning},
  author = {Iashin, Vladimir and Rahtu, Esa},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={958--959},
  year = {2020}
}
```

## Acknowledgments
Funding for this research was provided by the Academy of Finland projects 327910 & 324346. The authors acknowledge CSC — IT Center for Science, Finland, for computational resources for our experimentation.

* [Prithviraj](https://github.com/xanthan011) contributed to the [Google Colab demo](https://colab.research.google.com/github/v-iashin/BMT/blob/master/colab_demo_BMT.ipynb)

## Media Coverage
- [Dense Video Captioning Using Pytorch (Towards Data Science)](https://towardsdatascience.com/dense-video-captioning-using-pytorch-392ca0d6971a)
- (in Russian) [Рубрика «Читаем статьи за вас». Сентябрь — октябрь 2020 года (habr.com)](https://habr.com/ru/company/ods/blog/544320/)
