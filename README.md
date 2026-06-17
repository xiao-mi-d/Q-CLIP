<div align="center">


### **Unleashing the Power of Vision-Language Models for Video Quality Assessment through Unified Cross-Modal Adaptation**

Yachun Mi<sup>1,2</sup>, Yu Li<sup>1</sup>, Yanting Li<sup>1</sup>, Chen Hui<sup>3</sup>, Tong Zhang<sup>1</sup>, Zhixuan Li<sup>2</sup>, Chenyue Song<sup>1</sup>, Wei Yang Bryan Lim<sup>2</sup>, Shaohui Liu<sup>1</sup>

<sup>1</sup> School of Computer Science and Technology, Harbin Institute of Technology, Harbin, China  
<sup>2</sup> College of Computing and Data Science, Nanyang Technological University, Singapore  
<sup>3</sup> School of Artificial Intelligence, Nanjing University of Information Science and Technology, Nanjing, China

</div>

Q-CLIP is a fully CVLM-based VQA framework built on Perception Encoder (PE). The model freezes the PE backbone and trains lightweight Shared Cross-Modal Adapters (SCMA), learnable five-level quality prompts, and a final quality regressor.

## Structure

```text
.
├── model/
│   ├── qclip.py                    # Q-CLIP model
│   ├── core/vision_encoder/         # PE vision/text encoder code used by Q-CLIP
│   └── apps/pe/                     # retained PE reference/demo files
├── datasets/
│   └── vqa_datasets.py              # VQA datasets
├── options/
│   ├── pretrain/                    # LSVQ pretraining config
│   └── finetune/                    # small-dataset fine-tuning configs
├── data_labels/                     # MOS annotation examples
└── train.py
```

## Installation

```bash
conda create --name qclip python=3.12
conda activate qclip

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124

conda install ffmpeg -c conda-forge
pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Download [PE-Core-L14-336.pt](https://huggingface.co/facebook/PE-Core-L14-336/blob/main/PE-Core-L14-336.pt) from the official PE release and place it at:

```text
pretrained_weights/PE-Core-L14-336.pt
```


## Training

Set `DATA_ROOT` to the directory containing the VQA datasets before training:

```bash
export DATA_ROOT=/path/to/VQADatasets
```

Pretrain on LSVQ:

```bash
python train.py -o options/pretrain/qclip_lsvq.yml --device cuda:0
```

Alternatively, set `DATA_ROOT` for a single command:

```bash
DATA_ROOT=/path/to/VQADatasets python train.py --opt options/pretrain/qclip_lsvq.yml --device cuda:1
```

Fine-tune on KoNViD-1k:

```bash
python train.py -o options/finetune/konvid.yml --device cuda:0
```

The provided fine-tuning configs report a single 80/20 random split result. To report the average result over 10 random splits, manually run fine-tuning with 10 different split seeds and average the evaluation metrics.

The fine-tuning configs default to loading:

```text
pretrained_weights/qclip_lsvq_val-ltest_best.pth
```

Adjust `load_path` in the YAML if your pretrained checkpoint has a different name.

## Acknowledgement

This project is built on top of Meta's [Perception Models](https://github.com/facebookresearch/perception_models) repository. We thank the authors for releasing the Perception Encoder (PE) code and pretrained models. The PE encoder implementation under `model/core/vision_encoder` is adapted from the official repository and follows `LICENSE.PE`.

## Citation

```bibtex
@inproceedings{2026qclip,
title={Q-{CLIP}: Unleashing the Power of Vision-Language Models for Video Quality Assessment through Unified Cross-Modal Adaptation},
author={Yachun Mi and Yu Li and Yanting Li and Chen Hui and Tong Zhang and Zhixuan Li and Chenyue Song and Wei Yang Bryan Lim and Shaohui Liu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=zmAMq09bFs}
}

@article{bolya2025PerceptionEncoder,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  journal={arXiv:2504.13181},
  year={2025}
}
```
