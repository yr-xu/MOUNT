# Code for Multi-Coverage Guidance with Conformal Calibration
This repository contains the code used to implement the proposed methods and reproduce the simulations presented in ‘Multi-Coverage Guidance with Conformal Calibration for Semi-Supervised Multimodal OOD Detection‘.

Our implementation and dataset preparation workflow are adapted from [DPU](https://github.com/lili0415/DPU-OOD-Detection) and [MultiOOD](https://github.com/donghao51/MultiOOD).

### Datasets Preparation

Download and extract the datasets below, then organize their files as shown.

#### HMDB51

Download the [videos](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and [optical flow](https://huggingface.co/datasets/hdong51/MultiOOD/blob/main/hmdb51_flow_mp4.tar.gz).

<details>
<summary>Directory structure</summary>

```text
HMDB51/
├── video/
│   ├── catch/
│   │   └── *.avi
│   └── ...
└── flow/
    ├── *_flow_x.mp4
    └── *_flow_y.mp4
```

</details>

#### UCF101

Download the [videos](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) and [optical flow](https://huggingface.co/datasets/hdong51/MultiOOD/blob/main/ucf101_flow_mp4.tar.gz).

<details>
<summary>Directory structure</summary>

```text
UCF101/
├── video/
│   └── *.avi
└── flow/
    ├── *_flow_x.mp4
    └── *_flow_y.mp4
```

</details>

#### EPIC-Kitchens

Download the videos and optical flow using:

```bash
bash utils/download_epic_script.sh
```

Download and extract the [audio archive](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/blob/main/EPIC-KITCHENS-audio.zip). Place each audio file beside its corresponding video-frame directory.

<details>
<summary>Directory structure</summary>

```text
EPIC-KITCHENS/
├── rgb/
│   ├── train/D3/
│   │   ├── P22_05.wav
│   │   └── P22_05/
│   │       └── frame_0000000000.jpg, ...
│   └── test/D3/
│       ├── P22_01.wav
│       └── P22_01/
│           └── frame_0000000000.jpg, ...
└── flow/
    ├── train/D3/P22_05/
    │   └── frame_0000000000.jpg, ...
    └── test/D3/P22_01/
        └── frame_0000000000.jpg, ...
```

</details>

#### HAC

Download the [video, audio, and optical flow archives](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/tree/main).

<details>
<summary>Directory structure</summary>

```text
HAC/
├── human/
│   ├── videos/
│   ├── flow/
│   └── audio/
├── animal/
│   ├── videos/
│   ├── flow/
│   └── audio/
└── cartoon/
    ├── videos/
    ├── flow/
    └── audio/
```

</details>

#### Kinetics-600

Download the video archives:

```bash
wget -i utils/filtered_k600_train_path.txt
```

Extract the video archives, then generate audio files:

```bash
python utils/generate_audio_files.py
```

Download all `kinetics600_flow_mp4_part_*` files from [MultiOOD](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main). In the directory containing these files, combine and extract them:

```bash
cat kinetics600_flow_mp4_part_* > kinetics600_flow_mp4.tar.gz
tar -zxvf kinetics600_flow_mp4.tar.gz
```

<details>
<summary>Directory structure</summary>

```text
Kinetics-600/
├── video/
│   ├── acting in play/
│   │   ├── *.mp4
│   │   └── *.wav
│   └── ...
└── flow/
    ├── acting in play/
    │   ├── *_flow_x.mp4
    │   └── *_flow_y.mp4
    └── ...
```

</details>

### Dataset Splits

Semi-supervised multimodal Near-OOD and Far-OOD split files are stored in `HMDB-rgb-flow/splits/` for HMDB51, UCF101, HAC, and Kinetics-600, and `EPIC-rgb-flow/splits/` for EPIC-Kitchens.

- `[dataset]_3_pred.txt`: unlabeled samples.
- `[dataset]_fold*_cal.txt`: held-out labeled calibration samples.
- `[dataset]_fold*_train.txt`: labeled training samples.

Here, `[dataset]` denotes the dataset name and `*` denotes the fold index.

## Code

The code was tested with `Python 3.10.19`, `torch 1.11.0+cu113`, and `NVIDIA A100 80GB`. Additional dependencies are listed in `requirements.txt`.

### Pretrained Weights

Download the following checkpoints and place each in both directories:

- `HMDB-rgb-flow/pretrained_models/`
- `EPIC-rgb-flow/pretrained_models/`

1. [SlowFast](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) for RGB.
2. [SlowOnly](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) for optical flow.
3. [Audio model](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), renamed to `vggsound_avgpool.pth.tar`.

### Evaluation Notes

Replace dataset and checkpoint path placeholders with your local paths.

Select the scoring function using `--postprocessor`. For ASH or ReAct, first export evaluation files with `--use_ash` or `--use_react`, respectively, then evaluate with `--postprocessor ash` or `--postprocessor react`.

### Semi-supervised Multimodal Near-OOD Detection

Dataset headings indicate the numbers of ID/OOD classes.

#### HMDB51 40/11

<details>
<summary>Training and evaluation</summary>

```bash
cd HMDB-rgb-flow/
```

**1. Train a baseline model for each of the three folds using its labeled training split.**

```bash
python 1_train.py --near_ood --dataset HMDB --lr 0.0001 --seed 0 --bsz 64 --num_workers 16 --start_epoch 10 --use_single_pred --use_irm --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/HMDB51dataset/'
```

**2. Calibrate the fold models.**

```bash
python 2_cali.py --checkpoints '/path/to/checkpoints' --cal_splits fold0_cal fold1_cal fold2_cal --datapath '/path/to/HMDB51dataset/' --dataset HMDB --alphas 0.15 0.1 0.05 0.025 0.01 --near_ood --out_prefix cp_kfold
```

**3. Generate candidate-label sets for unlabeled samples using the fold checkpoints.**

```bash
python 3_pred.py --checkpoints '/path/to/checkpoints' --qhat_files cp_HMDB_alpha0.15.npy cp_HMDB_alpha0.1.npy cp_HMDB_alpha0.05.npy cp_HMDB_alpha0.025.npy cp_HMDB_alpha0.01.npy --datapath '/path/to/HMDB51dataset/' --dataset HMDB --near_ood --batch_size 8 --out_all_prefix splits/HMDB_train_candidates_near_ood
```

**4. Jointly retrain on labeled and unlabeled data.**

```bash
python 4_train.py --near_ood --dataset HMDB --lr 0.0001 --seed 0 --bsz 64 --num_workers 16 --start_epoch 10 --use_single_pred --use_irm --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/HMDB51dataset/' --use_pll --pll_cand_wts 0.47 0.35 0.18 --pll_weight 0.3
```

**5. Export evaluation files.**

```bash
python test_video_flow.py --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/checkpoint'
```

**6. Evaluate OOD detection.**

```bash
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --path 'HMDB-rgb-flow/' --resume_file '/path/to/checkpoint'
```

</details>


#### UCF101 80/21

<details>
<summary>Training and evaluation</summary>

```bash
cd HMDB-rgb-flow/
```

**1. Train a baseline model for each of the three folds using its labeled training split.**

```bash
python 1_train.py --near_ood --dataset UCF --lr 0.0001 --seed 0 --bsz 64 --num_workers 16 --start_epoch 10 --use_single_pred --use_irm --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/UCF101dataset/'
```

**2. Calibrate the fold models.**

```bash
python 2_cali.py --checkpoints '/path/to/checkpoints' --cal_splits fold0_cal fold1_cal fold2_cal --datapath '/path/to/UCF101dataset/' --dataset UCF --alphas 0.15 0.1 0.05 0.025 0.01 --near_ood --out_prefix cp_kfold
```

**3. Generate candidate-label sets for unlabeled samples using the fold checkpoints.**

```bash
python 3_pred.py --checkpoints '/path/to/checkpoints' --qhat_files cp_kfold_UCF_alpha0.15.npy cp_kfold_UCF_alpha0.1.npy cp_kfold_UCF_alpha0.05.npy cp_kfold_UCF_alpha0.025.npy cp_kfold_UCF_alpha0.01.npy --datapath '/path/to/UCF101dataset/' --dataset UCF --near_ood --batch_size 8 --out_all_prefix splits/UCF_train_candidates_near_ood
```

**4. Jointly retrain on labeled and unlabeled data.**

```bash
python 4_train.py --near_ood --dataset UCF --lr 0.0001 --seed 0 --bsz 64 --num_workers 16 --start_epoch 10 --use_single_pred --use_irm --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/UCF101dataset/' --use_pll --pll_cand_wts 0.4 0.4 0.2 --pll_weight 0.3
```

**5. Export evaluation files.**

```bash
python test_video_flow.py --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/checkpoint'
```

**6. Evaluate OOD detection.**

```bash
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'UCF' --path 'HMDB-rgb-flow/' --resume_file '/path/to/checkpoint'
```

</details>


#### EPIC-Kitchens 6/2

<details>
<summary>Training and evaluation</summary>

```bash
cd EPIC-rgb-flow/
```

**1. Train a baseline model for each of the three folds using its labeled training split.**

```bash
python 1_train_epic.py --dataset EPIC --lr 0.0001 --seed 0 --bsz 16 --num_workers 12 --start_epoch 10 --use_single_pred --use_irm --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/EPICdataset/'
```

**2. Calibrate the fold models.**

```bash
python 2_cali_epic.py --checkpoints '/path/to/checkpoints' --cal_splits fold0_cal fold1_cal fold2_cal --datapath '/path/to/EPICdataset/' --alphas 0.15 0.1 0.05 0.025 0.01 --out_prefix cp_kfold
```

**3. Generate candidate-label sets for unlabeled samples using the fold checkpoints.**

```bash
python 3_pred_epic.py --checkpoints '/path/to/checkpoints' --qhat_files cp_kfold_alpha0.15.npy cp_kfold_alpha0.1.npy cp_kfold_alpha0.05.npy cp_kfold_alpha0.025.npy cp_kfold_alpha0.01.npy --datapath '/path/to/EPICdataset/' --batch_size 8 --out_all_prefix splits/D3_train_candidates_near_ood
```

**4. Jointly retrain on labeled and unlabeled data.**

```bash
python 4_train_epic.py --dataset EPIC --lr 0.0001 --seed 0 --bsz 16 --num_workers 12 --start_epoch 10 --use_single_pred --use_irm --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/EPICdataset/' --use_pll --pll_cand_wts 0.42 0.40 0.18 --pll_cand_pkls '/path/to/D3_train_candidates_near_ood pkl files' --pll_weight 0.3
```

**5. Export evaluation files.**

```bash
python test_video_flow_epic.py --bsz 16 --num_workers 2 --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/checkpoint'
```

**6. Evaluate OOD detection.**

```bash
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'EPIC' --path 'EPIC-rgb-flow/' --resume_file '/path/to/checkpoint'
```

</details>

#### Kinetics-600 183/46

<details>
<summary>Training and evaluation</summary>

```bash
cd HMDB-rgb-flow/
```

**1. Train a baseline model for each of the three folds using its labeled training split.**

```bash
python 1_train.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 64 --num_workers 4 --start_epoch 10 --use_single_pred --use_irm --use_dynamic_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics600dataset/'
```

**2. Calibrate the fold models.**

```bash
python 2_cali.py --checkpoints '/path/to/checkpoints' --cal_splits fold0_cal fold1_cal fold2_cal --datapath '/path/to/Kinetics600dataset/' --dataset Kinetics --alphas 0.15 0.1 0.05 0.025 0.01 --near_ood --out_prefix cp_kfold
```

**3. Generate candidate-label sets for unlabeled samples using the fold checkpoints.**

```bash
python 3_pred.py --checkpoints '/path/to/checkpoints' --qhat_files cp_kfold_Kinetics_alpha0.15.npy cp_kfold_Kinetics_alpha0.1.npy cp_kfold_Kinetics_alpha0.05.npy cp_kfold_Kinetics_alpha0.025.npy cp_kfold_Kinetics_alpha0.01.npy --datapath '/path/to/Kinetics600dataset/' --dataset Kinetics --near_ood --batch_size 8 --out_all_prefix splits/Kinetics_train_candidates_near_ood
```

**4. Jointly retrain on labeled and unlabeled data.**

```bash
python 4_train_kinetics.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 64 --num_workers 4 --start_epoch 10 --use_single_pred --use_irm --use_dynamic_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics600dataset/' --use_pll --pll_weight 0.3 --pll_Kcap 3 --pll_mass_ratio 10 --pll_conf_thr 0.7 --pll_margin_thr 0.15 --pll_gamma 1.0 --pll_ramp_epochs 5
```

**5. Export evaluation files.**

```bash
python test_video_flow.py --bsz 16 --num_workers 2 --near_ood --dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/checkpoint' --datapath '/path/to/Kinetics600dataset/'
```

**6. Evaluate OOD detection.**

```bash
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'Kinetics' --path 'HMDB-rgb-flow/' --resume_file '/path/to/checkpoint'
```

</details>

### Semi-supervised Multimodal Far-OOD Detection

Use the checkpoint trained for the selected ID dataset. Change `--ood_dataset` to the OOD dataset being evaluated.

#### HMDB51 as ID

<details>
<summary>Evaluation</summary>

```bash
cd HMDB-rgb-flow/
```

**Export ID evaluation files.**

```bash
python test_video_flow.py --bsz 16 --num_workers 2 --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDBcheckpoint'
```

**Export OOD evaluation files for UCF101, HAC, or Kinetics-600.** The example below uses UCF101.

```bash
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDBcheckpoint'
```

**Export EPIC-Kitchens evaluation files using the EPIC script.**

```bash
cd EPIC-rgb-flow/
python test_video_flow_epic.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDBcheckpoint'
```

**Evaluate OOD detection.** Set `--ood_dataset` to `UCF`, `HAC`, `Kinetics`, or `EPIC`.

```bash
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/' --resume_file '/path/to/HMDBcheckpoint'
```

</details>

#### Kinetics-600 as ID

<details>
<summary>Evaluation</summary>

```bash
cd HMDB-rgb-flow/
```

**Export ID evaluation files.**

```bash
python test_video_flow.py --bsz 16 --num_workers 2 --dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics600checkpoint'
```

**Export OOD evaluation files for HMDB51, HAC, or UCF101.** The example below uses HMDB51.

```bash
python test_video_flow.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics600checkpoint'
```

**Export EPIC-Kitchens evaluation files using the EPIC script.**

```bash
cd EPIC-rgb-flow/
python test_video_flow_epic.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics600checkpoint'
```

**Evaluate OOD detection.** Set `--ood_dataset` to `HMDB`, `HAC`, `UCF`, or `EPIC`.

```bash
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'Kinetics' --ood_dataset 'HMDB' --path 'HMDB-rgb-flow/' --resume_file '/path/to/Kinetics600checkpoint'
```

</details>


## Acknowledgement

We thank the authors of [DPU](https://github.com/lili0415/DPU-OOD-Detection) and [MultiOOD](https://github.com/donghao51/MultiOOD) for making their implementations and dataset preparation resources available. Portions of our code and documentation are adapted from these projects.


