from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import imageio.v3 as iio
import re
import ast
import torch

_CAND_RE = re.compile(r"\[[\d\s,]+\]\s*$")

def load_txt_file_candidates(file_path):
    paths, cand_lists = [], []

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = _CAND_RE.search(line)

            cand_str = m.group(0)                    # like "[1, 7, 14, 16]"
            prefix = line[:m.start()].strip()        # everything before the candidate list

            # Remove optional y_hat if present
            if " " in prefix:
                left, last = prefix.rsplit(" ", 1)
                if last.lstrip("-").isdigit():
                    path = left
                else:
                    path = prefix
            else:
                path = prefix

            S = ast.literal_eval(cand_str)
            if not isinstance(S, (list, tuple)):
                raise ValueError(f"Candidate set is not a list/tuple: {line}")

            paths.append(path)
            cand_lists.append(list(S))

    return tuple(paths), tuple(cand_lists)
    
    

def load_txt_file(file_path, drop_unlabeled=False):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [line.strip().split() for line in lines if line.strip()]
    paths, labels = zip(*data)

    if drop_unlabeled:
        keep = [i for i, y in enumerate(labels) if int(y) != -1]
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]

    return tuple(paths), tuple(labels)

def load_txt_file_kinetics(file_path, drop_unlabeled=False):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [line.strip().rsplit(' ', 1) for line in lines if line.strip()]
    paths, labels = zip(*data)

    if drop_unlabeled:
        keep = [i for i, y in enumerate(labels) if int(y) != -1]
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]

    return tuple(paths), tuple(labels)

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', eval=False, cfg=None, cfg_flow=None, sample_dur=10, dataset='HMDB', near_ood=True, far_ood=False, ood_dataset='UCF', datapath='', labeled_only=False):
        self.base_path = datapath
        self.split = split
        self.interval = 9
        self.sample_dur = sample_dur

        # build the data pipeline
        if split == 'train' and not eval:
            train_pipeline = cfg.data.train.pipeline
            self.pipeline = Compose(train_pipeline)
            train_pipeline_flow = cfg_flow.data.train.pipeline
            self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.pipeline = Compose(val_pipeline)
            val_pipeline_flow = cfg_flow.data.val.pipeline
            self.pipeline_flow = Compose(val_pipeline_flow)
        if far_ood:
            train_file_name = "splits/" + "ID_" + dataset + "_OOD_" + ood_dataset + ".txt"
        elif near_ood:
            train_file_name = "splits/" + dataset + "_" + split + "_near_ood.txt"
        else:
            train_file_name = "splits/" + dataset + "_" + split + ".txt"

        if dataset == "Kinetics" or ood_dataset == "Kinetics":
            self.samples, self.labels = load_txt_file_kinetics(train_file_name, drop_unlabeled=labeled_only)
        else:
            self.samples, self.labels = load_txt_file(train_file_name, drop_unlabeled=labeled_only)

        self.cfg = cfg
        self.cfg_flow = cfg_flow
        if far_ood:
            self.dataset = ood_dataset
        else:
            self.dataset = dataset

    def __getitem__(self, index):
        video_path = ''
        video_file = self.base_path + 'video/' + self.samples[index]
        vid = iio.imread(video_file, plugin="pyav")

        frame_num = vid.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index,
            video=vid,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, frame_inds = self.pipeline(data)

        if self.dataset == "Kinetics":
            v_name_pure = self.samples[index][:-4]
        elif self.dataset == "HMDB":
            v_name = self.samples[index]
            start_index = v_name.index('/') + 1
            end_index = v_name.index('.')
            v_name_pure = v_name[start_index:end_index]
        elif self.dataset == "UCF":
            v_name = self.samples[index]
            end_index = v_name.index('.')
            v_name_pure = v_name[:end_index]

        video_file_x = self.base_path + 'flow/' + v_name_pure + '_flow_x.mp4'
        video_file_y = self.base_path + 'flow/' + v_name_pure + '_flow_y.mp4'

        vid_x = iio.imread(video_file_x, plugin="pyav")
        vid_y = iio.imread(video_file_y, plugin="pyav")

        frame_num = vid_x.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
        modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)
        flow = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index_flow,
            video=vid_x,
            video_y=vid_y,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow)
        flow, frame_inds_flow = self.pipeline_flow(flow)

        label1 = int(self.labels[index])
        
        return data, flow, label1

    def __len__(self):
        return len(self.samples)
        
class EPICDOMAIN_PLL(torch.utils.data.Dataset):
    """
    Returns: (rgb_data, flow_data, cand_masks)
      cand_masks: FloatTensor [K, num_class] with 1s on candidate labels for each txt file.
    """
    def __init__(self, split='train', eval=False, cfg=None, cfg_flow=None,
                 sample_dur=10, dataset='HMDB', near_ood=True, far_ood=False,
                 ood_dataset='UCF', datapath='', cand_txt='', num_class=0):

        assert cand_txt != "", "cand_txt must be provided for EPICDOMAIN_PLL"
        assert num_class > 0, "num_class must be provided for EPICDOMAIN_PLL"

        self.base_path = datapath
        self.split = split
        self.interval = 9
        self.sample_dur = sample_dur
        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.num_class = num_class

        if split == 'train' and not eval:
            self.pipeline = Compose(cfg.data.train.pipeline)
            self.pipeline_flow = Compose(cfg_flow.data.train.pipeline)
        else:
            self.pipeline = Compose(cfg.data.val.pipeline)
            self.pipeline_flow = Compose(cfg_flow.data.val.pipeline)

        # Allow cand_txt to be str or list[str]
        if isinstance(cand_txt, (list, tuple)):
            self.cand_txts = list(cand_txt)
        else:
            self.cand_txts = [cand_txt]

        # Load first file to define sample order
        samples0, cands0 = load_txt_file_candidates(self.cand_txts[0])
        self.samples = list(samples0)

        # Store aligned candidate lists per file: list of length K, each is list aligned to self.samples
        self.cand_lists_per_file = []
        self.cand_lists_per_file.append(list(cands0))

        # Load remaining files
        for fp in self.cand_txts[1:]:
            samples_i, cands_i = load_txt_file_candidates(fp)
            path2cand = {p: c for p, c in zip(samples_i, cands_i)}
            aligned = [path2cand.get(p, []) for p in self.samples] 
            self.cand_lists_per_file.append(aligned)

        self.dataset = ood_dataset if far_ood else dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path = ''
        video_file = self.base_path + 'video/' + self.samples[index]
        vid = iio.imread(video_file, plugin="pyav")

        frame_num = vid.shape[0]
        start_frame = 0
        end_frame = frame_num - 1

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index,
            video=vid,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, _ = self.pipeline(data)

        if self.dataset == "Kinetics":
            v_name_pure = self.samples[index][:-4]
        elif self.dataset == "HMDB":
            v_name = self.samples[index]
            s2 = v_name.index('/') + 1
            e2 = v_name.index('.')
            v_name_pure = v_name[s2:e2]
        elif self.dataset == "UCF":
            v_name = self.samples[index]
            e2 = v_name.index('.')
            v_name_pure = v_name[:e2]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        video_file_x = self.base_path + 'flow/' + v_name_pure + '_flow_x.mp4'
        video_file_y = self.base_path + 'flow/' + v_name_pure + '_flow_y.mp4'
        vid_x = iio.imread(video_file_x, plugin="pyav")
        vid_y = iio.imread(video_file_y, plugin="pyav")

        frame_num = vid_x.shape[0]
        start_frame = 0
        end_frame = frame_num - 1

        filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
        modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)
        flow = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index_flow,
            video=vid_x,
            video_y=vid_y,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow)
        flow, _ = self.pipeline_flow(flow)

        # Build K candidate masks
        masks = []
        K = len(self.cand_lists_per_file)
        for k in range(K):
            S = self.cand_lists_per_file[k][index]
            cand_mask = torch.zeros(self.num_class, dtype=torch.float32)
            for c in S:
                c = int(c)
                if 0 <= c < self.num_class:
                    cand_mask[c] = 1.0
            masks.append(cand_mask)

        cand_masks = torch.stack(masks, dim=0)  # [K, C]
        return data, flow, cand_masks

        
    def __len__(self):
        return len(self.samples)