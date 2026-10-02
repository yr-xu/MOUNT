from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

def _read_pkl_rows(pkl_path):
    obj = pd.read_pickle(pkl_path)

    if isinstance(obj, pd.DataFrame):
        rows = list(obj.itertuples(index=False, name=None))
    else:
        rows = list(obj)

    return rows


def _normalize_candidate_field(x):

    if isinstance(x, str):
        s = x.strip()
        try:
            x = ast.literal_eval(s)
        except Exception:
            if s.lstrip("-").isdigit():
                x = int(s)
            else:
                raise ValueError(f"Cannot parse candidate field string: {x}")

    # scalar int
    if isinstance(x, (int, np.integer)):
        x = int(x)
        return [] if x == -1 else [x]

    # iterable candidates
    if isinstance(x, (list, tuple, set, np.ndarray)):
        out = []
        for c in list(x):
            if isinstance(c, str) and c.strip().lstrip("-").isdigit():
                c = int(c.strip())
            elif isinstance(c, (int, np.integer)):
                c = int(c)
            else:
                # skip weird values
                continue
            if c != -1:
                out.append(c)
        return out

    raise ValueError(f"Unsupported candidate field type: {type(x)} -> {x}")


def _sample_key_from_row(row):
    return tuple(row[:-1])


class EPICDOMAIN_PLL(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 eval=False,
                 cfg=None,
                 cfg_flow=None,
                 sample_dur=10,
                 far_ood=False,
                 datapath='/scratch/project_2000948/data/haod/EPIC-KITCHENS/',
                 cand_pkl='',
                 num_class=0):
        assert cfg is not None and cfg_flow is not None, "cfg and cfg_flow must be provided"
        assert num_class > 0, "num_class must be > 0"

        self.base_path = datapath
        self.split = split
        self.interval = 9
        self.sample_dur = sample_dur
        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.num_class = num_class
        self.far_ood = far_ood

        if split == 'train' and not eval:
            self.pipeline = Compose(cfg.data.train.pipeline)
            self.pipeline_flow = Compose(cfg_flow.data.train.pipeline)
            self.meta_cfg = cfg.data.train
            self.meta_cfg_flow = cfg_flow.data.train
        else:
            self.pipeline = Compose(cfg.data.val.pipeline)
            self.pipeline_flow = Compose(cfg_flow.data.val.pipeline)
            self.meta_cfg = cfg.data.val
            self.meta_cfg_flow = cfg_flow.data.val

        if cand_pkl == '' or cand_pkl is None:
            if far_ood:
                self.cand_pkls = [f"splits/D3_{s}.pkl" for s in ['train', 'val', 'test']]
                self._merge_mode_single_stream = True
            else:
                self.cand_pkls = [f"splits/D3_{split}_near_ood.pkl"]
                self._merge_mode_single_stream = False
        elif isinstance(cand_pkl, (list, tuple)):
            self.cand_pkls = list(cand_pkl)
            self._merge_mode_single_stream = False
        else:
            self.cand_pkls = [cand_pkl]
            self._merge_mode_single_stream = False

        if far_ood and (cand_pkl == '' or cand_pkl is None):
            merged_rows = []
            for fp in self.cand_pkls:
                rows = _read_pkl_rows(fp)
                for r in rows:
                    merged_rows.append(tuple(r))
            self.samples = [tuple(r[:-1]) for r in merged_rows]
            self.cand_lists_per_file = []
            one_stream_cands = [_normalize_candidate_field(r[-1]) for r in merged_rows]
            self.cand_lists_per_file.append(one_stream_cands)  # K=1
        else:
            # support one or multiple candidate pkl files aligned by sample key
            rows0 = [tuple(r) for r in _read_pkl_rows(self.cand_pkls[0])]
            self.samples = [tuple(r[:-1]) for r in rows0]  # store all fields except candidate label field

            cands0 = [_normalize_candidate_field(r[-1]) for r in rows0]
            self.cand_lists_per_file = [cands0]

            # align remaining pkls by key = row[:-1]
            for fp in self.cand_pkls[1:]:
                rows_i = [tuple(r) for r in _read_pkl_rows(fp)]
                key2cand = {_sample_key_from_row(r): _normalize_candidate_field(r[-1]) for r in rows_i}
                aligned = [key2cand.get(tuple(s), []) for s in self.samples]
                self.cand_lists_per_file.append(aligned)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        vid_id = sample[0]
        start_f = int(sample[1])
        end_f = int(sample[2])

        if vid_id in ['P22_01', 'P22_02', 'P22_03', 'P22_04']:
            video_path = self.base_path + 'rgb/test/D3/' + vid_id
            flow_path = self.base_path + 'flow/test/D3/' + vid_id
        else:
            video_path = self.base_path + 'rgb/train/D3/' + vid_id
            flow_path = self.base_path + 'flow/train/D3/' + vid_id

        filename_tmpl = self.meta_cfg.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.meta_cfg.get('modality', 'RGB')
        start_index = self.meta_cfg.get('start_index', start_f)

        data = dict(
            frame_dir=video_path,
            total_frames=int(end_f - start_f),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality
        )
        data = self.pipeline(data)

        # Flow
        filename_tmpl_flow = self.meta_cfg_flow.get('filename_tmpl', 'frame_{:010}.jpg')
        modality_flow = self.meta_cfg_flow.get('modality', 'Flow')
        start_index_flow = self.meta_cfg_flow.get('start_index', int(np.ceil(start_f / 2)))

        flow = dict(
            frame_dir=flow_path,
            total_frames=int((end_f - start_f) / 2),
            label=-1,
            start_index=start_index_flow,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow
        )
        flow = self.pipeline_flow(flow)

        # Build candidate masks [K, C]
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
        
class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', eval=False, modality='rgb', cfg=None, cfg_flow=None, sample_dur=10, far_ood=False, datapath='/scratch/project_2000948/data/haod/EPIC-KITCHENS/', labeled_only=False):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur

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

        data1 = []
        if far_ood:
            splits = ['train', 'val', 'test']
            for spl in splits:
                train_file_name = "splits/D3_" + spl + ".pkl"
                train_file = pd.read_pickle(train_file_name)
                for line in train_file:
                    data1.append((line[0], line[1], line[2], line[3], line[4], line[5]))
        else:
            train_file_name = "splits/D3_" + split + "_near_ood.pkl"
            train_file = pd.read_pickle(train_file_name)
            for line in train_file:
                data1.append((line[0], line[1], line[2], line[3], line[4], line[5]))

        if labeled_only:
            filtered = []
            for row in data1:
                try:
                    y = int(row[-1])
                except Exception:
                    continue
                if y != -1:
                    filtered.append(row)
            data1 = filtered

        self.samples = data1
        self.cfg = cfg
        self.cfg_flow = cfg_flow

    def __getitem__(self, index):
        if self.samples[index][0] in ['P22_01', 'P22_02', 'P22_03', 'P22_04']:
            video_path = self.base_path +'rgb/test/D3/'+self.samples[index][0]
            flow_path = self.base_path + 'flow/test/D3/'+self.samples[index][0]
        else:
            video_path = self.base_path +'rgb/train/D3/'+self.samples[index][0]
            flow_path = self.base_path + 'flow/train/D3/'+self.samples[index][0]
        
        filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
        data = dict(
            frame_dir=video_path,
            total_frames=int(self.samples[index][2] - self.samples[index][1]),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data = self.pipeline(data)

        filename_tmpl_flow = self.cfg_flow.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality_flow = self.cfg_flow.data.train.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.train.get('start_index', int(np.ceil(self.samples[index][1] / 2)))
        flow = dict(
            frame_dir=flow_path,
            total_frames=int((self.samples[index][2] - self.samples[index][1])/2),
            label=-1,
            start_index=start_index_flow,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow)
        flow = self.pipeline_flow(flow)

        label1 = int(self.samples[index][-1])

        return data, flow, label1


    def __len__(self):
        return len(self.samples)

