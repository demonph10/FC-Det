import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import h5py
import json


class FakingRecipe_Dataset(Dataset):
    def __init__(self, vid_path, dataset):
        self.dataset = dataset
        if dataset == 'fakesv':
            self.data_all = pd.read_json('./features/fakesv/metainfo.json', orient='records', dtype=False, lines=True)
        elif dataset == 'fakett':
            self.data_all = pd.read_json('./fea/fakett/metainfo.json', orient='records', lines=True,
                                         dtype={'video_id': str})

        self.vid = []
        with open(vid_path, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
        self.data.reset_index(inplace=True)

        self.audio_fea_path = f'./features/{dataset}/preprocess_audio'
        self.visual_fea_path = f'./features/{dataset}/preprocess_image'
        self.raw_visual_fea_path = f'./fea/{dataset}/preprocess_visual'
        self.patch_fea_path = f'./features/{dataset}/preprocess_image_patch'
        self.text_fea_path = f'./features/{dataset}/preprocess_text'
        self.token_fea_path = f'./features/{dataset}/preprocess_text_token'
        self.reason_fea_path = f'./features/{dataset}/preprocess_reason_token'

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        a_fea_path = os.path.join(self.audio_fea_path, vid + '.pkl')
        audio_features = torch.load(open(a_fea_path, 'rb'))  # 1*768

        visual_features = pickle.load(open(os.path.join(self.visual_fea_path, vid + '.pkl'), 'rb')).cpu()
        visual_features = torch.tensor(visual_features, dtype=torch.float)

        patch_features = pickle.load(open(os.path.join(self.patch_fea_path, vid + '.pkl'), 'rb')).cpu()
        patch_features = torch.tensor(patch_features, dtype=torch.float)

        text_features = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb')).cpu()
        text_features = torch.tensor(text_features, dtype=torch.float)

        token_features = pickle.load(open(os.path.join(self.token_fea_path, vid + '.pkl'), 'rb')).cpu()
        token_features = torch.tensor(token_features, dtype=torch.float)

        reason_features = pickle.load(open(os.path.join(self.reason_fea_path, vid + '.pkl'), 'rb')).cpu()
        reason_features = torch.tensor(reason_features, dtype=torch.float)

        return {
            'vid': vid,
            'label': label,
            'audio_features': audio_features,
            'visual_features': visual_features,
            'patch_features': patch_features,
            'text_features': text_features,
            'token_features': token_features,
            'reason_features': reason_features,
        }


def pad_visual(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat(
                (video, torch.zeros([seq_len - ori_len, video.shape[1], video.shape[2]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def pad_frame_by_seg(seq_len, lst, seg):
    result = []
    seg_indicators = []
    sampled_seg = []
    for i in range(len(lst)):
        video = lst[i]
        v_sampled_seg = []
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        seg_video = seg[i]
        seg_len = len(seg_video)
        if seg_len >= seq_len:
            gap = seg_len // seq_len
            seg_video = seg_video[::gap][:seq_len]
            sample_index = []
            sample_seg_indicator = []
            for j in range(len(seg_video)):
                v_sampled_seg.append(seg_video[j])
                if seg_video[j][0] == seg_video[j][1]:
                    sample_index.append(seg_video[j][0])
                else:
                    sample_index.append(np.random.randint(seg_video[j][0], seg_video[j][1]))
                sample_seg_indicator.append(j)
            video = video[sample_index]
            mask = sample_seg_indicator
        else:
            if ori_len < seq_len:
                video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)

                mask = []
                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    mask.extend([j] * (seg_video[j][1] - seg_video[j][0] + 1))
                mask.extend([-1] * (seq_len - len(mask)))

            else:

                sample_index = []
                sample_seg_indicator = []
                seg_len = [(x[1] - x[0]) + 1 for x in seg_video]
                sample_ratio = [seg_len[i] / sum(seg_len) for i in range(len(seg_len))]
                sample_len = [seq_len * sample_ratio[i] for i in range(len(seg_len))]
                sample_per_seg = [int(x) + 1 if x < 1 else int(x) for x in sample_len]

                sample_per_seg = [x if x <= seg_len[i] else seg_len[i] for i, x in enumerate(sample_per_seg)]
                additional_sample = sum(sample_per_seg) - seq_len
                if additional_sample > 0:
                    idx = 0
                    while additional_sample > 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if sample_per_seg[idx] > 1:
                            sample_per_seg[idx] = sample_per_seg[idx] - 1
                            additional_sample = additional_sample - 1
                        idx += 1

                elif additional_sample < 0:
                    idx = 0
                    while additional_sample < 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if seg_len[idx] - sample_per_seg[idx] >= 1:
                            sample_per_seg[idx] = sample_per_seg[idx] + 1
                            additional_sample = additional_sample + 1
                        idx += 1

                for seg_idx in range(len(sample_per_seg)):
                    sample_seg_indicator.extend([seg_idx] * sample_per_seg[seg_idx])

                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    if sample_per_seg[j] == seg_len[j]:
                        sample_index.extend(np.arange(seg_video[j][0], seg_video[j][1] + 1))

                    else:
                        sample_index.extend(
                            np.sort(np.random.randint(seg_video[j][0], seg_video[j][1] + 1, sample_per_seg[j])))

                sample_index = np.array(sample_index)
                sample_index = np.sort(sample_index)
                video = video[sample_index]
                batch_sample_seg_indicator = np.array(sample_seg_indicator)
                mask = batch_sample_seg_indicator
                v_sampled_seg.sort(key=lambda x: x[0])

        result.append(video)
        mask = torch.IntTensor(mask)
        sampled_seg.append(v_sampled_seg)
        seg_indicators.append(mask)
    return torch.stack(result), torch.stack(seg_indicators), sampled_seg


def pad_segment(seg_lst, target_len):
    for sl_idx in range(len(seg_lst)):
        for s_idx in range(len(seg_lst[sl_idx])):
            seg_lst[sl_idx][s_idx] = torch.tensor(seg_lst[sl_idx][s_idx])
        if len(seg_lst[sl_idx]) < target_len:
            seg_lst[sl_idx].extend([torch.tensor([-1, -1])] * (target_len - len(seg_lst[sl_idx])))
        else:
            seg_lst[sl_idx] = seg_lst[sl_idx][:target_len]
        seg_lst[sl_idx] = torch.stack(seg_lst[sl_idx])

    return torch.stack(seg_lst)


def pad_unnatural_phrase(phrase_lst, target_len):
    for pl_idx in range(len(phrase_lst)):
        if len(phrase_lst[pl_idx]) < target_len:
            phrase_lst[pl_idx] = torch.cat((phrase_lst[pl_idx], torch.zeros(
                [target_len - len(phrase_lst[pl_idx]), phrase_lst[pl_idx].shape[1]], dtype=torch.long)), dim=0)
        else:
            phrase_lst[pl_idx] = phrase_lst[pl_idx][:target_len]
    return torch.stack(phrase_lst)


def collate_fn_FakeingRecipe(batch):
    num_visual_frames = 20

    vid = [item['vid'] for item in batch]
    label = torch.stack([item['label'] for item in batch])

    audio_features = [item['audio_features'] for item in batch]
    audio_features = torch.cat(audio_features, dim=0)

    visual_features = [item['visual_features'] for item in batch]
    visual_features, _ = pad_visual(num_visual_frames, visual_features)

    patch_features = [item['patch_features'] for item in batch]
    patch_features, _ = pad_visual(num_visual_frames, patch_features)

    text_features = [item['text_features'] for item in batch]
    text_features = torch.stack(text_features, dim=0)

    token_features = [item['token_features'] for item in batch]
    token_features = [
        x if x.shape[1] == 77 else torch.cat((x, torch.zeros([1, 77 - x.shape[1], x.shape[2]], dtype=torch.float)),
                                             dim=1) for x in token_features]  # batch*77*512
    token_features = torch.cat(token_features, dim=0)

    reason_features = [item['reason_features'] for item in batch]
    reason_features = [
        x if x.shape[1] == 77 else torch.cat((x, torch.zeros([1, 77 - x.shape[1], x.shape[2]], dtype=torch.float)),
                                             dim=1) for x in reason_features]  # batch*77*512
    reason_features = torch.cat(reason_features, dim=0)

    return {
        'vid': vid,
        'label': label,
        'audio_features': audio_features,
        'visual_features': visual_features,
        'patch_features': patch_features,
        'text_features': text_features,
        'token_features': token_features,
        'reason_features': reason_features,
    }
