import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class CustomSampler(Sampler):
    def __init__(self, dset, do_hard_negative: bool = False, shuffle: bool = True, seed: int = 42, oversample=False):
        self.dset = dset
        self.shuffle = shuffle
        self.seed = seed
        self.ind_cls = 0
        self.epoch = 0
        self.do_hard_negative = do_hard_negative

        self.cls = {}
        for ind, data in enumerate(self.dset.data):
            label = data["category_id"]
            if label in self.cls:
                self.cls[label].append(ind)
            else:
                self.cls[label] = [ind]
        """
        for i in range(max(self.cls)):
            if i not in self.cls:
                self.cls[i] = []
        """
        if self.do_hard_negative:
            self.cls_class = defaultdict(list)
            with open("./food_id_to_category_id.json") as f:
                self.food_to_category = json.load(f)
            self.food_to_category = {int(k): int(v) for k, v in self.food_to_category.items()}

            for key in self.cls.keys():
                self.cls_class[self.food_to_category[key]].extend(self.cls[key])

        self.cls_inds = [0 for _ in range(max(self.cls) + 1)]
        self.max_n_sample = max([len(samples) for _, samples in self.cls.items()])
        for label, samples in self.cls.items():
            if oversample:
                pad_size = self.max_n_sample - len(samples)
                self.cls[label].extend(samples[:pad_size])

        self.cls_indicies = list(self.cls.keys())
        self.cls_matcher = {idx: idx for idx in range(max(self.cls) + 1)}

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)

            for label, indicies in self.cls.items():
                inds = np.array(indicies)
                self.cls[label] = inds[torch.randperm(len(indicies), generator=g).tolist()].tolist()

            if self.epoch != 0:
                cls_inds = np.arange(0, len(self.cls_inds))
                cls_inds = cls_inds[torch.randperm(len(self.cls_inds), generator=g).tolist()]
                self.cls_matcher = {idx: cls_inds[idx] for idx in range(len(self.cls_inds))}
            self.epoch += 1

        indicies = []
        self.ind_step = self.cls_indicies[0]
        for _ in range(len(self.dset)):
            cls_ind = self.cls_indicies[self.ind_step % len(self.cls)]
            cls_ind = self.cls_matcher[cls_ind]
            if len(self.cls[cls_ind]):
                smp_ind = self.cls_inds[cls_ind] % len(self.cls[cls_ind])
                index = self.cls[cls_ind][smp_ind]

                self.ind_step += 1
                self.cls_inds[cls_ind] += 1
                indicies.append(index)

                if self.do_hard_negative:
                    cls_index = random.choice(self.cls[cls_ind])
                    while index == cls_index:
                        cls_index = random.choice(self.cls[cls_ind])
                    indicies.append(cls_index)

                    cls_idx = random.choice(self.cls_class[self.food_to_category[cls_ind]])

                    while index == cls_index or cls_ind == self.dset.data[cls_idx]["category_id"]:
                        cls_idx = random.choice(self.cls_class[self.food_to_category[cls_ind]])

                    indicies.append(cls_idx)

        assert len(indicies) == (len(self.dset) * 3 if self.do_hard_negative else len(self.dset))
        assert len(self.cls) == len(set([self.dset.data[idx]["category_id"] for idx in indicies]))
        return iter(indicies)

    def __len__(self):
        return len(self.dset)


class CategoryContrastiveSampler(Sampler):
    def __init__(self, dset, shuffle: bool = True, seed: int = 42, oversample=False):
        self.dset = dset
        self.shuffle = shuffle
        self.seed = seed
        self.ind_cls = 0
        self.epoch = 0

        with open("data/category_dict.json", encoding="euc-kr") as f:
            category_json = json.load(f)
        category_to_id = {k: idx for idx, k in enumerate(category_json.keys())}

        with open("data/food_to_category.json") as f:
            food_to_category = json.load(f)

        with open("data/labels.json") as f:
            labels_json = json.load(f)
        food_labels = {item["id"]: item["label"] for item in labels_json["categories"]}

        self.cls = {}
        for ind, data in enumerate(self.dset.data):
            label = category_to_id[food_to_category[food_labels[data["category_id"]]]]
            if label in self.cls:
                self.cls[label].append(ind)
            else:
                self.cls[label] = [ind]

        self.cls_inds = [0 for _ in range(len(self.cls))]
        self.max_n_sample = max([len(samples) for _, samples in self.cls.items()])
        for label, samples in self.cls.items():
            if oversample:
                pad_size = self.max_n_sample - len(samples)
                self.cls[label].extend(samples[:pad_size])

        self.cls_indicies = list(self.cls.keys())
        self.cls_matcher = {idx: idx for idx in range(len(self.cls_inds))}

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)

            for label, indicies in self.cls.items():
                inds = np.array(indicies)
                self.cls[label] = inds[torch.randperm(len(indicies), generator=g).tolist()].tolist()

            if self.epoch != 0:
                cls_inds = np.arange(0, len(self.cls_inds))
                cls_inds = cls_inds[torch.randperm(len(self.cls_inds), generator=g).tolist()]
                self.cls_matcher = {idx: cls_inds[idx] for idx in range(len(self.cls_inds))}
            self.epoch += 1

        indicies = []
        self.ind_step = self.cls_indicies[0]
        for _ in range(len(self.dset)):
            cls_ind = self.cls_indicies[self.ind_step % len(self.cls)]
            cls_ind = self.cls_matcher[cls_ind]
            smp_ind = self.cls_inds[cls_ind] % len(self.cls[cls_ind])
            index = self.cls[cls_ind][smp_ind]

            self.ind_step += 1
            self.cls_inds[cls_ind] += 1
            indicies.append(index)

        assert len(indicies) == len(self.dset)
        # assert len(self.cls) == len(set([self.dset.data[idx]["category_id"] for idx in indicies]))
        return iter(indicies)

    def __len__(self):
        return len(self.dset)
