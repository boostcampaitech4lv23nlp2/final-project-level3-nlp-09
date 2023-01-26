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
        self.food_to_category = {}
        self.cls_class = {}

        self.cls = self.get_cls(self.dset.data)

        if self.do_hard_negative:
            self.food_to_category = self.get_food_to_category()
            self.cls_class = self.get_cls_class(cls=self.cls, food_to_category=self.food_to_category)

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

            self.cls = self.shuffle_cls(g, cls=self.cls)
            self.cls_matcher = self.shuffle_cls_matcher(g, self.cls_inds, self.cls_matcher)

        self.ind_step = self.cls_indicies[0]
        indicies = self.get_indicies(
            self.dset,
            self.cls_indicies,
            self.ind_step,
            self.cls,
            self.cls_inds,
            self.food_to_category,
            self.cls_class,
            self.cls_matcher,
            self.do_hard_negative,
        )

        assert len(indicies) == (len(self.dset) * 3 if self.do_hard_negative else len(self.dset))
        assert len(self.cls) == len(set([self.dset.data[idx]["category_id"] for idx in indicies]))
        return iter(indicies)

    def __len__(self):
        return len(self.dset)

    def get_cls(self, data):
        cls = {}

        for ind, data in enumerate(data):
            label = data["category_id"]
            if label in cls:
                cls[label].append(ind)
            else:
                cls[label] = [ind]
        """
        for i in range(max(self.cls)):
            if i not in self.cls:
                self.cls[i] = []
        """
        return cls

    def get_food_to_category(self):
        with open("./food_id_to_category_id.json") as f:
            food_to_category = json.load(f)
            food_to_category = {int(k): int(v) for k, v in food_to_category.items()}

        return food_to_category

    def get_cls_class(self, cls, food_to_category):
        cls_class = defaultdict(list)

        for key in cls.keys():
            cls_class[food_to_category[key]].extend(cls[key])

        return cls_class

    def shuffle_cls(self, g, cls):
        new_cls = {key: cls[key] for key in cls}

        for label, indicies in cls.items():
            inds = np.array(indicies)
            new_cls[label] = inds[torch.randperm(len(indicies), generator=g).tolist()].tolist()

        return new_cls

    def shuffle_cls_matcher(self, g, cls_inds, cls_matcher):
        new_cls_matcher = {key: cls_matcher[key] for key in cls_matcher}

        if self.epoch != 0:
            cls_inds = np.arange(0, len(cls_inds))
            cls_inds = cls_inds[torch.randperm(len(cls_inds), generator=g).tolist()]
            new_cls_matcher = {idx: cls_inds[idx] for idx in range(len(cls_inds))}
        self.epoch += 1

        return new_cls_matcher

    def get_indicies(
        self, dset, cls_indicies, ind_step, cls, cls_inds, food_to_category, cls_class, cls_matcher, do_hard_negative
    ):
        indicies = []

        for _ in range(len(dset)):
            cls_ind = cls_indicies[ind_step % len(cls)]
            cls_ind = cls_matcher[cls_ind]
            if len(cls[cls_ind]):
                smp_ind = cls_inds[cls_ind] % len(cls[cls_ind])
                index = cls[cls_ind][smp_ind]

                ind_step += 1
                cls_inds[cls_ind] += 1
                indicies.append(index)

                if do_hard_negative:
                    indicies = self.get_hard_negative(indicies, cls, cls_ind, index, cls_class, food_to_category, dset)

        return indicies

    def get_hard_negative(self, indicies, cls, cls_ind, index, cls_class, food_to_category, dset):
        new_indicies = indicies[:]
        cls_index = random.choice(cls[cls_ind])
        while index == cls_index:
            cls_index = random.choice(cls[cls_ind])
        new_indicies.append(cls_index)

        cls_idx = random.choice(cls_class[food_to_category[cls_ind]])

        while index == cls_index or cls_ind == dset.data[cls_idx]["category_id"]:
            cls_idx = random.choice(cls_class[food_to_category[cls_ind]])

        new_indicies.append(cls_idx)

        return new_indicies
