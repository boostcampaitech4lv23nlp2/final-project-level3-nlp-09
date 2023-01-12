import math
import torch
import numpy as np

from functools import reduce
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


class ContrastiveSampler(Sampler):

    def __init__(self, dset, shuffle: bool = True, seed: int = 42, oversample=False):
        self.dset = dset
        self.shuffle = shuffle
        self.seed = seed
        self.ind_cls = 0
        self.epoch = 0

        self.cls = {}
        for ind, data in enumerate(self.dset.samples):
            label = data[-1]
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
        assert len(self.cls) == len(set([self.dset.samples[idx][-1] for idx in indicies]))
        return iter(indicies)

    def __len__(self):
        return len(self.dset)


class DistContrastiveSampler(DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas: int = None,
                 rank: int = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False,
                 oversample: bool = True):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.cls = {}
        for ind, data in enumerate(self.dataset.samples):
            label = data[-1]
            if label in self.cls:
                self.cls[label].append(ind)
            else:
                self.cls[label] = [ind]

        self.cls_inds = [0 for _ in range(len(self.cls))]
        self.max_n_sample = max([len(samples) for _, samples in self.cls.items()])
        for label, samples in self.cls.items():
            if not oversample:
                pad_size = self.max_n_sample - len(samples)
                self.total_size += pad_size
        self.reset_num_samples()

        self.cls_indicies = list(self.cls.keys())
        self.cls_matcher = {idx: idx for idx in range(len(self.cls_inds))}

    def reset_num_samples(self):
        if self.drop_last and self.total_size % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (self.total_size - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(self.total_size / self.num_replicas)  # type: ignore
        self.total_size = self.num_replicas * self.num_samples

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            for label, indicies in self.cls.items():
                inds = np.array(indicies)
                self.cls[label] = inds[torch.randperm(len(indicies), generator=g).tolist()].tolist()

            if self.epoch != 0:
                cls_inds = np.arange(0, len(self.cls_inds))
                cls_inds = cls_inds[torch.randperm(len(self.cls_inds), generator=g).tolist()]
                self.cls_matcher = {idx: cls_inds[idx] for idx in range(len(self.cls_inds))}

        replica_idx = 0
        cnt_samples = 0
        indicies = [[] for _ in range(self.num_replicas)]
        while cnt_samples < self.total_size:
            for cls_idx in range(len(self.cls)):
                target_cls = self.cls_matcher[cls_idx]
                cur_sample_idx = self.cls_inds[target_cls] % len(self.cls[target_cls])

                self.cls_inds[target_cls] = cur_sample_idx + 1
                sample = self.cls[target_cls][cur_sample_idx]
                indicies[replica_idx].append(sample)
                cnt_samples += 1
                replica_idx += 1
                replica_idx %= self.num_replicas

                if cnt_samples >= self.total_size:
                    break

        assert sum([len(inds) for inds in indicies]) == self.total_size
        assert len(self.cls) == len(
            set(reduce(lambda x, y: x + y, [[self.dataset.samples[idx][-1] for idx in inds] for inds in indicies])))

        # subsample
        indicies = indicies[self.rank]
        assert len(indicies) == self.num_samples
        return iter(indicies)


class DistValidationSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas: int = None, rank: int = None, shuffle: bool = True):
        super(DistValidationSampler, self).__init__(dataset, num_replicas, rank, shuffle, drop_last=False)
        self.dataset = dataset

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class ImbalanceSampler:

    def __init__(self, args):
        self.args = args
        self.under_thr = args.data.imbalance_sampler.undersample.threshold
        self.undersampler = RandomUnderSampler(random_state=args.seed)
        self.oversampler = RandomOverSampler(random_state=args.seed)
        self.oversampled = False
        self.undersampled = False
        self.do_sample = True if args.data.imbalance_sampler.undersample.use or args.data.imbalance_sampler.oversample else False

    def __call__(self, dataset):
        if self.do_sample:
            samples = np.asarray(dataset.samples)
            data = samples[:, 0].reshape(-1, 1)
            labels = samples[:, 1]
            self.set_params(labels)
            if self.args.data.imbalance_sampler.undersample.use:
                data, labels = self.undersample(data, labels)
                self.undersampled = True
            if self.args.data.imbalance_sampler.oversample:
                data, labels = self.oversample(data, labels)
                self.oversampled = True

            dataset.samples = [(str(x[0]), int(y)) for x, y in zip(data, labels)]

    def set_params(self, labels):
        if self.under_thr is not None:
            majority_ids = self.get_majority_threshold(labels)
            self.undersampler.set_params(sampling_strategy=majority_ids)

    def get_majority_threshold(self, labels):
        counter = Counter(labels)
        majority_classes = {k: self.under_thr for k, c in counter.items() if c > self.under_thr}

        return majority_classes

    def oversample(self, data, labels):
        print(f'Before  oversampling ==> {len(data)}')
        data, labels = self.oversampler.fit_resample(data, labels)
        print(f'After   oversampling ==> {len(data)}')

        return data, labels

    def undersample(self, data, labels):
        print(f'Before  undersampling ==> {len(data)}')
        data, labels = self.undersampler.fit_resample(data, labels)
        print(f'After   undersampling ==> {len(data)}')
        return data, labels
