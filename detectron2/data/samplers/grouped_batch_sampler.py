# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size):
        """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]  # yield a copy of the list
                del group_buffer[:]

    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")


class TripleBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain two elements from one group and one element from the another group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, datasets, batch_size=3, same_cate=True, cate_num=None):
        """
        Args:
            sampler (Sampler): Samplers yielding pair id, not each image data
            pair_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        import time
        start_time = time.time()
        self.sampler = sampler
        cur_pair_id = 1
        self.pair_to_indices = {cur_pair_id: []}
        self.gt_classes = torch.zeros(len(datasets))
        for idx, data in enumerate(datasets):
            if idx % 1000 == 0:
                print('processing {}th...elapsed time: {:.2f}s'.format(idx, time.time() - start_time))
            data_inst = data['instances']
            pair_id = data_inst.pair_id[0].item()
            if same_cate:
                gt_class = data_inst.gt_classes[data_inst.style > 0]
                gt_class = gt_class[torch.randperm(len(gt_class))[0]] if len(gt_class) > 0 else torch.tensor(-1)
                self.gt_classes[idx] = gt_class
            if pair_id == cur_pair_id:
                self.pair_to_indices[pair_id].append(idx)
            elif pair_id > cur_pair_id:
                self.pair_to_indices[cur_pair_id] = np.asarray(self.pair_to_indices[cur_pair_id])
                np.random.shuffle(self.pair_to_indices[cur_pair_id])
                self.pair_to_indices[pair_id] = []
                cur_pair_id = pair_id
            else:
                'can not make pair dict!'
                continue

        self.batch_size = batch_size
        self.used_pair_indices_count = {k: 0 for k in self.pair_to_indices.keys()}
        self.cate_num = cate_num if same_cate else 1
        self.buffer_per_cate = {k: [] for k in np.arange(self.cate_num)}

    def __iter__(self):
        for idx in self.sampler:     # yield positive pair id, negative pair id
            cate = (self.gt_classes[self.pair_to_indices[idx][self.used_pair_indices_count[idx]]]).item() if self.cate_num > 1 else 0
            if cate < 0:
                continue
            to_add = min(self.batch_size - len(self.buffer_per_cate[cate]), 2)
            if self.used_pair_indices_count[idx] + to_add >= len(self.pair_to_indices[idx]):
                self.used_pair_indices_count[idx] = 0
                continue
            self.buffer_per_cate[cate].extend(self.pair_to_indices[idx][self.used_pair_indices_count[idx]:self.used_pair_indices_count[idx]+to_add])
            self.used_pair_indices_count[idx] += to_add
            if len(self.buffer_per_cate[cate]) == self.batch_size:
                yield self.buffer_per_cate[cate]
                self.buffer_per_cate[cate] = []

    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")