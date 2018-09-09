import random
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
import blosc


class ReplayBuffer(object):
    """
    strategy: circular write
    """

    def __init__(self, size):
        super().__init__()
        self.capacity = size
        self.buf = []
        self.next_row = 0

    def push(self, record):
        if len(self.buf) < self.capacity:
            self.buf.append(record)
        else:
            self.buf[self.next_row] = record
            self.next_row = (self.next_row + 1) % self.capacity

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size):
        if len(self.buf) < batch_size:
            return None
            # raise RuntimeError('Replay Buffer is not buffered enough to sample')
        indexes = np.random.randint(0, len(self.buf), batch_size)

        # batch_list = [self.buf[idx] for idx in indexes]  # [{'s0':Lazy, 's1':Lazy..}, {}, {}]
        batch = []
        for idx in indexes:
            batch.append(self.buf[idx])

        batch = default_collate(batch)
        return batch


class IndexTree(object):
    def __init__(self, n, init_value):
        super().__init__()
        self.n = n
        N = 1
        while N < n:
            N <<= 1
        N -= 1
        self.offset = N
        # self.tree = np.full(self.offset + n + 1, init_value)

        # full binary tree for convenience
        self.tree = np.full(self.offset * 2 + 2, init_value, dtype=np.float32)


class SumTree(IndexTree):
    def __init__(self, n):
        super().__init__(n, init_value=0)

    def update(self, index, new_value):
        assert 1 <= index <= self.n
        assert new_value >= 0
        index += self.offset
        delta = new_value - self.tree[index]

        # 32 log n = 320 << n
        while index > 0:
            self.tree[index] += delta
            index >>= 1

    def sample(self, batch_size):
        ret = []
        probs = []
        for t in self.tree[1] * np.random.random_sample(size=batch_size):
            index = 1
            while index <= self.offset:
                if t <= self.tree[index << 1]:
                    index <<= 1
                else:
                    t -= self.tree[index << 1]
                    index = (index << 1) + 1

            ret.append(index - self.offset)
            probs.append(self.tree[index] / self.tree[1])

        return ret, probs


class MinTree(IndexTree):
    def __init__(self, n):
        # n+1 to prevent oob
        super().__init__(n, init_value=float('inf'))

    def update(self, index, new_value):
        assert 1 <= index <= self.n

        index += self.offset
        self.tree[index] = new_value
        index >>= 1
        while index:
            self.tree[index] = min(self.tree[index * 2], self.tree[index * 2 + 1])
            index >>= 1

    def query_argmin(self):
        index = 1
        while index <= self.offset:
            if self.tree[index] == self.tree[index << 1]:
                index <<= 1
            else:
                index = (index << 1) + 1
        return index - self.offset


class PrioritizedReplayBuffer(object):
    """
    strategy: sample with TD error
    """

    def __init__(self, capacity, alpha=0.6):  # 0.6 for atari
        super().__init__()
        self.capacity = capacity
        self.size = 0
        self.alpha = alpha
        self.buf = [None] * (capacity + 1)

        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)  # to evict minimum priority

        self._max_priority = 1.0

    def push(self, record):
        if self.size < self.capacity:
            self.size += 1
            row = self.size
        else:
            row = self.min_tree.query_argmin()

        self.buf[row] = record
        self.sum_tree.update(row, self._max_priority)
        self.min_tree.update(row, self._max_priority)

    def sample(self, batch_size, beta=1):
        if self.size < batch_size:
            return None

        rows, probs = self.sum_tree.sample(batch_size)
        weights = torch.Tensor(probs)
        weights = weights ** -beta
        weights /= weights.max()

        batch = []
        for idx in rows:
            batch.append(self.buf[idx])

        batch = default_collate(batch)
        return batch, rows, weights

    def update_priority(self, rma, priority: torch.Tensor):
        priority = priority ** self.alpha
        for row, p in zip(rma, priority):
            p = p.item()
            if p > self._max_priority:
                self._max_priority = p
            self.sum_tree.update(row, p)
            self.min_tree.update(row, p)


def test_PER():
    buf = PrioritizedReplayBuffer(capacity=5)

    buf.push({'a': 'A', 'b': 'B1'})
    buf.push({'a': 'A', 'b': 'B2'})

    # print(buf.sample(2))
    # print(buf.sample(1))
    # print(buf.sample(1))
    # print(buf.sample(1))
    # print('------')

    buf.push({'a': 'A', 'b': 'B3'})
    buf.push({'a': 'A', 'b': 'B4'})

    # print(buf.sample(2))
    print(buf.sample(2))
    print(buf.sample(2))
    print('------')

    buf.update_priority([2, 3], torch.Tensor([1.0, 2.0]))
    print(buf.sample(4))
    print(buf.sample(4))
    print('------')
