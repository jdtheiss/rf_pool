from unittest import TestCase

import _pool

import torch
import numpy as np

class Test_pool(TestCase):
    def test_matmul(self):
        a = torch.rand(1, 4, 4)

        b = _pool.max_pool(a, kernel_size=(2,2))[0]

        d = np.random.rand(4,4)

        e = torch.mm(torch.tensor(d).squeeze().float(), torch.rand(4,4))
        assert isinstance(e, torch.Tensor)
