from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
import torch
from tests.unit_tests.test_utilities import Utils
import numpy as np

def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4,2)    # tensor parallel: 4, pipeline parallel: 2
    vocab_parallel_logits = torch.range(0,7).repeat(16,4).cuda()
    # 16-times repeat on 1st axis, 4-times repeat on 2nd axis,
    # 1: [0-7, 0-7, 0-7, 0-7] 32
    # 2: [0-7, 0-7, 0-7, 0-7] 32
    # ...
    # 16: [0-7, 0-7, 0-7, 0-7] 32
    # => vocab_parallel_logits.shape == 16x32(8*4)

    # Vocab split
    # device#0 = 0-31
    # device#1 = 32-63
    # device#2 = 64-95
    # device#3 = 96-127

    target = torch.arange(0,32,2).cuda()    # [0, 2, 4, ..., 30] (16)
    # target will be used only on device #0 (0-31), rests are masked

    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor([10.2309,  8.2309,  6.2309,  4.2309, 10.2309,  8.2309,  6.2309,  4.2309,
                                    10.2309,  8.2309,  6.2309,  4.2309, 10.2309,  8.2309,  6.2309,  4.2309]).cuda()
    assert(torch.equal(torch.round(expected_output), torch.round(output)))
    Utils.destroy_model_parallel()