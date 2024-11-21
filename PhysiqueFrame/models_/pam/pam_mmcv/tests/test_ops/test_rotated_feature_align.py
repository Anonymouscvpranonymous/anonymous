# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from pam_mmcv.ops import rotated_feature_align
from pam_mmcv.utils import IS_CUDA_AVAILABLE


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'cpu',
        marks=pytest.mark.skipif(
            torch.__version__ == 'parrots', reason='requires PyTorch support'))
])
def test_rotated_feature_align(device):
    feature = torch.tensor([[[[1.2924, -0.2172, -0.5222, 0.1172],
                              [0.9144, 1.2248, 1.3115, -0.9690],
                              [-0.8949, -1.1797, -0.9093, -0.3961],
                              [-0.4586, 0.5062, -0.7947, -0.7397]],
                             [[-1.0943, -0.7495, 1.3461, -1.1652],
                              [0.2034, 0.6763, -1.2357, 0.5231],
                              [-1.0062, 1.2592, 1.4225, -0.3951],
                              [-0.1242, -1.6240, 0.1932, 2.7181]],
                             [[-1.6271, -1.0276, 0.0578, -0.2997],
                              [-0.9684, -1.6946, -1.3188, -1.1938],
                              [-1.6744, -0.8917, -0.6556,
                               1.0073], [-0.1205, 0.3671, -0.3731, -0.5347]]],
                            [[[0.7035, 0.2089, -0.1774, 3.4670],
                              [-0.8505, -0.9278, 1.4714, 0.1644],
                              [0.0898, 0.3531, -0.4007, 0.1927],
                              [1.2569, -0.2636, -0.5223, 0.0616]],
                             [[0.1760, -0.7639, -0.4600, -1.3260],
                              [-0.9921, -0.2970, -0.8955, 1.0508],
                              [1.3515, -0.1641, 1.9679, 1.1986],
                              [-0.3616, 0.6287, 0.4933, 0.3360]],
                             [[-0.5860, 0.2124, -0.8700, 2.4200],
                              [-0.0551, -1.5103, -1.6779, 0.8399],
                              [0.8431, 1.2414, -1.1243, -0.3887],
                              [-2.1254, 0.6047, -0.3515, 0.7254]]]],
                           device=device,
                           requires_grad=True)

    bbox = torch.tensor(
        [[[[1.3080e+01, 1.2688e+01, 1.1214e+01, 9.3944e+01, -9.1905e-01],
           [3.8104e+01, 1.0134e+01, 1.4659e+02, 9.0306e+01, -9.8211e-01],
           [-5.3213e+01, 4.9508e+01, 5.1513e+01, 3.2055e+01, -3.1954e-01],
           [2.6974e+01, 2.5248e+01, 5.4495e+01, 3.1083e+00, -6.2127e-01]],
          [[-1.5604e+01, -5.1908e+01, 2.3998e+02, 1.5008e+01, -1.2546e+00],
           [3.1354e+01, -7.3635e+00, 6.7879e+01, 3.5081e+01, -3.3851e-01],
           [-5.3292e+00, 9.1946e+00, 1.2834e+01, 1.0485e+01, -1.3039e+00],
           [-2.3925e+01, 3.6623e+01, 3.9875e+01, 7.2009e+01, -6.5934e-01]],
          [[7.2114e+01, -2.3781e+01, 2.9106e+01, 8.4501e+01, -1.1340e+00],
           [2.6258e+01, -7.7034e+00, 1.7629e+02, 1.0615e+02, -1.2156e+00],
           [3.8057e+01, 4.6016e+01, 1.2965e+01, 6.9384e+00, -1.0855e+00],
           [2.4428e+01, -1.6189e+01, 2.0572e+02, 3.1622e+01, -1.5719e-01]],
          [[3.8226e+00, 2.9608e+01, 1.4457e+01, 6.8179e+01, -9.1997e-01],
           [2.5003e+01, -4.2490e+01, 9.6007e+01, 4.9086e+01, -1.4786e+00],
           [8.5983e+01, 5.4980e+01, 7.8080e+01, 1.0003e+02, -1.0926e+00],
           [9.9065e+00, 4.1457e+01, 5.9799e+00, 1.7973e+01, -5.6313e-01]]],
         [[[-1.8244e+01, 4.6309e+00, 5.3010e+01, 2.4310e+01, -7.0345e-01],
           [1.9419e+01, 3.6704e+01, 5.2390e+01, 5.4133e+01, -3.7730e-01],
           [5.6387e+01, 2.3752e+01, 9.0441e+00, 1.7792e+01, -1.5583e+00],
           [3.6303e+01, 1.6396e+01, 2.0283e+01, 1.9148e+01, -8.3419e-01]],
          [[3.2169e+01, 3.0521e+01, 2.6283e+01, 1.9680e+02, -3.0454e-01],
           [2.5788e+01, -3.2189e+01, 8.8882e+01, 1.0207e+02, -1.5328e+00],
           [8.4676e+00, -1.6668e+01, 2.4657e+01, 1.1275e+02, -4.0388e-01],
           [-1.0799e+01, 6.0422e+00, 9.5807e+00, 3.3677e+01, -3.5438e-01]],
          [[6.9363e+01, 1.0850e+01, 2.5968e+01, 2.2311e+01, -1.6408e-01],
           [2.8140e+00, 4.6843e+00, 3.1289e+00, 2.1480e+01, -6.7583e-01],
           [2.6661e+01, 4.5290e+01, 6.1679e+00, 3.0005e+01, -8.9806e-01],
           [5.0871e+00, 1.3234e+01, 9.2087e+01, 4.9622e+01, -2.8020e-01]],
          [[-1.2643e+01, 2.5176e+01, 5.0488e+01, 5.4246e+01, -4.4840e-01],
           [-3.4521e+01, 9.8435e-01, 5.2413e+01, 9.7996e+00, -8.4218e-01],
           [4.9829e+01, -1.0808e+01, 2.9848e+01, 7.3579e+01, -6.2672e-01],
           [8.0446e+01, 2.8064e+01, 4.5273e+01, 5.3809e+01, -1.2359e+00]]]],
        device=device,
        requires_grad=True)

    expected_output = torch.tensor([[[[1.1095, -0.2172, -0.5222, -0.6225],
                                      [0.9144, 0.7662, 1.0487, -0.9690],
                                      [-0.8949, -1.6384, -0.9093, -0.3961],
                                      [-0.8604, 0.5062, -0.7947, -0.7397]],
                                     [[-0.3961, -0.7495, 1.3461, 1.5528],
                                      [0.2034, 0.5522, -1.6722, 0.5231],
                                      [-1.0062, 1.1350, 1.4225, -0.3951],
                                      [-0.4826, -1.6240, 0.1932, 2.7181]],
                                     [[-2.6436, -1.0276, 0.0578, -0.8344],
                                      [-0.9684, -1.8151, -2.1843, -1.1938],
                                      [-1.6744, -1.0121, -0.6556, 1.0073],
                                      [-0.8474, 0.3671, -0.3731, -0.5347]]],
                                    [[[0.7035, 0.2089, -0.1774, 3.4670],
                                      [-0.8505, -0.9278, 1.4714, 0.1644],
                                      [0.0898, 0.3064, -0.4007, 0.5849],
                                      [1.2569, -0.2636, -0.5223, 0.0616]],
                                     [[0.1760, -0.7639, -0.4600, -1.3260],
                                      [-0.9921, -0.2970, -0.8955, 1.0508],
                                      [1.3515, -0.6125, 1.9679, 0.5550],
                                      [-0.3616, 0.6287, 0.4933, 0.3360]],
                                     [[-0.5860, 0.2124, -0.8700, 2.4200],
                                      [-0.0551, -1.5103, -1.6779, 0.8399],
                                      [0.8431, 0.8455, -1.1243, -1.5994],
                                      [-2.1254, 0.6047, -0.3515, 0.7254]]]],
                                   device=device)

    expected_grad = torch.tensor([
        [[[1.0000, 1.8507, 1.1493, 1.5222], [1.0000, 1.1511, 1.2139, 1.4778],
          [1.0000, 1.2629, 1.3721, 1.0000], [3.0000, 1.0000, 1.0000, 2.0000]],
         [[1.0000, 1.8507, 1.1493, 1.5222], [1.0000, 1.1511, 1.2139, 1.4778],
          [1.0000, 1.2629, 1.3721, 1.0000], [3.0000, 1.0000, 1.0000, 2.0000]],
         [[1.0000, 1.8507, 1.1493, 1.5222], [1.0000, 1.1511, 1.2139, 1.4778],
          [1.0000, 1.2629, 1.3721, 1.0000], [3.0000, 1.0000, 1.0000, 2.0000]]],
        [[[1.2687, 1.5055, 1.2382, 1.0000], [1.1458, 1.4258, 1.4160, 1.0000],
          [1.0000, 1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000, 1.0000]],
         [[1.2687, 1.5055, 1.2382, 1.0000], [1.1458, 1.4258, 1.4160, 1.0000],
          [1.0000, 1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000, 1.0000]],
         [[1.2687, 1.5055, 1.2382, 1.0000], [1.1458, 1.4258, 1.4160, 1.0000],
          [1.0000, 1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000, 1.0000]]]
    ],
                                 device=device)

    output = rotated_feature_align(
        feature, bbox, spatial_scale=1 / 8, points=1)
    output.backward(torch.ones_like(output))
    assert torch.allclose(output, expected_output, 1e-2)
    assert torch.allclose(feature.grad, expected_grad, 1e-2)
