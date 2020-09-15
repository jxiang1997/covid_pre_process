# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch as th
import torch.nn.functional as F

import numpy as np

# Regulariser base class (standard from PyTorch)
class _Regulariser(th.nn.modules.Module):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(_Regulariser, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._weight = 1
        self._dim = len(pixel_spacing)
        self._pixel_spacing = pixel_spacing
        self.name = "parent"
        self._mask = None

    def SetWeight(self, weight):
        print("SetWeight is deprecated. Use set_weight instead.")
        self.set_weight(weight)

    def set_weight(self, weight):
        self._weight = weight

    def set_mask(self, mask):
        self._mask = mask

    def _mask_2d(self, df):
        if not self._mask is None:
            nx, ny, d = df.shape
            return df * self._mask.image.squeeze()[:nx,:ny].unsqueeze(-1).repeat(1,1,d)
        else:
            return df

    def _mask_3d(self, df):
        if not self._mask is None:
            nx, ny, nz, d = df.shape
            return df * self._mask.image.squeeze()[:nx,:ny,:nz].unsqueeze(-1).repeat(1,1,1,d)
        else:
            return df

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return self._weight*tensor.mean()
        if not self._size_average and self._reduce:
            return self._weight*tensor.sum()
        if not self._reduce:
            return self._weight*tensor
