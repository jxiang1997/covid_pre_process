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

from align.regularization.regulariser import _Regulariser
from align.regularization.factory import RegisterRegularizer

"""
    Diffusion regularisation 
"""
@RegisterRegularizer('diffusion')
class DiffusionRegulariser(_Regulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(DiffusionRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "L2"

        if self._dim == 2:
            self._regulariser = self._l2_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._l2_regulariser_3d  # 3d regularisation

    def _l2_regulariser_2d(self, displacement):
        dx = (displacement[1:, 1:, :] - displacement[:-1, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, :] - displacement[1:, :-1, :]).pow(2) * self._pixel_spacing[1]

        return self._mask_2d(F.pad(dx + dy, (0, 1, 0, 1)))

    def _l2_regulariser_3d(self, displacement):
        dx = (displacement[1:, 1:, 1:, :] - displacement[:-1, 1:, 1:, :]).pow(2) * self._pixel_spacing[0]
        dy = (displacement[1:, 1:, 1:, :] - displacement[1:, :-1, 1:, :]).pow(2) * self._pixel_spacing[1]
        dz = (displacement[1:, 1:, 1:, :] - displacement[1:, 1:, :-1, :]).pow(2) * self._pixel_spacing[2]

        return self._mask_3d(F.pad(dx + dy + dz, (0, 1, 0, 1, 0, 1)))

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))
