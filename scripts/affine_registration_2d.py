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
from align.datasets.detroit_mammo_with_prior import Detroit_Mammo_Cancer_With_Prior_Dataset #type: ignore
from torch.utils import data

import align.parsing as parsing
import sys
import os
import time

import matplotlib.pyplot as plt
import torch as th

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import align as al

def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    # device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:0")

    args = parsing.parse_args()

    Detroit_Mammo_Cancer_With_Prior_Dataset.set_args(args)
    train_data, dev_data, test_data = Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'train'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'dev'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'test')


    for index in range(200):
        x = train_data[index]['x']
        ipdb.set_trace()
        print("HI")





