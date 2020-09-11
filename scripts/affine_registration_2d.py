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
import ipdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import align as al

PLOT_DIR = "/data/rsg/mammogram/jxiang/plots"

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

    print("train data length: ", len(train_data))
    print("dev data length: ", len(dev_data))
    print("test data length: ", len(test_data))

    total_loss = 0

    for index in range(2):
        image_paths = train_data[index]['paths']

        fixed_image = al.image_utils.read_image_as_tensor(image_paths[0], dtype=dtype, device=device)
        moving_image = al.image_utils.read_image_as_tensor(image_paths[1], dtype=dtype, device=device)

        fixed_image, moving_image = al.image_filters.normalize_images(fixed_image, moving_image)

        # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
        # calculate the center of mass of the object
        fixed_image.image = 1 - fixed_image.image
        moving_image.image = 1 - moving_image.image

        # create pairwise registration object
        registration = al.registration.PairwiseRegistration()

        # choose the affine transformation model
        transformation = al.transform.SimilarityTransformation(moving_image, opt_cm=True)
        # initialize the translation with the center of mass of the fixed image
        transformation.init_translation(fixed_image)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        image_loss = al.loss.MSE(fixed_image, moving_image)

        registration.set_image_loss([image_loss])

        # choose the Adam optimizer to minimize the objective
        optimizer = th.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(100)

        # start the registration
        registration.start()

        total_loss += registration.loss

        # set the intensities back to the original for the visualisation
        fixed_image.image = 1 - fixed_image.image
        moving_image.image = 1 - moving_image.image

        # warp the moving image with the final transformation result
        displacement = transformation.get_displacement()
        warped_image = al.utils.warp_image(moving_image, displacement)

        end = time.time()

        print("=================================================================")

        print("Registration done in:", end - start, "s")
        print("Result parameters:")
        transformation.print()

        # plot the results
        plt.subplot(131)
        plt.imshow(fixed_image.numpy(), cmap='gray')
        plt.title('Fixed Image')

        plt.subplot(132)
        plt.imshow(moving_image.numpy(), cmap='gray')
        plt.title('Moving Image')

        plt.subplot(133)
        plt.imshow(warped_image.numpy(), cmap='gray')
        plt.title('Warped Moving Image')

        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)

        plt.savefig(os.path.join(PLOT_DIR, index + "_plot.png"))

    print("average loss over 100 iterations: ", total_loss/100)


  

if __name__ == "__main__":
    main()





