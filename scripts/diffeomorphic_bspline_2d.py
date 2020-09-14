
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
import numpy as np

import matplotlib.pyplot as plt
import torch as th
import ipdb
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import align as al

PLOT_DIR = "/data/rsg/mammogram/jxiang/diffeomorphic_bspline_2d_plots"

def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    # device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:3")

    args = parsing.parse_args()

    Detroit_Mammo_Cancer_With_Prior_Dataset.set_args(args)
    train_data, dev_data, test_data = Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'train'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'dev'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'test')

    total_loss = 0

    for index in range(2):

        image_paths = train_data[index]['paths']

        fixed_image = al.image_utils.read_image_as_tensor(image_paths[0], dtype=dtype, device=device)
        moving_image = al.image_utils.read_image_as_tensor(image_paths[1], dtype=dtype, device=device)

        fixed_image, moving_image = al.image_filters.normalize_images(fixed_image, moving_image)
        # create image pyramide size/4, size/2, size/1
        fixed_image_pyramid = al.image_utils.create_image_pyramid(fixed_image, [[4, 4], [2, 2]])
        moving_image_pyramid = al.image_utils.create_image_pyramid(moving_image, [[4, 4], [2, 2]])

        constant_displacement = None
        regularisation_weight = [1, 5, 50]
        number_of_iterations = [500, 500, 500]
        sigma = [[11, 11], [11, 11], [3, 3]]

        for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

            registration = al.registration.PairwiseRegistration(verbose=True)

            # define the transformation
            transformation = al.transform.BsplineTransformation(mov_im_level.size,
                                                                            sigma=sigma[level],
                                                                            order=3,
                                                                            dtype=dtype,
                                                                            device=device,
                                                                            diffeomorphic=True)

            if level > 0:
                constant_flow = al.utils.upsample_displacement(constant_flow,
                                                                                    mov_im_level.size,
                                                                                    interpolation="linear")
                transformation.set_constant_flow(constant_flow)

            registration.set_transformation(transformation)

            # choose the Mean Squared Error as image loss
            image_loss = al.loss.MSE(fix_im_level, mov_im_level)

            registration.set_image_loss([image_loss])

            # define the regulariser for the displacement
            regulariser = al.regularization.DiffusionRegulariser(mov_im_level.spacing)
            regulariser.SetWeight(regularisation_weight[level])

            registration.set_regulariser_displacement([regulariser])

            #define the optimizer
            optimizer = th.optim.Adam(transformation.parameters())

            registration.set_optimizer(optimizer)
            registration.set_number_of_iterations(number_of_iterations[level])

            registration.start()

            constant_flow = transformation.get_flow()

        # create final result
        displacement = transformation.get_displacement()
        warped_image = al.utils.warp_image(moving_image, displacement)
        displacement = al.image_utils.create_displacement_image_from_image(displacement, moving_image)

        end = time.time()

        print("=================================================================")

        print("Registration done in: ", end - start)
        print("Result parameters:")

        # plot the results
        plt.subplot(141)
        plt.imshow(fixed_image.numpy(), cmap='gray')
        plt.title('Fixed Image')

        plt.subplot(142)
        plt.imshow(moving_image.numpy(), cmap='gray')
        plt.title('Moving Image')

        plt.subplot(143)
        red_warped_image = np.repeat(warped_image.numpy()[:,:,np.newaxis], 3, axis=2)
        red_warped_image[:,:,1] = 0
        red_warped_image[:,:,2] = 0

        blue_fixed_image = np.repeat(fixed_image.numpy()[:,:,np.newaxis], 3, axis=2)
        blue_fixed_image[:,:,0] = 0
        blue_fixed_image[:,:,1] = 0

        print("WAREPD IMAGE SAME AS FIXED: ", warped_image == fixed_image)

        img = red_warped_image + blue_fixed_image
        plt.imshow(img)
        plt.title('Warped Moving Image')

        plt.subplot(144)
        plt.imshow(displacement.magnitude().numpy(), cmap='jet')
        plt.title('Magnitude Displacement')

        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)

        plt.rcParams["figure.figsize"] = [64,36]
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, str(index) + "_plot.png"))

if __name__ == '__main__':
	main()