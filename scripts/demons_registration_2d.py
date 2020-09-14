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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import align as al

import align.loss.factory as loss_factory

PLOT_DIR = "/data/rsg/mammogram/jxiang/demons_registration_2d_plots"

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

        fixed_image = al.utils.image_utils.read_image_as_tensor(image_paths[0], dtype=dtype, device=device)
        moving_image = al.utils.image_utils.read_image_as_tensor(image_paths[1], dtype=dtype, device=device)

        # create pairwise registration object
        registration = al.registration.DemonsRegistraion(verbose=True)

        # choose the affine transformation model
        transformation = al.transform.transform.NonParametricTransformation(moving_image.size,
                                                                                dtype=dtype,
                                                                                device=device,
                                                                                diffeomorphic=True)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        image_loss = loss_factory.get_loss(args.loss)(fixed_image, moving_image)

        registration.set_image_loss([image_loss])

        # choose a regulariser for the demons
        regulariser = al.demons_regularization.GaussianRegulariser(moving_image.spacing, sigma=[2, 2], dtype=dtype,
                                                                device=device)
        
        registration.set_regulariser([regulariser])

        # choose the Adam optimizer to minimize the objective
        optimizer = th.optim.Adam(transformation.parameters(), lr=0.001)

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(1000)

        # start the registration
        registration.start()

        # warp the moving image with the final transformation result
        displacement = transformation.get_displacement()

        # use the shaded version of the fixed image for visualization
        warped_image = al.transform.utils.warp_image(moving_image, displacement)

        end = time.time()

        displacement = al.utils.image_utils.create_displacement_image_from_image(displacement, moving_image)

        print("=================================================================")

        print("Registration done in: ", end - start)

        # plot the results
        plt.subplot(221)
        plt.imshow(fixed_image.numpy(), cmap='gray')
        plt.title('Fixed Img')

        plt.subplot(222)
        plt.imshow(moving_image.numpy(), cmap='gray')
        plt.title('Moving Img')

        plt.subplot(223)

        red_warped_image = np.repeat(warped_image.numpy()[:,:,np.newaxis], 3, axis=2)
        red_warped_image[:,:,1] = 0
        red_warped_image[:,:,2] = 0

        blue_fixed_image = np.repeat(fixed_image.numpy()[:,:,np.newaxis], 3, axis=2)
        blue_fixed_image[:,:,0] = 0
        blue_fixed_image[:,:,1] = 0

        print("WAREPD IMAGE SAME AS FIXED: ", warped_image == fixed_image)

        img = red_warped_image + blue_fixed_image
        plt.imshow(warped_image.numpy())
        plt.title('Warped Moving Img')

        plt.subplot(224)
        plt.imshow(displacement.magnitude().numpy(), cmap='jet')
        plt.title('Displacement')

        plot_dir = PLOT_DIR + '_' + args.loss

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.rcParams["figure.figsize"] = [64,36]
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, str(index) + "_plot.png"))

        # write result images
        # sitk.WriteImage(warped_image.itk(), '/tmp/demons_warped_image.vtk')
        # sitk.WriteImage(moving_image.itk(), '/tmp/demons_moving_image.vtk')
        # sitk.WriteImage(fixed_image.itk(), '/tmp/demons_fixed_image.vtk')
        # sitk.WriteImage(shaded_image.itk(), '/tmp/demons_shaded_image.vtk')
        # sitk.WriteImage(displacement.itk(), '/tmp/demons_displacement_image.vtk')


if __name__ == '__main__':
    main()