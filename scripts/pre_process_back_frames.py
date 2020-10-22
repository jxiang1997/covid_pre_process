import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import time
from PIL import Image
import os
import align as al
import align.loss.factory as loss_factory
import align.regularization.factory as regularizer_factory
import SimpleITK as sitk
import torch as th
import json
from p_tqdm import p_map
import tqdm
import warnings
warnings.filterwarnings("ignore")

NP_SAVE_DIR_BACK = "/Users/jxiang/Dropbox/Covid-Thermal-Images-MIT-Tec/back_image_relevant_frames"
NP_SAVE_DIR_FRONT = "/Users/jxiang/Dropbox/Covid-Thermal-Images-MIT-Tec/front_image_relevant_frames"
NP_SAVE_DIR_RIGHT = "/Users/jxiang/Dropbox/Covid-Thermal-Images-MIT-Tec/right_image_relevant_frames"
NP_SAVE_DIR_LEFT = "/Users/jxiang/Dropbox/Covid-Thermal-Images-MIT-Tec/left_image_relevant_frames"

THERMAL_VID_FILES_DIR = "/Users/jxiang/Dropbox/Covid-Thermal-Images-MIT-Tec/np_files"
COVID_FILE_METADATA = "/Users/jxiang/Dropbox/Omega TI Analyzer/file_metadata.json"

THERMAL_TEMPLATE_PATH = "/Users/jxiang/Dropbox/Covid-Thermal-Images-MIT-Tec/np_files/2020-08-12_130643.IRS.npy"

FRONT_FRAME_FACTOR = 0.08
BACK_FRAME_FACTOR = 0.69
LEFT_FRAME_FACTOR = 0.35
RIGHT_FRAME_FACTOR = 0.92


def rotate_landscape_image(landscape_image, width, height):
    rotated_image = np.rot90(landscape_image, k=3)
    res = rotated_image[int((width-height)/2):int(width - (width-height)/2),:]
    min_val = np.amin(res)
    res = np.pad(res, ((0,0),(0,int(width-height))), 'constant', constant_values=np.amin(res))
    return res

def get_frames(thermal_np_path, frame_factor, covid_file_metadata_path):
    with open(covid_file_metadata_path) as json_file:
        covid_file_metadata = json.load(json_file)
        
        thermal_width, thermal_height = 384, 288
        thermal_vid = np.load(thermal_np_path)
        num_frame, height_width = thermal_vid.shape

        landscape = 'Landscape' in covid_file_metadata[thermal_np_path.split('/')[-1]]["Quality Observation"] or 'Lanscape' in covid_file_metadata[thermal_np_path.split('/')[-1]]["Quality Observation"]
        frame_percent = [i for i in np.arange(frame_factor - 0.08, frame_factor + 0.08, 0.01)]
        frames = [int(i * num_frame) for i in frame_percent]

        if landscape:
            thermal_images = [rotate_landscape_image(np.reshape(thermal_vid[frame], (thermal_height, thermal_width)), thermal_width, thermal_height) for frame in frames]
        else:
            thermal_images = [np.reshape(thermal_vid[frame], (thermal_height, thermal_width)) for frame in frames]
    
    return thermal_images

THERMAL_TEMPLATE_BACK = get_frames(THERMAL_TEMPLATE_PATH, BACK_FRAME_FACTOR, COVID_FILE_METADATA)[3]
THERMAL_TEMPLATE_FRONT = get_frames(THERMAL_TEMPLATE_PATH, FRONT_FRAME_FACTOR, COVID_FILE_METADATA)[2]
THERMAL_TEMPLATE_LEFT = get_frames(THERMAL_TEMPLATE_PATH, LEFT_FRAME_FACTOR, COVID_FILE_METADATA)[4]
THERMAL_TEMPLATE_RIGHT = get_frames(THERMAL_TEMPLATE_PATH, RIGHT_FRAME_FACTOR, COVID_FILE_METADATA)[2]

def save_numpy_file(np_data, np_save_dir, filename):
    save_path = os.path.join(np_save_dir, filename)
    with open(save_path, 'wb') as f:
        np.save(f, np_data)



def affine_registration(fixed_image_np, moving_image_np):
    fixed_image = sitk.GetImageFromArray(fixed_image_np)
    moving_image = sitk.GetImageFromArray(moving_image_np)
    fixed_image = al.utils.image_utils.create_tensor_image_from_itk_image(fixed_image)
    moving_image = al.utils.image_utils.create_tensor_image_from_itk_image(moving_image)
    
    fixed_min = fixed_image.image.min()
    moving_min = moving_image.image.min()
    
    fixed_max = fixed_image.image.max() - fixed_min
    moving_max = moving_image.image.max() - moving_min 
    
    fixed_image, moving_image = al.image_filters.normalize_images(fixed_image, moving_image)

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
    # calculate the center of mass of the object
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.registration.PairwiseRegistration(verbose=False)

    # choose the affine transformation model
    transformation = al.transform.transform.SimilarityTransformation(moving_image, opt_cm=True)
    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    image_loss = al.loss.ncc.NCC(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.LBFGS(transformation.parameters(), lr=0.01)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(5)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transform.utils.warp_image(moving_image, displacement)
        
    return fixed_image, moving_image, warped_image, fixed_max, fixed_min, moving_max, moving_min, registration.loss

def unnormalize_image(image, normalize_max, normalize_min):
    image.image *= normalize_max 
    image.image += normalize_min
    return image

def screen_frames(test_thermal, thermal_template):
    image_list = []
    loss_list = []
    for test_image in test_thermal:
        fixed_image, moving_image, warped_image, fixed_max, fixed_min, moving_max, moving_min, loss = affine_registration(thermal_template, test_image)
        unnormalized_fixed_image = unnormalize_image(fixed_image, fixed_max, fixed_min)
        unnormalized_moving_image = unnormalize_image(moving_image, moving_max, moving_min)
        unnormalized_warped_image = unnormalize_image(warped_image, moving_max, moving_min)
        image_list.append((unnormalized_fixed_image, unnormalized_moving_image, unnormalized_warped_image)) 
        loss_list.append(loss)
    
    return image_list[np.argmin(loss_list)]

def screen_relevant_frames(test_thermal, thermal_template):
    '''
    gets 15 most relevant frames
    '''
    image_list = []
    loss_list = []
    for test_image in test_thermal:
        fixed_image, moving_image, warped_image, fixed_max, fixed_min, moving_max, moving_min, loss = affine_registration(thermal_template, test_image)
        unnormalized_fixed_image = unnormalize_image(fixed_image, fixed_max, fixed_min)
        unnormalized_moving_image = unnormalize_image(moving_image, moving_max, moving_min)
        unnormalized_warped_image = unnormalize_image(warped_image, moving_max, moving_min)
        image_list.append((unnormalized_fixed_image, unnormalized_moving_image, unnormalized_warped_image)) 
        loss_list.append(loss)
    
    most_relevant_frame_index = np.argmin(loss_list)

    if most_relevant_frame_index < 7:
        relevant_frames = [image_list[i][1].numpy() for i in range(0, 15)]
    elif most_relevant_frame_index > len(image_list) - 8:
        relevant_frames = [image_list[i][1].numpy() for i in range(- 16, -1)]
    else:
        relevant_frames = [image_list[i][1].numpy() for i in range(most_relevant_frame_index - 7, most_relevant_frame_index + 8)]
    
    return relevant_frames

def convert_videos_to_frames(file_name):
    example = os.path.join(THERMAL_VID_FILES_DIR, file_name)  
    back_thermal_frames = get_frames(example, BACK_FRAME_FACTOR, COVID_FILE_METADATA)
    front_thermal_frames = get_frames(example, FRONT_FRAME_FACTOR, COVID_FILE_METADATA)
    right_thermal_frames = get_frames(example, RIGHT_FRAME_FACTOR, COVID_FILE_METADATA)
    left_thermal_frames = get_frames(example, LEFT_FRAME_FACTOR, COVID_FILE_METADATA)

    # warped_back = screen_frames(back_thermal_frames, THERMAL_TEMPLATE_BACK)[2].numpy()
    # warped_front = screen_frames(front_thermal_frames, THERMAL_TEMPLATE_FRONT)[2].numpy()
    # warped_left = screen_frames(left_thermal_frames, THERMAL_TEMPLATE_LEFT)[2].numpy()
    # warped_right =  screen_frames(right_thermal_frames, THERMAL_TEMPLATE_RIGHT)[2].numpy()

    back_frames = screen_relevant_frames(back_thermal_frames, THERMAL_TEMPLATE_BACK)
    front_frames = screen_relevant_frames(front_thermal_frames, THERMAL_TEMPLATE_FRONT)
    left_frames = screen_relevant_frames(left_thermal_frames, THERMAL_TEMPLATE_LEFT)
    right_frames = screen_relevant_frames(right_thermal_frames, THERMAL_TEMPLATE_RIGHT)

    for i in range(len(back_frames)):
        save_numpy_file(back_frames[i], NP_SAVE_DIR_BACK, file_name[:-8] + '_' + str(i) + file_name[-8:])
        save_numpy_file(front_frames[i], NP_SAVE_DIR_FRONT, file_name[:-8] + '_' + str(i) + file_name[-8:])
        save_numpy_file(left_frames[i], NP_SAVE_DIR_LEFT, file_name[:-8] + '_' + str(i) + file_name[-8:])
        save_numpy_file(right_frames[i], NP_SAVE_DIR_RIGHT, file_name[:-8] + '_' + str(i) + file_name[-8:])

if __name__ == "__main__":

    files = os.listdir(THERMAL_VID_FILES_DIR)

    with open(COVID_FILE_METADATA) as json_file:
        covid_file_metadata = json.load(json_file)

        IRS_files = [f for f in files if 'IRS' in f and f in covid_file_metadata]

        # for f in IRS_files:
        #     convert_videos_to_frames(f)

        p_map(convert_videos_to_frames,IRS_files)
        



