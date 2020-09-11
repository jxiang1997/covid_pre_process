import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # image loading
    parser.add_argument('--img_dir', type=str, default='/data/rsg/mammogram/detroit_data/pngs', help='dir of images. Note, image path in dataset jsons should stem from here')
    parser.add_argument('--metadata_dir', type=str, default='/data/rsg/mammogram/detroit_data/json_files', help='dir of metadata files.')

    parser.add_argument('--multi_image', action='store_true', default=False, help='Whether image will contain multiple slices. Slices could indicate different times, depths, or views')
    parser.add_argument('--num_images', type=int, default=1, help='In multi image setting, the number of images per single sample.')
    parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')

    args = parser.parse_args()

    return args
