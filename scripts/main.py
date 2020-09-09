from datasets.detroit_mammo_with_prior import Detroit_Mammo_Cancer_With_Prior_Dataset

import argparse
import ipdb

parser = argparse.ArgumentParser()
# image loading
parser.add_argument('--img_dir', type=str, default='/data/rsg/mammogram/detroit_data/pngs', help='dir of images. Note, image path in dataset jsons should stem from here')
parser.add_argument('--metadata_dir', type=str, default='/data/rsg/mammogram/detroit_data/json_files', help='dir of metadata files.')

parser.add_argument('--multi_image', action='store_true', default=False, help='Whether image will contain multiple slices. Slices could indicate different times, depths, or views')
parser.add_argument('--num_images', type=int, default=1, help='In multi image setting, the number of images per single sample.')
parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')


if __name__ == "__main__":
    args = parser.parse_args()
    Detroit_Mammo_Cancer_With_Prior_Dataset.set_args(args)
    train_data, dev_data, test_data = Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'train'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'dev'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'test')


    print("train data length: ", len(train_data))
    ipdb.set_trace()
    print("example train data: ", train_data[0])
    print("dev data length: ", len(dev_data))
    print("test data length: ", len(test_data))