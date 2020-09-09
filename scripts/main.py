from datasets.detroit_mammo_with_prior import Detroit_Mammo_Cancer_With_Prior_Dataset

import argparse

parser = argparse.ArgumentParser()
# image loading
parser.add_argument('--img_dir', type=str, default='/data/rsg/mammogram/detroit_data/pngs', help='dir of images. Note, image path in dataset jsons should stem from here')
parser.add_argument('--metadata_dir', type=str, default='/data/rsg/mammogram/detroit_data/json_files', help='dir of metadata files.')

if __name__ == "__main__":
    args = parser.parse_args()
    train_data, dev_data, test_data = Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'train'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'dev'), Detroit_Mammo_Cancer_With_Prior_Dataset(args, 'test')


    print("train data length: ", len(train_data))
    print("dev data length: ", len(dev_data))
    print("test data length: ", len(test_data))