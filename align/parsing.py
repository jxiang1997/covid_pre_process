import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # image loading
    parser.add_argument('--img_dir', type=str, default='/data/rsg/mammogram/detroit_data/pngs', help='dir of images. Note, image path in dataset jsons should stem from here')
    parser.add_argument('--metadata_dir', type=str, default='/data/rsg/mammogram/detroit_data/json_files', help='dir of metadata files.')

    parser.add_argument('--multi_image', action='store_true', default=False, help='Whether image will contain multiple slices. Slices could indicate different times, depths, or views')
    parser.add_argument('--num_images', type=int, default=1, help='In multi image setting, the number of images per single sample.')
    parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')

    # hyper parameter tuning
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate (default = 0.01)')
    parser.add_argument('--loss', type=str, default='mse', help="Type of loss functions to use. Choose between mse, ncc (normalized cross correlation), lcc (local normalized cross correlation), mi (mutual information), ngf (normlized gradient fields), and ssim (structural similarity image measure loss)")
    parser.add_argument('--regularization_weights', type=int, nargs='*', default=[1, 5, 50], help='regularization weight at each iteration. Each iteration is at a different downsample layer.')
    parser.add_argument('--displacement_regularizer', type=str, default='diffusion', help='displacement regularizer to use during bspline transformation. Can choose to incorporate multiple regularizers. Options include: isotropic_tv, tv, diffusion, sparsity')

    args = parser.parse_args()

    return args
