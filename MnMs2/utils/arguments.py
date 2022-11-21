import argparse
import json
import os


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='PAIP 2020 Challenge - CRC Prediction', formatter_class=SmartFormatter)

parser.add_argument("--gpu", type=str, default="0,1")
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument('--output_dir', type=str, default="C:/Users/alejo/Downloads/4toSemestre/MnMsCardiac-master/results",help='Where progress/checkpoints will be saved')

parser.add_argument('--epochs', type=int, default=100, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size for training')
parser.add_argument('--data_fold', type=str, default="1",help='Which training Fold (Cross Validation)')
parser.add_argument('--num_classes', type=int, default=1, help='Model output neurons')
parser.add_argument('--data_fold_validation', type=str, help='Which testing Fold (Only when folding by vendor)')
parser.add_argument('--fold_system', type=str, default="patient",help='How to create data folds')
parser.add_argument('--dataset', type=str,default="mnms", help='Dataset to use')
parser.add_argument('--label_type', type=str, default="mask",help='"mask" for segmentation or "vendor_label" for classification')

parser.add_argument('--model_name', type=str, default='resnet34_unet_scratch', help='Model name for training')
parser.add_argument('--data_augmentation', type=str,default="combination_old", help='Apply data augmentations at train time')
parser.add_argument('--crop_size', type=int, default=224, help='Center crop squared size')
parser.add_argument('--img_size', type=int, default=224, help='Final img squared size')

parser.add_argument('--binary_threshold', type=float, default=0.5, help='Threshold for masks probabilities')

parser.add_argument('--criterion', type=str, default='bce_dice_border_ce', help='Criterion for training')
parser.add_argument('--weights_criterion', type=str, default='0.5,0.2,0.2,0.2,0.5', help='Weights for each subcriterion')

parser.add_argument('--model_checkpoint', type=str, default=r"models\model_resnet34_unet_scratch_best_dice.pt", help='Where is the model checkpoint saved')
parser.add_argument('--segmentator_checkpoint', type=str, default="", help='Segmentator checkpoint (predict v2)')
parser.add_argument('--discriminator_checkpoint', type=str, default="", help='Dicriminator checkpoint (predict v2)')
parser.add_argument('--defrost_epoch', type=int, default=7, help='Number of epochs to defrost the model')
parser.add_argument('--normalization', type=str, default="standardize",required=False, help='Data normalization method')

parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
parser.add_argument('--scheduler', type=str, default="steps", help='Where is the model checkpoint saved')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--min_lr', type=float, default=0.0001, help='Minimun Learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='Maximum Learning rate')
parser.add_argument('--scheduler_steps', '--arg', nargs='+', type=int, default=[55,89],help='Steps when steps scheduler choosed')

parser.add_argument('--add_depth',action='store_true', help='If apply image transformation 1 to 3 channels or not')
parser.add_argument('--weakly_labelling', action='store_true', help='Use weakly labels for class C')

parser.add_argument('--apply_swa', action='store_true', help='Apply stochastic weight averaging')
parser.add_argument('--swa_freq', type=int, default=1, help='SWA Frequency')
parser.add_argument('--swa_start', type=int, default=60, help='SWA_LR')
parser.add_argument('--swa_lr', type=float, default=0.0001, help='SWA_LR')

parser.add_argument('--eval_overlays', action='store_true', help='Generate predictions overlays')
parser.add_argument(
    '--eval_overlays_path', type=str, default='none',
    help='Where to save predictions overlays. If "none" no overlays are generated'
)

# For fold_eval.py
parser.add_argument('--evaluation_folder', type=str, default="",
                    help='Folder to save evaluation results. If empty same as model path')
parser.add_argument('--evaluation_descriptor', type=str, default="eval",
                    help='Subfolder name to save evaluation results')

# For prediction/submission
parser.add_argument('--input_data_directory', type=str, default="C:/Users/alejo/Downloads/4toSemestre/MnMsCardiac-master/Data/Testing", help='Folder with volumes to predict')
parser.add_argument('--output_data_directory', type=str, default="C:/Users/alejo/Downloads/4toSemestre/MnMsCardiac-master/results", help='Folder to save prediction')

# For prediction v2 entropy adaptation
parser.add_argument('--target', type=str, default='B', help='Desired domain to transform')
parser.add_argument('--out_threshold', type=float, default=0.01, help='Difference stop condition')
parser.add_argument('--max_iters', type=int, default=100, help='Maximum number of iters to apply entropy')

parser.add_argument('--entropy_lambda', type=float, default=0.99, help='Learning rate')

parser.add_argument('--add_l1', action='store_true', help='If add L1 loss or not')
parser.add_argument('--l1_lambda', type=float, default=0.0, help='L1 impact factor')

parser.add_argument('--add_blur_param', action='store_true', help='Add blur matrix param or not')
parser.add_argument('--blur_lambda', type=float, default=0.0, help='Blur param impact factor')

parser.add_argument('--add_unblur_param', action='store_true', help='Add unblur matrix param or not')
parser.add_argument('--unblur_lambda', type=float, default=0.0, help='Unblur param impact factor')

parser.add_argument('--add_gamma_param', action='store_true', help='Add gamma param or not')
parser.add_argument('--gamma_lambda', type=float, default=0.0, help='Gamma param impact factor')

parser.add_argument('--segmentator_model_name', type=str, default='simple_unet', help='Segmentator model name')
parser.add_argument('--discriminator_model_name', type=str, default='simple_unet', help='Discriminator model name')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if args.output_data_directory == "":
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # https://stackoverflow.com/a/55114771
    with open(args.output_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
