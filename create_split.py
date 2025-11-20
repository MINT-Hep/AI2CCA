import os
from dataset import save_splits
import argparse
import numpy as np
from dataset import Generic_WSI_Classification_Dataset

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--val_frac', type=float, default=0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default=0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--save_dir',  type=str, default=None,
                    help='split save directory')
parser.add_argument('--csv_dir',  type=str, default=None,
                    help='csv file directory')

args = parser.parse_args()

args.n_classes = 2
dataset = Generic_WSI_Classification_Dataset(csv_path=args.csv_dir,
                                             shuffle=False,
                                             seed=args.seed,
                                             print_info=True,
                                             label_dict={'Metastasis': 0, 'ICCA': 1},
                                             patient_strat=True,
                                             ignore=[])

num_slides_cls = len(dataset)

val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    split_dir = args.save_dir
    os.makedirs(split_dir, exist_ok=True)
    dataset.create_splits(k=args.k, val_num=val_num, test_num=test_num, custom_test_ids=None)
    for i in range(args.k):
        dataset.set_splits()
        descriptor_df = dataset.test_split_gen(return_descriptor=True)
        splits = dataset.return_splits(from_id=True)
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)),
                    boolean_style=True)
        descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))


