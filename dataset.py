from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils import generate_split, nth


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    """
    Save current split slide_ids as a CSV.
    """
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

    df.to_csv(filename)
    print()


class Generic_WSI_Classification_Dataset(Dataset):
    """
    Slide-level dataset with patient-level optional stratification.
    Always reads features from .h5 files.
    """

    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 filter_dict={},
                 ignore=[],
                 patient_strat=False,
                 label_col=None,
                 patient_voting='max'
                 ):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            shuffle (bool): Whether to shuffle rows (slide order).
            seed (int): Random seed for shuffling and split generation.
            print_info (bool): Whether to print dataset summary.
            label_dict (dict): Mapping from raw label to int id.
            filter_dict (dict): Filter rows where df[key] in val.
            ignore (list): Label names to ignore/drop.
            patient_strat (bool): If True, stratify at patient/case level.
            label_col (str): Column name to use as label (copied to 'label').
            patient_voting (str): 'max' or 'maj' to aggregate slide labels to a patient label.
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        # Optional shuffle
        if shuffle:
            np.random.seed(seed)
            slide_data = slide_data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        # will store fixed test ids (in patient or slide index space depending on patient_strat)
        self._fixed_test_ids = None

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        """Prepare class index lists at patient and slide levels."""
        # patient-level class ids
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        # slide-level class ids
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        """Aggregate slide labels to patient/case labels by voting."""
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max()  # MIL convention
            elif patient_voting == 'maj':
                label = stats.mode(label, keepdims=False)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        """Map raw labels to int ids and drop ignored labels."""
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    def filter_df(self, df, filter_dict={}):
        """Filter dataframe by dict of column -> allowed values."""
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        return len(self.patient_data['case_id']) if self.patient_strat else len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts:\n", self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; #samples in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL;   #samples in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    # -------------------------------
    # FIXED-TEST implementation
    # -------------------------------
    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        """
        Create a generator that yields (train_ids, val_ids, test_ids).
        Here we ensure TEST SET IS FIXED across all folds:
          - If custom_test_ids is None:
              run a single split generation (n_splits=1) to sample a fixed test set,
              store it in self._fixed_test_ids
          - Then generate k folds with custom_test_ids=self._fixed_test_ids,
            so test_ids stay fixed while train/val rotate.
        """
        # Step 1: decide the index space and class ids
        if self.patient_strat:
            index_space = 'patient'
            cls_ids = self.patient_cls_ids
            num_samples = len(self.patient_data['case_id'])
        else:
            index_space = 'slide'
            cls_ids = self.slide_cls_ids
            num_samples = len(self.slide_data)

        # Step 2: determine fixed test ids once
        if custom_test_ids is None and self._fixed_test_ids is None:
            # generate exactly 1 split to pick test ids based on the given test_num
            first_settings = {
                'n_splits': 1,
                'val_num': val_num,
                'test_num': test_num,
                'label_frac': label_frac,
                'seed': self.seed,
                'custom_test_ids': None,
                'cls_ids': cls_ids,
                'samples': num_samples
            }
            first_gen = generate_split(**first_settings)
            first_ids = next(first_gen)  # (train_ids, val_ids, test_ids) in index_space
            self._fixed_test_ids = first_ids[2]
        elif custom_test_ids is not None:
            self._fixed_test_ids = custom_test_ids  # respect external fixed test ids

        # Step 3: main k-fold generator with fixed test ids
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,       # value may be ignored by generate_split when custom_test_ids given
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': self._fixed_test_ids,
            'cls_ids': cls_ids,
            'samples': num_samples
        }
        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        """
        Consume the split generator and map ids back to slide indices.
        Test ids remain fixed across folds thanks to create_splits().
        """
        if start_from is not None:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            # Map patient indices to slide indices
            slide_ids = [[] for _ in range(len(ids))]
            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)
            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            # Already in slide-index space
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        """
        Build a Generic_Split from a pre-saved CSV split file for a given split key.
        """
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        """
        Merge multiple split keys into a single dataset.
        """
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(merged_split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id=True, csv_path=None):
        """
        Return (train_split, val_split, test_split).
        If from_id=True, use indices already stored in self.train_ids/val_ids/test_ids.
        Otherwise read from a CSV with columns ['train','val','test'].
        """
        if from_id:
            if self.train_ids is not None and len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)
            else:
                train_split = None

            if self.val_ids is not None and len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)
            else:
                val_split = None

            if self.test_ids is not None and len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
            else:
                test_split = None

        else:
            assert csv_path
            # Important: keep dtype consistent with slide_id dtype
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        # Base class returns nothing; use child classes below.
        return None

    def test_split_gen(self, return_descriptor=False):
        """
        Print split statistics and optionally return a descriptor DataFrame.
        """
        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in
                     range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index,
                              columns=columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        """
        Save current split IDs (by slide_id) into CSV columns: train, val, test.
        """
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    """
    TITAN variant that loads features+coords from <data_dir>/<slide_id>.h5
    and returns (features, coords, patch_size_lv0, label).
    """

    def __init__(self, data_dir, **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]

        if isinstance(self.data_dir, dict):
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        full_path = os.path.join(data_dir, f'{slide_id}.h5')
        with h5py.File(full_path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        features = torch.from_numpy(features)
        patch_size_lv0 = torch.tensor(512)  # constant for TITAN pipeline
        return features, coords, patch_size_lv0, label


class Generic_Split(Generic_MIL_Dataset):
    """
    A split view over a subset of slide_data rows.
    """

    def __init__(self, slide_data, data_dir=None, num_classes=2):
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)
