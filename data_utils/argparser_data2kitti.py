import argparse

class argparser_data2kitti():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='')
        self.parser.add_argument('--kaggle-dataset-path', dest='kaggle_dataset_path',
                                 help='path to kaggle dataset train and validation images', type=str)
        self.parser.add_argument('--mafa-dataset-path', dest='mafa_dataset_path',
                                 help='path to MAFA dataset train and validation images', type=str)
        self.parser.add_argument('--fddb-dataset-path', dest='fddb_dataset_path', help='path to fddb dataset train and validation images', type=str)
        self.parser.add_argument('--widerface-dataset-path', dest='widerface_dataset_path', help='path to widerface dataset train and validation images', type=str)
        self.parser.add_argument('--kitti-base-path', dest='kitti_base_path',
                                 help='path to save converted data set', type=str)
        self.parser.add_argument('--category-limit', dest='category_limit', default=6000,
                                 help='data limit for TLT', type=int)
        self.parser.add_argument('--tlt-input-dims_width', dest='tlt_input_dims_width', default=960,
                                 help = 'TLT input dimensions', type = int)
        self.parser.add_argument('--tlt-input-dims_height', dest='tlt_input_dims_height', default=544,
                                 help='TLT input dimensions', type=int)
        self.parser.add_argument('--label_filename', dest='label_filename', default='000_1OC3DT',
                                 help='File name for label checking', type=str)
        data_group = self.parser.add_mutually_exclusive_group()
        data_group.add_argument('--train', dest='train', help='Convert Training dataset to KITTI', action='store_true')
        data_group.add_argument('--val', dest='val', help='Convert validation dataset to KITTI', action='store_true')
        data_group.add_argument('--check_labels', dest='check_labels', help='Check if Converted dataset is right', action='store_true')

    def make_args(self):
        return self.parser.parse_args()