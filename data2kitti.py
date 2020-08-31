from data_utils.widerface2kitti import widerFace2kitti
from data_utils.mafa2kitti import mafa2kitti
from data_utils.fddb2kitti import fddb2kitti
from data_utils.kaggle2kitti import kaggle2kitti
from data_utils.argparser_data2kitti import argparser_data2kitti
from data_utils.check_labels import test_labels
import os

def main():
    # Set Parameters
    arg_parser = argparser_data2kitti()
    args = arg_parser.make_args()
    # Datasets for Masked Faces
    kaggle_base_dir = args.kaggle_dataset_path # Use all data from Kaggle
    mafa_base_dir = args.mafa_dataset_path # Use only about 4000 images from MAFA
    # Datasets for No-Masked Faces
    fddb_base_dir = args.fddb_dataset_path # Use all data from FDDB
    widerface_base_dir = args.widerface_dataset_path # Use only from selected sub-folders
    ''' Note: Kaggle, FDDB Data sets does not have validation data thus we use all data for training '''
    # Store Converted annotations in KITTI format
    kitti_base_dir = args.kitti_base_path
    category_limit = [args.category_limit, args.category_limit]  # Mask / No-Mask Limits
    kitti_resize_dims = (args.tlt_input_dims_width, args.tlt_input_dims_height)  # Default for DetectNet-v2 : Look at TLT model requirements

    total_masks, total_no_masks = 0, 0
    count_masks, count_no_masks = 0, 0
    # To check if labels are converted in right format
    if args.check_labels:
        # Check from train directory
        test_labels(kitti_base_dir=kitti_base_dir + '/train/', file_name=args.label_filename)
    else:
        # ----------------------------------------
        # Kaggle Dataset Conversion
        # ----------------------------------------
        if args.train:
            images_dir = os.path.join(kaggle_base_dir, 'images') #r'C:\Users\ameykulkarni\Downloads\527030_966454_bundle_archive\images'
            labels_dir = os.path.join(kaggle_base_dir, 'labels') #r'C:\Users\ameykulkarni\Downloads\527030_966454_bundle_archive\labels'
            medical_mask2kitti = kaggle2kitti(images_dir=images_dir, labels_dir=labels_dir,
                                              category_limit=category_limit,
                                              kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims)
            count_masks, count_no_masks = medical_mask2kitti.get_data_attributes()
        # ----------------------------------------
        # MAFA Dataset Conversion
        # ----------------------------------------
        if args.train:
            annotation_file = os.path.join(mafa_base_dir, 'MAFA-Label-Train/LabelTrainAll.mat')
            mafa_base_dir = os.path.join(mafa_base_dir, 'train-images/images')
        if args.val:
            annotation_file = os.path.join(mafa_base_dir, 'MAFA-Label-Test/LabelTestAll.mat')
            mafa_base_dir = os.path.join(mafa_base_dir, 'test-images/images')

        total_masks += count_masks
        total_no_masks += count_no_masks
        print("Total Mask Labelled:{} and No-Mask Labelled:{}".format(total_masks, total_no_masks))
        category_limit_mod = [category_limit[0] - total_masks, category_limit[1] - total_no_masks]

        kitti_label = mafa2kitti(annotation_file=annotation_file, mafa_base_dir=mafa_base_dir,
                                 kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims,
                                 category_limit=category_limit_mod, train=args.train)
        count_masks, count_no_masks = kitti_label.mat2data()

        # ----------------------------------------
        # FDDB Dataset Conversion
        # ----------------------------------------
        if args.train:
            # Modifying category limit based on FDDB
            total_masks += count_masks
            total_no_masks += count_no_masks
            print("Total Mask Labelled:{} and No-Mask Labelled:{}".format(total_masks, total_no_masks))
            category_limit_mod = [category_limit[0]-total_masks, category_limit[1]-total_no_masks]
            fddb_base_dir = os.path.join(fddb_base_dir, 'originalPics') # r'C:\Users\ameykulkarni\Downloads\FDDB-folds\originalPics'
            annotation_path = os.path.join(fddb_base_dir, 'FDDB-folds') #r'C:\Users\ameykulkarni\Downloads\FDDB-folds\FDDB-folds'
            kitti_label = fddb2kitti(annotation_path=annotation_path, fddb_base_dir=fddb_base_dir,
                                     kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims,
                                     category_limit=category_limit_mod)
            count_masks, count_no_masks = kitti_label.fddb_data()

        # ----------------------------------------
        # Wider-Face Dataset Conversion
        # ----------------------------------------
        total_masks += count_masks
        total_no_masks += count_no_masks
        print("Total Mask Labelled:{} and No-Mask Labelled:{}".format(total_masks, total_no_masks))
        category_limit_mod = [category_limit[0] - total_masks, category_limit[1] - total_no_masks]

        if args.train:
            annotation_file = os.path.join(widerface_base_dir, 'wider_face_split/wider_face_train.mat')
            widerFace_base_dir = os.path.join(widerface_base_dir, 'WIDER_train/images')
        if args.val:
            # Modify this
            annotation_file = os.path.join(widerface_base_dir, 'wider_face_split/wider_face_val.mat')
            widerFace_base_dir = os.path.join(widerface_base_dir, 'WIDER_val/images')

        kitti_label = widerFace2kitti(annotation_file=annotation_file, widerFace_base_dir=widerFace_base_dir,
                                      kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims,
                                      category_limit=category_limit_mod, train=args.train)
        count_masks, count_no_masks = kitti_label.mat2data()
        total_masks += count_masks
        total_no_masks += count_no_masks
        print("----------------------------")
        print("Final: Total Mask Labelled:{}\nTotal No-Mask Labelled:{}".format(total_masks, total_no_masks))
        print("----------------------------")


if __name__ == '__main__':
    main()
