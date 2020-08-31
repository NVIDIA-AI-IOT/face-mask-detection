import scipy.io
import os
from PIL import Image, ImageDraw
import numpy as np

class mafa2kitti():
    def __init__(self, annotation_file, mafa_base_dir, kitti_base_dir, kitti_resize_dims, category_limit, train):
        self.annotation_file = annotation_file
        self.data = scipy.io.loadmat(self.annotation_file)
        self.kitti_base_dir = kitti_base_dir
        self.mafa_base_dir = mafa_base_dir
        self.count_mask = category_limit[0]
        self.count_no_mask = category_limit[1]
        self.kitti_resize_dims = kitti_resize_dims
        self.train = train
        if self.train:
            self.len_dataset = len(self.data["label_train"][0])
            try:
                os.makedirs(self.kitti_base_dir+'/train/images',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")
            self.kitti_images = os.path.join(self.kitti_base_dir, 'train/images')
            try:
                os.makedirs(self.kitti_base_dir+ '/train/labels',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")
            self.kitti_labels = os.path.join(self.kitti_base_dir, 'train/labels')
        else:
            self.len_dataset = len(self.data["LabelTest"][0])
            try:
                os.makedirs(self.kitti_base_dir+'/test/images',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")
            self.kitti_images = os.path.join(self.kitti_base_dir, 'test/images')
            try:
                os.makedirs(self.kitti_base_dir+'/test/labels',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")
            self.kitti_labels = os.path.join(self.kitti_base_dir, 'test/labels')

    def extract_labels(self, i, train_flag, _count_mask, _count_no_mask):
        if train_flag:
            train_image = self.data["label_train"][0][i]
            train_image_name = str(train_image[1]).strip("['']")  # Test [0]
            categories = []
            bboxes = []
            for i in range(0, len(train_image[2])):
                _bbox_label = train_image[2][i]  # Test[1][0]
                _category_id = _bbox_label[12]  # Occ_Type: For Train: 13th, 10th in Test
                _occulution_degree = _bbox_label[13]
                bbox = [_bbox_label[0], _bbox_label[1], _bbox_label[0]+_bbox_label[2], _bbox_label[1]+_bbox_label[3]]
                if (_category_id != 3 and _occulution_degree > 2) and (_count_mask < self.count_mask):
                    category_name = 'Mask'  # Faces with Mask
                    _count_mask += 1
                    count = 0
                    categories.append(category_name)
                    bboxes.append(bbox)
                elif (_category_id==3 and _occulution_degree<2) and (_count_no_mask < self.count_no_mask):
                    category_name = 'No-Mask'  # Faces with Mask
                    _count_no_mask += 1
                    count = 0
                    categories.append(category_name)
                    bboxes.append(bbox)
            if bboxes:
                if not self.check_image_dims(image_name=train_image_name):
                    self.make_labels(image_name=train_image_name, category_names=categories,
                                             bboxes=bboxes)

        else:
            test_image = self.data["LabelTest"][0][i]
            test_image_name = str(test_image[0]).strip("['']")  # Test [0]
            categories = []
            bboxes = []
            for i in range(0, len(test_image[1])):
                _bbox_label = test_image[1][i]  # Test[1][0]
                # Occ_Type: For Train: 13th, 10th in Test
                # In test Data: refer to Face_type, 5th
                _face_type = _bbox_label[4] # Face Type
                _occ_type = _bbox_label[9]
                _occ_degree = _bbox_label[10]
                bbox = [_bbox_label[0], _bbox_label[1], _bbox_label[0] + _bbox_label[2], _bbox_label[1] + _bbox_label[3]]
                if (_face_type==1 and _occ_type!=3 and _occ_degree > 2) and _count_mask < self.count_mask:
                    category_name = 'Mask'
                    bboxes.append(bbox)
                    categories.append(category_name)
                    _count_mask+=1
                elif (_face_type==2) and _count_mask < self.count_mask:
                    category_name = 'No-Mask'
                    bboxes.append(bbox)
                    categories.append(category_name)
                    _count_no_mask+1
            if bboxes:
                if not self.check_image_dims(image_name=test_image_name):
                    self.make_labels(image_name=test_image_name, category_names=categories, bboxes=bboxes)
        return _count_mask, _count_no_mask

    def check_image_dims(self, image_name):
        file_name=os.path.join(self.mafa_base_dir, image_name)
        img = Image.open(file_name).convert("RGB")
        img_w, img_h = img.size
        if img_w < img_h:
            return True
        return False

    def make_labels(self, image_name, category_names, bboxes):
        # Process image
        file_image = os.path.splitext(image_name)[0]
        img = Image.open(os.path.join(self.mafa_base_dir, image_name)).convert("RGB")
        resize_img = img.resize(self.kitti_resize_dims)
        resize_img.save(os.path.join(self.kitti_images, file_image + '.jpg'), 'JPEG')
        # Process labels
        with open(os.path.join(self.kitti_labels, file_image + '.txt'), 'w') as label_file:
            for i in range(0, len(bboxes)):
                resized_bbox = self.resize_bbox(img=img, bbox=bboxes[i], dims=self.kitti_resize_dims)
                out_str = [category_names[i].replace(" ", "")
                           + ' ' + ' '.join(['0'] * 1)
                           + ' ' + ' '.join(['0'] * 2)
                           + ' ' + ' '.join([b for b in resized_bbox])
                           + ' ' + ' '.join(['0'] * 7)
                           + '\n']
                label_file.write(out_str[0])

    def resize_bbox(self, img, bbox, dims):
        img_w, img_h = img.size
        x_min, y_min, x_max, y_max = bbox
        ratio_w, ratio_h = dims[0] / img_w, dims[1]/img_h
        new_bbox = [str(int(np.round(x_min*ratio_w))), str(int(np.round(y_min*ratio_h))), str(int(np.round(x_max*ratio_w))), str(int(np.round(y_max *ratio_h)))]
        return new_bbox

    def mat2data(self):
        _count_mask, _count_no_mask = 0,0
        for i in range(0, self.len_dataset):
            _count_mask, _count_no_mask = self.extract_labels(i=i, train_flag=self.train,
                                                              _count_mask=_count_mask,
                                                              _count_no_mask=_count_no_mask)
        print("MAFA Dataset: Total Mask faces: {} and No-Mask faces:{}".format(_count_mask, _count_no_mask))
        return _count_mask, _count_no_mask

    def test_labels(self, file_name):
        img = Image.open(os.path.join(self.kitti_images, file_name + '.jpg'))
        text_file = open(os .path.join(self.kitti_labels, file_name + '.txt'), 'r')
        features = []
        bbox = []
        category = []
        for line in text_file:
            features = line.split()
            bbox.append([float(features[4]), float(features[5]), float(features[6]), float(features[7])])
            category.append(features[0])
        print("Bounding Box", bbox)
        print("Category:", category)
        i = 0
        for bb in bbox:
            cc = category[i]
            if cc == 'Mask':
                outline_box = 'red'
            elif cc == "No-Mask":
                outline_box = 'green'
            draw_img = ImageDraw.Draw(img)
            shape = ((bb[0], bb[1]), (bb[2], bb[3]))
            draw_img.rectangle(shape, fill=None, outline=outline_box)
            draw_img.text((bb[0], bb[1]), cc, (255,255,255))

            i+=1

        img.show()

def main():
    mafa_base_dir = r'C:\Users\ameykulkarni\Downloads\MAFA\MAFA'
    kitti_base_dir = r'C:\Users\ameykulkarni\Downloads\MAFA\KITTI_test'
    train = True
    if train:
        annotation_file = os.path.join(mafa_base_dir, 'MAFA-Label-Train/LabelTrainAll.mat')
        mafa_base_dir = os.path.join(mafa_base_dir, 'train-images\images')
    else:
        annotation_file = os.path.join(mafa_base_dir, 'MAFA-Label-Test/LabelTestAll.mat')
        mafa_base_dir = os.path.join(mafa_base_dir, 'test-images\images')

    category_limit = [25000, 25000] # Mask / No-Mask Limits
    kitti_resize_dims = (480, 272) # Look at TLT model requirements
    kitti_label = mafa2kitti(annotation_file=annotation_file, mafa_base_dir=mafa_base_dir,
                             kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims,
                             category_limit=category_limit, train=train)
    count_masks, count_no_masks = kitti_label.mat2data()
    kitti_label.test_labels(file_name='train_00006597')

if __name__ == '__main__':
    main()
