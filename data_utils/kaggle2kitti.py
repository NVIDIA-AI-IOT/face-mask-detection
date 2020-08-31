from xml.etree import ElementTree
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

class kaggle2kitti():
    def __init__(self, images_dir, labels_dir, kitti_base_dir, kitti_resize_dims, category_limit):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.count_mask = category_limit[0]
        self.count_no_mask = category_limit[1]
        self.kitti_base_dir = kitti_base_dir
        self.kitti_resize_dims = kitti_resize_dims
        try:
            os.makedirs(self.kitti_base_dir+'/train/images',mode=0o777)
        except:
            print("Directory Already Exists")
        self.kitti_images = os.path.join(self.kitti_base_dir, 'train/images')
        try:
            os.makedirs(self.kitti_base_dir+ '/train/labels',mode=0o777)
        except:
            print("Directory Already Exists")
        self.kitti_labels = os.path.join(self.kitti_base_dir, 'train/labels')
    def get_image_metafile(self, image_file):
        image_name = os.path.splitext(image_file)[0]
        return os.path.join(self.labels_dir, str(image_name+'.xml'))

    def make_labels(self, image_name, category_names, bboxes):
        # Process image
        file_image = os.path.splitext(image_name)[0]
        img = Image.open(os.path.join(self.images_dir, image_name)).convert("RGB")
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
        ratio_w, ratio_h = dims[0] / img_w, dims[1] / img_h
        new_bbox = [str(int(np.round(x_min * ratio_w))), str(int(np.round(y_min * ratio_h))), str(int(np.round(x_max * ratio_w))),
                    str(int(np.round(y_max * ratio_h)))]
        return new_bbox

    def get_data_attributes(self):
        image_extensions = ['.jpeg', '.jpg', '.png']
        _count_mask = 0
        _count_no_mask = 0
        for image_name in os.listdir(self.images_dir):
            if image_name.endswith('.jpeg') or image_name.endswith('.jpg') or image_name.endswith('.png'):
                labels_xml = self.get_image_metafile(image_file=image_name)
                if os.path.isfile(labels_xml):
                    labels = ElementTree.parse(labels_xml).getroot()
                    bboxes = []
                    categories = []
                    for object_tag in labels.findall("object"):
                        cat_name = object_tag.find("name").text

                        if (cat_name == 'mask'):
                            category = 'Mask'
                            xmin = int(object_tag.find("bndbox/xmin").text)
                            xmax = int(object_tag.find("bndbox/xmax").text)
                            ymin = int(object_tag.find("bndbox/ymin").text)
                            ymax = int(object_tag.find("bndbox/ymax").text)
                            bbox = [xmin, ymin, xmax, ymax]
                            categories.append(category)
                            bboxes.append(bbox)
                            _count_mask += 1
                        elif cat_name == 'none':
                            category = 'No-Mask'
                            xmin = int(object_tag.find("bndbox/xmin").text)
                            xmax = int(object_tag.find("bndbox/xmax").text)
                            ymin = int(object_tag.find("bndbox/ymin").text)
                            ymax = int(object_tag.find("bndbox/ymax").text)
                            bbox = [xmin, ymin, xmax, ymax]
                            categories.append(category)
                            bboxes.append(bbox)
                            _count_no_mask += 1
                    if bboxes:
                        self.make_labels(image_name=image_name, category_names=categories, bboxes=bboxes)
        print("Kaggle Dataset: Total Mask faces: {} and No-Mask faces:{}".format(_count_mask, _count_no_mask))
        return _count_mask, _count_no_mask


def main():
    images_dir = '/home/nvidia/face-mask-detection/datasets/medical-masks-dataset/images'
    labels_dir = '/home/nvidia/face-mask-detection/datasets/medical-masks-dataset/labels'
    kitti_base_dir = '/home/nvidia/face-mask-detection/datasets/medical-masks-dataset/KITTI_1024'
    kitti_resize_dims = (960,544)
    category_limit = [10,10]
    medical_mask2kitti = kaggle2kitti(images_dir=images_dir, labels_dir=labels_dir,
                                      category_limit=category_limit,
                                      kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims)
    medical_mask2kitti.get_data_attributes()


if __name__ == '__main__':
    main()
