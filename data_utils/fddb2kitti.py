import scipy.io
import os
from PIL import Image
import math
import re
import numpy as np

class fddb2kitti():
    def __init__(self, annotation_path, fddb_base_dir, kitti_base_dir, kitti_resize_dims, category_limit):
        self.annot_path = annotation_path
        self.kitti_base_dir = kitti_base_dir
        self.fddb_base_dir = fddb_base_dir
        self.count_mask = category_limit[0]
        self.count_no_mask = category_limit[1]
        self.kitti_resize_dims = kitti_resize_dims
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

    def ellipese2bbox(self, face_annotations):
        major_axis_radius = int(float(face_annotations[0]))
        minor_axis_radius = int(float(face_annotations[1]))
        angle = int(float(face_annotations[2]))
        center_x = int(float(face_annotations[3]))
        center_y = int(float(face_annotations[4]))

        cosin = math.cos(math.radians(-angle))
        sin = math.sin(math.radians(-angle))

        x1 = cosin * (-minor_axis_radius) - sin * (-major_axis_radius) + center_x
        y1 = sin * (-minor_axis_radius) + cosin * (-major_axis_radius) + center_y
        x2 = cosin * (minor_axis_radius) - sin * (-major_axis_radius) + center_x
        y2 = sin * (minor_axis_radius) + cosin * (-major_axis_radius) + center_y
        x3 = cosin * (minor_axis_radius) - sin * (major_axis_radius) + center_x
        y3 = sin * (minor_axis_radius) + cosin * (major_axis_radius) + center_y
        x4 = cosin * (-minor_axis_radius) - sin * (major_axis_radius) + center_x
        y4 = sin * (-minor_axis_radius) + cosin * (major_axis_radius) + center_y

        '''pts = cv.ellipse2Poly((center_x, center_y), (major_axis_radius, minor_axis_radius), angle, 0, 360, 10)
        rect = cv.boundingRect(pts)'''
        x_cords = [x1, x2, x3, x4]
        y_cords = [y1, y2, y3, y4]
        x_min = min(x_cords)
        x_max = max(x_cords)
        y_min = min(y_cords)
        y_max = max(y_cords)
        left = x_min
        top = y_min
        right = x_max
        bottom = y_max
        width = right - left + 1
        height = bottom - top + 1
        return left, top, width, height

    def make_labels(self, image_name, category_names, bboxes):
        # Process image
        file_image = os.path.splitext(os.path.split(image_name)[1])[0]
        img = Image.open(os.path.join(self.fddb_base_dir, image_name)).convert("RGB")
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

    def fddb_data(self):
        _count_mask, _count_no_mask = 0, 0
        for root, dirs, files in os.walk(self.annot_path):
            for file in files:
                if file.endswith('ellipseList.txt'):
                    file_name = os.path.join(root, file)
                    _count_mask, _count_no_mask = self.mat2data(read_file=file_name,
                                                                _count_no_mask=_count_no_mask,
                                                                _count_mask = _count_mask)
        print("FDDB Dataset: Mask Labelled:{} and No-Mask Labelled:{}".format(_count_mask, _count_no_mask))
        return _count_mask, _count_no_mask

    def mat2data(self, read_file, _count_no_mask, _count_mask):
        # print("File Name: {}".format(read_file))
        strings = ("2002/", "2003/")
        with open(read_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                line = lines[i]
                if any(s in line for s in strings) and _count_no_mask < self.count_no_mask:
                    image_file_location = line.strip('\n')
                    num_faces_line = re.search(r"(\d+).*?", lines[i + 1])
                    num_faces = int(num_faces_line.group(1))
                    image_name = image_file_location + '.jpg'
                    category_name = 'No-Mask'
                    bboxes = []
                    category_names = []
                    for j in range(1, num_faces + 1):
                        annot_line = str(lines[i + j + 1])
                        faces = annot_line.split()
                        left, top, width, height = self.ellipese2bbox(face_annotations=faces[0:5])
                        bbox = [left, top, width+left, top+height]
                        bboxes.append(bbox)
                        category_names.append(category_name)
                    if bboxes:
                        self.make_labels(image_name=image_name, category_names=category_names, bboxes=bboxes)
                    _count_no_mask+=1

        return _count_mask, _count_no_mask

def main():
    fddb_base_dir = r'C:\Users\ameykulkarni\Downloads\FDDB-folds\originalPics'
    annotation_path = r'C:\Users\ameykulkarni\Downloads\FDDB-folds\FDDB-folds'
    kitti_base_dir = r'C:\Users\ameykulkarni\Downloads\MAFA\KITTI'

    category_limit = [1000, 1000] # Mask / No-Mask Limits
    kitti_resize_dims = (480, 272) # Look at TLT model requirements
    kitti_label = fddb2kitti(annotation_path=annotation_path, fddb_base_dir=fddb_base_dir,
                             kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims,
                             category_limit=category_limit)
    count_masks, count_no_masks = kitti_label.fddb_data()

if __name__ == '__main__':
    main()
