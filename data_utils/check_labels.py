import os
from PIL import Image, ImageDraw

def test_labels(kitti_base_dir, file_name):
    img = Image.open(os.path.join(kitti_base_dir+'images/', file_name + '.jpg'))
    text_file = open(os.path.join(kitti_base_dir+'labels/', file_name + '.txt'), 'r')
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
        draw_img = ImageDraw.Draw(img)
        shape = ((bb[0], bb[1]), (bb[2], bb[3]))
        if category[i] == 'No-Mask':
            outline_clr = "red"
        elif category[i] == 'Mask':
            outline_clr = "green"
        draw_img.rectangle(shape, fill=None, outline=outline_clr, width=4)
        i += 1
    img.show()