from glob import glob
import xmltodict
import os
import shutil

if __name__ == '__main__':
    DATA_ROOT = "/pasteur/u/esui/data/imagenet/val"
    ANNOTATION_ROOT = '/pasteur/u/esui/data/extra_imagenet_stuff/Annotations/val'

    annotations = glob(f'{ANNOTATION_ROOT}/*')

    for ann_file in annotations:
        with open(ann_file) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
        
        img_filename = data_dict['annotation']['filename'] + ".JPEG"

        try:
            class_id = data_dict['annotation']['object']['name']
        except:
            class_id = data_dict['annotation']['object'][0]['name'] # get first object class

        class_folder = os.path.join(DATA_ROOT, class_id)
        os.makedirs(class_folder, exist_ok=True)

        filename_in_class_folder = os.path.join(class_folder, img_filename)

        img_filename = os.path.join(DATA_ROOT, img_filename)

        if os.path.exists(img_filename):
            dest = shutil.move(img_filename, filename_in_class_folder)
            assert dest == filename_in_class_folder
            print(f"Moved {img_filename} to {dest}")
        else:
            print(f"Skip move {img_filename} to {filename_in_class_folder}")
    



