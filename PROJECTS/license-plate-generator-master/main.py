#python3 /Users/hossein/Desktop/CV/PROJECTS/license-plate-generator-master/main.py --noise --dirt --transform --fill --blur --translate


import argparse
import os
import psutil
import _jsonnet
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from plate_generator import PlateGenerator
from transformations import perspective_transform
from asset_manager import AssetManager
from pascal_voc import bounding_rects_to_xml

project_config_path = '/Users/hossein/Desktop/CV/PROJECTS/license-plate-generator-master/project_configurations.jsonnet'
project_config = _jsonnet.evaluate_file(project_config_path)
assets = AssetManager(project_config)

parser = argparse.ArgumentParser(description='Reading input arguments.')
parser.add_argument('--count', default=assets.generator_config['num_out_img'], type=int)
parser.add_argument('--outdir', default=assets.generator_config['output_directory'], type=str)
parser.add_argument('--package', default=assets.generator_config['img_per_package'], type=int)
parser.add_argument('--noise', default=assets.generator_config['apply_misc_noise'], action=argparse.BooleanOptionalAction)
parser.add_argument('--dirt', default=assets.generator_config['apply_dirt'], action=argparse.BooleanOptionalAction)
parser.add_argument('--transform', default=assets.generator_config['apply_transform'], action=argparse.BooleanOptionalAction)
parser.add_argument('--fill', default=assets.generator_config['fill_background'], action=argparse.BooleanOptionalAction)
parser.add_argument('--blur', action=argparse.BooleanOptionalAction)  # Added blur option
parser.add_argument('--translate', action=argparse.BooleanOptionalAction)  # Added translate option
args = parser.parse_args()

shutil.rmtree(args.outdir, ignore_errors=True)
os.makedirs(args.outdir)
annotation_path = ''
images_path = ''
xmls_path = ''
package_counter = 0
print(f'\ngenerating {args.count} images.')
progress = tqdm(range(args.count))

def apply_translation(image, max_translation=10):
    rows, cols, _ = image.shape
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

for index in progress:
    plate_generator = PlateGenerator(assets)
    plate, annotation = plate_generator.get_rnd_plate(apply_misc_noise=args.noise,
                                                      apply_dirt=args.dirt)
    if args.transform:
        plate, annotation = perspective_transform(plate, annotation, assets.transformations_config)
    if args.fill:
        plate = plate_generator.fill_background(plate)
    
    if args.blur:
        plate = cv2.GaussianBlur(plate, (7, 7), 0) 
    
    if args.translate:
        plate = apply_translation(plate)

    if index % args.package == 0:
        current_directory = os.path.join(args.outdir, f'{package_counter:02}')
        os.mkdir(current_directory)

        annotation_path = os.path.join(current_directory, 'anns')
        os.mkdir(annotation_path)
        xmls_path = os.path.join(annotation_path, 'xmls')
        os.mkdir(xmls_path)
        images_path = os.path.join(current_directory, 'imgs')
        os.mkdir(images_path)
        package_counter += 1

    cv2.imwrite(os.path.join(images_path, f'{index:05}.jpg'), plate)
    cv2.imwrite(os.path.join(annotation_path, f'{index:05}.png'), annotation)

for index in range(package_counter):
    input_address = os.path.join(args.outdir, f'{index:02}/anns')
    bounding_rects_to_xml(input_address + '/*.png', os.path.join(input_address, 'xmls'), assets.annotations_config)
