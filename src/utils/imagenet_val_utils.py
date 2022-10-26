from pathlib import Path
from loguru import logger
import xmltodict


# def list_imagenet_val(ilsvrc_root_dir: Path):
#     # image_paths = sorted((ilsvrc_root_dir / "Data/CLS-LOC/val").glob("*"))
#     # annotations = sorted((ilsvrc_root_dir / "Annotations/CLS-LOC/val").glob("*"))
#     image_paths = sorted((ilsvrc_root_dir / "val").glob("*"))
#     class_names = []
#     for annotation in annotations:
#         with open(annotation, "r") as f:
#             current_annotation = xmltodict.parse(f.read())["annotation"]["object"]
#             if type(current_annotation) == list:
#                 class_name = current_annotation[0]["name"]
#             else:
#                 class_name = current_annotation["name"]
#         class_names.append(class_name)

#     return image_paths, class_names


def list_imagenet_val(ilsvrc_root_dir: Path):
    # image_paths = sorted((ilsvrc_root_dir / "Data/CLS-LOC/val").glob("*"))
    # annotations = sorted((ilsvrc_root_dir / "Annotations/CLS-LOC/val").glob("*"))
    image_paths = sorted((ilsvrc_root_dir / "val").glob("**/*.JPEG"))
    class_names = []
    for img_path in image_paths:
        class_name = img_path.parts[-2]
        class_names.append(class_name)

    return image_paths, class_names
