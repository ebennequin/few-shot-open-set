from pathlib import Path
import pickle

from easyfsl.data_tools import EasySet
from loguru import logger
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import typer


def save_tiered_imagenet(
    specs_file: Path = Path("data/tiered_imagenet/specs/test.json"),
    save_dir: Path = Path("tiered_imagenet_tests"),
    image_size: int = 84,
    use_bgr: bool = False,
    interpolation: str = "bilinear",
):
    """
    Save the tiered imagenet dataset to the save_dir.
    """
    print(use_bgr, interpolation)
    # Get dataset
    dataset = EasySet(specs_file)
    dataset.transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode[interpolation.upper()],
            ),
            transforms.ToTensor(),
        ]
    )

    logger.info(f"Treating images...")
    all_labels = []
    all_images = []
    for image, label in tqdm(dataset):
        all_labels.append(label)
        if use_bgr:
            transformed_image = image[[2, 1, 0], :].permute(1, 2, 0).numpy()
        else:
            transformed_image = image.permute(1, 2, 0).numpy()
        all_images.append(transformed_image)

    # Create save_dir
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save labels
    labels_file = save_dir / f"{specs_file.stem}_labels.pkl"
    logger.info(f"Saving labels to {labels_file}...")
    with open(labels_file, "wb") as f:
        pickle.dump({"labels": all_labels}, f)

    # Save images
    images_file = save_dir / f"{specs_file.stem}_images.npz"
    logger.info(f"Saving images to {images_file}...")
    np.savez(images_file, images=np.stack(all_images))


if __name__ == "__main__":
    typer.run(save_tiered_imagenet)
