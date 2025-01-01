import json
import os
import torch
from matplotlib import pyplot as plt
from skimage.draw import polygon
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.label_id_map = None
        map_path = os.path.join(root_dir, "Label_ID.json")
        with open(map_path, "r") as file:
            self.label_id_map = json.load(file)

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Collect all image and mask paths recursively
        self.images_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.gtFine_dir = os.path.join(root_dir, "gtFine", split)
        self.image_paths = self._get_all_files_images(self.images_dir)
        self.polygon_paths = self._get_all_files_polygons(self.gtFine_dir)

        # Ensure image and mask paths match
        self.image_paths.sort()
        self.polygon_paths.sort()
        self.image_paths, self.polygon_paths = self.match_paths(
            self.image_paths,
            self.polygon_paths
        )

        assert len(self.image_paths) == len(self.polygon_paths), \
            "Mismatch between images and masks!"

    def _get_all_files_images(self, directory):
        """Recursively collect all file paths in the given directory."""
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):  # Adjust extension as needed
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def _get_all_files_polygons(self, directory):
        """Recursively collect all file paths in the given directory."""
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):  # Adjust extension as needed
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def match_paths(self, list1, list2):
        """
    Remove unmatched paths from two lists by matching the first three parts of the filenames.
    """
        # Extract the first three parts of the filename as the identifier
        def extract_identifier(path):
            filename = os.path.basename(path)
            # Split by underscore and join the first three parts
            return "_".join(filename.split("_")[:3])

        # Create mappings from identifiers to paths
        id_to_path1 = {extract_identifier(path): path for path in list1}
        id_to_path2 = {extract_identifier(path): path for path in list2}

        # Find common identifiers
        common_ids = set(id_to_path1.keys()).intersection(set(id_to_path2.keys()))

        # Filter paths based on common identifiers
        matched_list1 = [id_to_path1[common_id] for common_id in common_ids]
        matched_list2 = [id_to_path2[common_id] for common_id in common_ids]

        return matched_list1, matched_list2

    def __len__(self):
        return len(self.image_paths)

    def convert_polygons(self, polygons):
        label_id_map = self.label_id_map
        imgHeight = polygons["imgHeight"]
        imgWidth = polygons["imgWidth"]
        mask = np.zeros((imgHeight, imgWidth), dtype=np.uint8)

        objects = polygons["objects"]
        for object in objects:
            polygon_vertices = object["polygon"]
            label = object["label"]
            id = label_id_map[label]
            rr, cc = polygon([v[1] for v in polygon_vertices], [v[0] for v in polygon_vertices], mask.shape)

            # Update the pixels inside the polygon to a specific value (e.g., 255)
            mask[rr, cc] = id

        return mask

    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.image_paths[idx]
        polygon_path = self.polygon_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = np.zeros_like(np.array(image)[:, :, 0])


        with open(polygon_path, "r") as file:
            polygons = json.load(file)
            mask = self.convert_polygons(polygons)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image).transpose((2, 0, 1)), dtype=torch.float32)

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
