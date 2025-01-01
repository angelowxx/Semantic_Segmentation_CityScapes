import json
import os


class Label_Id_Maps_Extracter():
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Collect all image and mask paths recursively
        self.dir = os.path.join(root_dir, "gtFine", split)
        self.polygons_paths = self._get_all_files(self.dir)

        # Ensure image and mask paths match
        self.polygons_paths.sort()

        self.maps = {}

    def _get_all_files(self, directory):
        """Recursively collect all file paths in the given directory."""
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):  # Adjust extension as needed
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def __len__(self):
        return len(self.polygons_paths)

    def __getitem__(self, idx):
        # Load image and mask
        polygons_path = self.polygons_paths[idx]
        maps = self.maps


        with open(polygons_path, "r") as file:
            polygons = json.load(file)
            objects = polygons["objects"]
            for object in objects:
                label = object["label"]

                if label not in maps:
                    maps[label] = len(maps)


    def extract_id_color_maps(self):
        json_file_path = os.path.join(self.root_dir, "Label_ID.json")
        length = self.__len__()
        for i in range(length):
            self.__getitem__(i)

        with open(json_file_path, "w") as json_file:
            json.dump(self.maps, json_file, indent=4)  # Use indent=4 to format the JSON with indentation

if __name__ == '__main__':
    data_dir = json_file_path = os.path.join(os.getcwd(), "dataset")
    label_id_maps_extracter = Label_Id_Maps_Extracter(data_dir)
    label_id_maps_extracter.extract_id_color_maps()