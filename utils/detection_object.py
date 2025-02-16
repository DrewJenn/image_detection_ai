import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, img_folder, annotations_folder, transform=None):
        self.img_folder = img_folder
        self.annotations_folder = annotations_folder
        self.transform = transform
        self.annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
        self.class_map = self.build_class_map()

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        annotation_path = os.path.join(self.annotations_folder, self.annotation_files[idx])

        with open(annotation_path) as f:
            annotations = json.load(f)
    
        image_name = annotations[0]['image']
        image_path = os.path.join(self.img_folder, image_name)
        image = Image.open(image_path).convert("RGB")
    
        boxes = []
        labels = []
    
        img_width, img_height = image.size 
        print(image.size)
    
        for ann in annotations[0]['annotations']:
            label = ann['label']
            coords = ann['coordinates']

            x_min = coords['x']
            y_min = coords['y']
            x_max = x_min + coords['width']
            y_max = y_min + coords['height']
            
            x_min /= img_width
            y_min /= img_height
            x_max /= img_width
            y_max /= img_height

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.get_class_id(label))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {'boxes': boxes, 'labels': labels}
        print("Normalized Bounding Boxes:", boxes)


        return image, target

      

    def build_class_map(self):
        all_labels = set() 

        for annotation_file in self.annotation_files:
            annotation_path = os.path.join(self.annotations_folder, annotation_file)

            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
                for item in annotations:
                    for ann in item.get('annotations', []):
                        all_labels.add(ann['label'])
        
        class_map = {'background': 0}  
        class_map.update({label: idx + 1 for idx, label in enumerate(sorted(all_labels))})

        print(f"Class map: {class_map}")
        return class_map

    def get_class_id(self, label):
        return self.class_map.get(label, -1) 
    
    def get_num_classes(self):
        return len(self.class_map) - 1  

