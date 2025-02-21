import torch
import torch.nn as nn
from . import detection_object as detection
from torchvision import transforms


def createNewModel(data_loader, backbone, device, optimizer):
    backbone.to(device)
    for image, target in data_loader:
        image = image.to(device).float()  
        target_classes = target['labels'].squeeze(1).to(device)  
        target_boxes = target['boxes'].to(device).squeeze(1) 
        optimizer.zero_grad()
        pred_class_logits, pred_bbox_preds = backbone(image)
        spaceoutLogic()  
        print("Raw Predictions:")
        print("Class Logits:", pred_class_logits)
        print("Bounding Box Predictions:", pred_bbox_preds)
        spaceoutLogic()
        pred_bbox_preds = pred_bbox_preds.clamp(min=0) 
        loss = compute_loss(pred_class_logits, pred_bbox_preds, target_classes, target_boxes)
        loss.backward()
        optimizer.step()
        print(f"Output from Backbone: {pred_class_logits.shape}")
        print(f"Target (annotations) - Boxes shape: {target_boxes.shape}, Labels shape: {target_classes.shape}")
        print(f"Loss: {loss.item()}")
        spaceoutLogic()
    
    return backbone


def denormalize_coords(coords, img_width, img_height):
    x_min, y_min, x_max, y_max = coords
    x_min = x_min * img_width
    y_min = y_min * img_height
    x_max = x_max * img_width
    y_max = y_max * img_height
    return [x_min, y_min, x_max, y_max]


def compute_loss(pred_class_logits, pred_bbox_preds, target_class, target_bbox):
    print("Ground Truth Bounding Boxes:")
    print(target_bbox)
    print("Predicted Bounding Boxes (Normalized):")
    print(pred_bbox_preds)
    spaceoutLogic()
    class_loss = nn.CrossEntropyLoss()(pred_class_logits, target_class)
    spaceoutLogic()
    bbox_loss = nn.SmoothL1Loss()(pred_bbox_preds, target_bbox)
    total_loss = class_loss + bbox_loss

    print(f"Classification Loss: {class_loss.item()}, BBox Loss: {bbox_loss.item()}")
    print('Total Loss: ' + str(total_loss.item()))
    
    return total_loss



def mainTrainingLoop(model, data_loader, device, optimizer, num_epochs=10):
    model.train()  
    for epoch in range(num_epochs):
        epoch_loss = 0
        for image, target in data_loader:
            image = image.to(device).float()  
            target_classes = target['labels'].squeeze(1).to(device) 
            target_boxes = target['boxes'].to(device).squeeze(1) 

            optimizer.zero_grad()  
            pred_class_logits, pred_bbox_preds = model(image)

            spaceoutLogic() 
            print("Raw Predictions:")
            print("Class Logits:", pred_class_logits)
            print("Bounding Box Predictions:", pred_bbox_preds)

            class_probs = torch.softmax(pred_class_logits, dim=-1)
            print(f"Predicted Class Probabilities (Softmax): {class_probs}")

            pred_bbox_preds = pred_bbox_preds.clamp(min=0)
            loss = compute_loss(pred_class_logits, pred_bbox_preds, target_classes, target_boxes)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(data_loader)}")

    return model



def getDetector(image_folder_path, annotations_folder_path):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(800, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])   
    return detection.CustomObjectDetectionDataset(
        img_folder=image_folder_path,
        annotations_folder=annotations_folder_path,
        transform=transform
    )


def spaceoutLogic():
    print('\n\n')



def file_selection(file_path):
    try:
        folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
    except FileNotFoundError:
        print(f"The directory {file_path} was not found.")
        return None
    except PermissionError:
        print(f"Permission denied to access {file_path}.")
        return None
        
    print("please slect the corresponding number for the training dataset you wish to use.")
    for i in range(len(folders)):
        print(str((i + 1)) + ": " + folders[i])
    user_selection = int(input())
    return (folders[user_selection - 1])
