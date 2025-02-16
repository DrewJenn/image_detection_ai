import torch
import os
import utils.main_helper_functions as helper
from torch.utils.data import DataLoader
from models.backbone import CustomBackboneWithHead
from pathlib import Path


print("Please input the folder name of the model we are currently working with: ")
model_folder = input()

script_dir = Path(__file__).parent  

image_folder = script_dir / "data" / "Transformed_Images"
annotations_folder = script_dir / "data" / "annotations"
trained_model = script_dir / "trained_models"
model_filename = model_folder + ".pth"


myDetector = helper.getDetector(os.path.join(image_folder, model_folder), os.path.join(annotations_folder, model_folder))
data_loader = DataLoader(myDetector, batch_size=1, shuffle=True)
for image, target in data_loader:
    print("Image shape:", image.shape)
    print("Target shape:", target['labels'].shape)  
    break  
backbone = CustomBackboneWithHead(myDetector.get_num_classes() + 1)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-4)  


if not os.path.exists(os.path.join(trained_model, model_folder)):
    os.makedirs(os.path.join(trained_model, model_folder))
    model = helper.createNewModel(data_loader, backbone, device, optimizer)

    
else:
    model = torch.load(os.path.join(os.path.join(trained_model, model_folder), model_filename), weights_only=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  
    try:
        print("How many training cycles would you like to run?")
        num_epochs = input()
        model = helper.mainTrainingLoop(model, data_loader, device, optimizer, int(num_epochs))
    except:
        model = helper.mainTrainingLoop(model, data_loader, device, optimizer)


torch.save(model, os.path.join(os.path.join(trained_model, model_folder), model_filename))  

