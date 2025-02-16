import os
from PIL import Image

resize_to = (800, 800)

def resize_images(input_file_path, output_file_path):

    for filename in os.listdir(input_file_path):
        file_path = os.path.join(input_file_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(file_path)
                resized_img = img.resize(resize_to)
                resized_img.save(os.path.join(output_file_path, filename))
                os.remove(file_path)
                print(f"Resized and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    print("Resizing complete!")


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data")
print("Please input the name of the folder we are transforming: ")
folder_name = input()
if not os.path.exists(os.path.join(os.path.join(file_path, 'Transformed_Images'), folder_name)):
    os.makedirs(os.path.join(os.path.join(file_path, 'Transformed_Images'), folder_name))
    os.makedirs(os.path.join(os.path.join(file_path, 'annotations'), folder_name))

resize_images(os.path.join(os.path.join(file_path, 'images'), folder_name),
            os.path.join(os.path.join(file_path, 'Transformed_Images'), folder_name))
