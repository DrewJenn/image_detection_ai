import subprocess 
import os

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
image_path = os.path.join(file_path, 'Transformed_Images')
annotation_path = os.path.join(file_path, 'annotations')

print("Please input the name of the AI we are annotating the data for: ")
ai_project = input()

command = [('cd ' + os.path.join(file_path, 'launch_package')) + '\nmake qt5py3' + '\npython3 labelImg.py' + ' ' + os.path.join(image_path, ai_project)]     
try:
    result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Output of {' '.join(command)}:\n{result.stdout}")
    if result.stderr:
        print(f"Errors from {' '.join(command)}:\n{result.stderr}")
except subprocess.CalledProcessError as e:
    print(f"Command failed: {e}")


