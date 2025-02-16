import subprocess 
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data')
command = [('cd ' + os.path.join(file_path, 'launch_package')) + '\nmake qt5py3' + '\npython3 labelImg.py' ]
try:
    result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Output of {' '.join(command)}:\n{result.stdout}")
    if result.stderr:
        print(f"Errors from {' '.join(command)}:\n{result.stderr}")
except subprocess.CalledProcessError as e:
    print(f"Command failed: {e}")


