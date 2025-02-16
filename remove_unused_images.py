import os

def list_folders_in_directory(folder_path):
    try:
        if not os.path.exists(folder_path):
            print(f"Error: The path '{folder_path}' does not exist.")
            return []
        folders = [f for f in os.listdir(folder_path) 
                   if os.path.isdir(os.path.join(folder_path, f))]
        return folders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def list_files_in_folder(folder_path):
    try:
        if not os.path.exists(folder_path):
            print(f"Error: The path '{folder_path}' does not exist.")
            return []
        files = [f for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def removeDuplicateImages(transformed_filepath, transformed_folders, annotated_filepath, annotated_folders):
    for i in range(len(transformed_folders)):
        test_path = os.path.join(transformed_filepath, transformed_folders[i])
        annotated_files = list_files_in_folder(os.path.join(annotated_filepath, annotated_folders[i]))
        transformed_files = list_files_in_folder(test_path)
        transformed_test = [os.path.splitext(f)[0] for f in transformed_files]
        annotated_test = [os.path.splitext(f)[0] for f in annotated_files]
        for j in range(len(transformed_test)):
            if transformed_test[j] not in annotated_test:
                os.remove(os.path.join(test_path, transformed_files[j]))
                print(f"Removed: {transformed_files[j]}")
    return None


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data")
removeDuplicateImages(os.path.join(file_path, 'Transformed_images'),
                    list_folders_in_directory(os.path.join(file_path, 'Transformed_images')), 
                    os.path.join(file_path, 'annotations'),
                    list_folders_in_directory(os.path.join(file_path, 'annotations')))