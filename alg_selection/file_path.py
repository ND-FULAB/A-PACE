#to obtain the file list to save time for alg_select.py

import os
import json

def generate_file_list(root_path):
    folder_file_dict = {}
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            file_list = [file for file in os.listdir(folder_path) if file.endswith('.pssession')]
            folder_file_dict[folder] = file_list
    return folder_file_dict

def save_file_list_to_json(folder_file_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(folder_file_dict, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    os.chdir('../')
    root_directory =os.getcwd()  # Change this to your root directory path
    output_json_path = 'folder_file_list.json'

    folder_file_list = generate_file_list(root_directory)
    os.chdir(root_directory)
    os.chdir('alg_selection_final_UPM_SRCRE_0.65_R2')
    save_file_list_to_json(folder_file_list, output_json_path)
    print(f"File list saved to {output_json_path}")
