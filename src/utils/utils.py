import os
import sys
import shutil


def check_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(os.listdir(save_dir)) != 0:
        proceed = input('Experiment directory not empty, continue? [y/n]: ')
        if proceed != 'y':
            print('Aborted')
            sys.exit()
        print("Cleaning Experiment directory")
        delete_files_from_dir(save_dir)


def delete_files_from_dir(dir):
    for root, dirs, files in os.walk(dir, topdown=False):
        # Delete files
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file: {file_path}: {e}")
        
        # Delete directories
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")
            except Exception as e:
                print(f"Error deleting directory: {dir_path}: {e}")


def remove_files_with_prefix(dir, prefix):
    for filename in os.listdir(dir):
        if filename.startswith(prefix):
            os.remove(os.path.join(dir, filename))
