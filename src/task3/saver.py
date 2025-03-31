import os

def get_incremented_filename(base_name, extension=".png"):
    """
    Generate a unique, incremented filename in the 'results/task_2' directory.
    This function checks if a file with the specified base name and extension
    already exists in the 'results/task_2' directory. If it does, it appends
    an incrementing index to the base name until a unique filename is found.
    If the directory does not exist, it is created automatically.
    Args:
        base_name (str): The base name for the file (without extension).
        extension (str, optional): The file extension, default is ".png".
    Returns:
        str: A unique file path with an incremented filename if necessary.
    """
    
    # Go back to the previous directory and then into the 'tests' folder
    save_dir = os.path.join(os.getcwd(), "results/task_2")
    os.makedirs(save_dir, exist_ok=True)  # Create 'results' folder if it doesn't exist

    index = 1
    new_name = os.path.join(save_dir, f"{base_name}{extension}")
    
    while os.path.exists(new_name):
        new_name = os.path.join(save_dir, f"{base_name}_{index}{extension}")
        index += 1
    
    return new_name
