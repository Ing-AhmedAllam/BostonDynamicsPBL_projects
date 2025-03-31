import os
import pandas as pd

def load_dataset(csv_file_path: str):
    """
    Loads and parses a dataset from a CSV file.
    This function reads a CSV file containing geometric data, validates its structure, 
    and converts each row into a structured dictionary format. The dataset must contain 
    specific required columns: "Figure", "origin_x", "origin_y", "major2x", "major2y", 
    "minor2x", and "minor2y".
    Args:
        csv_file_path (str): The file path to the CSV file.
    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a parsed row 
        from the dataset. Each dictionary contains:
            - "figure" (str): The figure name or identifier.
            - "origin" (list[float]): The [x, y] coordinates of the origin.
            - "major" (dict): A dictionary with:
                - "start" (list[float]): The [x, y] coordinates of the major axis start point.
                - "end" (list[float]): The [x, y] coordinates of the major axis end point.
            - "minor" (dict): A dictionary with:
                - "start" (list[float]): The [x, y] coordinates of the minor axis start point.
                - "end" (list[float]): The [x, y] coordinates of the minor axis end point.
    Raises:
        AssertionError: If the file path is None, the file is not a CSV, or the file does not exist.
        Exception: If there is an error while reading the CSV file.
    Notes:
        - If the dataset is missing required columns or has an incorrect format, an empty list is returned.
        - If an error occurs while loading the dataset, an empty list is returned.
    """
    
    assert csv_file_path is not None, "CSV file path is None"
    assert csv_file_path.endswith('.csv'), "File is not a CSV"
    assert os.path.exists(csv_file_path), "CSV file does not exist"
    
    #Load Dataset
    try:
        dataset = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    required_columns = ["Figure", "origin_x", "origin_y", "major2x", "major2y", "minor2x", "minor2y"]
    if not all(col in dataset.columns for col in required_columns):
        print("Dataset format incorrect or missing required columns")
        return []
    
    # Convert each row into a structured dictionary
    parsed_data = []
    for _, row in dataset.iterrows():
        entry = {
            "figure": row["Figure"],
            "origin": [row["origin_x"], row["origin_y"]],
            "major": {
                "start": [row["origin_x"], row["origin_y"]],
                "end": [row["major2x"], row["major2y"]]
            },
            "minor": {
                "start": [row["origin_x"], row["origin_y"]],
                "end": [row["minor2x"], row["minor2y"]]
            }
        }
        parsed_data.append(entry)

    return parsed_data