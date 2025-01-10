from typing import List
import csv

def read_csv_file(file_path:str)->List[list]:
    """A function to read a csv file 

    Args:
        file_path (str): path to the file

    Returns:
        List[list]: a list of list indicating all of the cells in the csv
    """
    data = []
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data