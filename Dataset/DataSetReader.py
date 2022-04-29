import csv
import pandas as pd

"""
Class Name: DataSetReader
Functionality: Class Method; Reads the posturalDatset.csv
Methods: getCsvDataset()
"""


class DataSetReader:
    """
    Method Name: getCsvDataset
    Functionality: Reads the csv files. Returns the Values and Replaces the Tag Identifier
    Return: pandas.core.frame.DataFrame format of the Dataset
    """

    @classmethod
    def getCsvDataset(cls):

        # The Tag Identifier Information is in the dataset/dataSetDescription.txt
        tag_identifier = {"010-000-024-033": "ANKLE_LEFT", "010-000-030-096": "ANKLE_RIGHT", "020-000-033-111": "CHEST",
                          "020-000-032-221": "BELT"}

        try:
            # Reading the File form the dataset/posturalDatset.csv directory
            with open('dataset/posturalDatset.csv', 'r') as f:
                reader = csv.DictReader(f)
                items = list(reader)

        except FileNotFoundError:
            print("Unable to Find the File, Check the File Path")

        except UnboundLocalError:
            print("Items is being Iterated with a Blank File, Check the File Path")

        # Replacing the Tag Numbers with the Tag_Identifier Dictionary else; None
        for item in items:
            item["Tag"] = tag_identifier.get(item["Tag"], None)

        dataFrame_Dataset = pd.DataFrame(items, index=None)
        return dataFrame_Dataset
