"""
    This file performs One-Hot encoding over the original Monk Files.

    To understand the choice made for the encoding, checking this site:
    https://archive.ics.uci.edu/dataset/70/monk+s+problems
    at the 'Additional Variable Information' paragraph is particularly helpful.

    The datasets that we end with have:
    - the target in the first column;
    - encoded attributes in the remaining 17 columns;
    - IDs are not kept.
"""
import numpy as np
import pandas as pd



def one_hot_encode(row, attribute_sizes):
    """
        Since each attribute has a different size,
        we perform a singular encoding for each.
    """
    encoded = []
    for i, val in enumerate(row):
        one_hot = [0] * attribute_sizes[i]
        one_hot[val - 1] = 1
        encoded.extend(one_hot)
    return encoded

if __name__ == "__main__":
    # Input and output files dir
    input_file = "original_MonkFiles/monks-1.train"
    output_file = "encoded_MonkFiles/m1training.csv"

    # Reading file as df
    data = pd.read_csv(input_file, delim_whitespace=True, header=None)
    print(f"Encoding file: {input_file}")

    # Extracting target (1st column) and attributes (last column is ID)
    targets = data.iloc[:, 0].values
    attributes = data.iloc[:, 1:-1].values

    # Defining the size for each attribute
    attribute_sizes = [3, 3, 2, 3, 4, 2]

    # Applying One-Hot encoding
    encoded_data = [one_hot_encode(row, attribute_sizes) for row in attributes]
    print(f"One-Hot encoding completed.")

    # Creating the new df with encoded data:
    # stacks targets in the first column and the encoded attributes in the remaining
    encoded_df = pd.DataFrame(np.hstack([targets.reshape(-1, 1), encoded_data]))

    # Saving file
    encoded_df.to_csv(output_file, index=False, header=False)
    print(f"File saved as: {output_file}")