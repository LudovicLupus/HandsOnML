# End-to-End ML project from Chapter 2 (Hands-on ML with Scikit-Learn and TensorFlow - O'Riley)

import os
import tarfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit  # as sss

pd.options.display.width = 0    # Autodetect width of terminal window for displaying dataframes

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):     # Return True if path is an existing directory
        os.makedirs(housing_path)           # Create a directory named path
    tgz_path = os.path.join(housing_path, "housing.tgz")    # Target path
    response = requests.get(housing_url, stream=True)       # stream=True defers downloading the response body
    if response.status_code == 200:
        with open(tgz_path, 'wb') as f:     # open returns a file object f which has I/O methods and attributes
            f.write(response.raw.read())    # reads the raw response and writes to file
    housing_tgz = tarfile.open(tgz_path)    # opens tgz file (i.e. the raw response) at tgz_path (just a TAR file)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()    # Fetch the data

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()   # Load the housing data

# TAKE A QUICK LOOK AT THE DATA
print(housing.head(5))
print(housing.info())   # Quick description of the data (e.g. row count, non-null count, dtype, etc.)

# From the first few rows and pd.info, col "ocean_proximity" appears to be categorical
print(housing["ocean_proximity"].value_counts())    # View categories and respective counts

print(housing.describe())   # Quick description of numerical attributes

# housing.hist(bins=50, figsize=(20, 15))  # Creates histogram of numerical attributes

# SPLIT DATA INTO TRAINING AND TEST SETS (via random sampling)
# Too simple - don't use this method as it will return different sets each run
def split_train_test(data, test_ratio):
    shuffled_indicies = np.random.permutation(len(data))    # Returns a randomly permuted sequence
    test_set_size = int(len(data) * test_ratio)             # Define relative size of test set
    test_indicies = shuffled_indicies[:test_set_size]
    train_indicies = shuffled_indicies[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]  # Return train and test sets (integer-loc based indexing)


# Generate train and test sets (relative size determined by test_ratio input param)
train_set, test_set = split_train_test(housing, 0.2)
print(f"Length of train_set: {len(train_set)}")
print(f"Length of test_set: {len(test_set)}")

# Alternatively, you can use the hash of a unique and immutable identifier
# A hash function is any function that can be used to map data of arbitrary size to fixed-size values
# Here, the test_ratio will be relative to the maximum hash value
# Cyclic redundancy check (CRC) typically used for error-checking and hashing
def test_set_check(id, test_ratio):
    return crc32(np.int64(id)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# Since there is no ID column available, we create one based on row num
# If you use the row index as a unique identifier, new data must get appended to the end of the data set
# and existing rows cannot be removed. If not possible, use combinations of the most stable features
# housing_with_id = housing.reset_index()     # adds index column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# Splitting train/test using Scikit-Learn
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Create medium income category attribute (since we're told medium income is an important signal)
# These categories will let us preform stratified sampling further downstream via sklearn
housing["income_cat"] = np.ceil((housing["median_income"]/1.5))
print(f"Income categories: \n{housing['income_cat'].value_counts()}")   # View income category distribution

# Only keep categories less than 5, and merge those greater than 5 into cat 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)   # Last param changes data inplace (returns nothing)

# View distribution of new "income_cat" attribute
housing["income_cat"].hist()    # Distribution is much less "long-tailed" than original medium income field
# plt.show()  # Render plot

# Stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)      # Provides train/test indices to split data in train/test sets
for train_index, test_index in split.split(housing, housing["income_cat"]):     # split.split returns generator obj
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# View income cat proportions in the test set
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# Now that you have created your train/test sets, drop the income_cat attribute we created
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)   # Whether to drop labels from the index (0) or columns (1)

# Create copy of training set as to not hurt the original
housing = strat_train_set.copy()

# VISUALIZING GEOGRAPHICAL DATA
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)    # alpha controls transparency/opacity (shows densities here)















