# End-to-End ML project from Chapter 2

import os
import tarfile
import requests

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")    # Target path
    response = requests.get(housing_url, stream=True)   # stream=True defers downloading the response body
    if response.status_code == 200:
        with open(tgz_path, 'wb') as f:     # open returns a file object f which has I/O methods and attributes
            f.write(response.raw.read())    # reads the raw response and writes to file
    housing_tgz = tarfile.open(tgz_path)    # opens tgz file (i.e. the raw response) at tgz_path (just a TAR file)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()    # Fetch the data





