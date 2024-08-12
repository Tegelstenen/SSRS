import pytest
import pandas as pd

from datetime import datetime
import os

from modules.update_data import _get_querying_dates, _call_pipeline_script, _load_new_data

@pytest.fixture(scope="module")
def data():
    data = {
        "date": ["2023-02-01", "2023-02-02", "2023-02-03", "2023-03-01", "2024-08-01"]
    }
    df = pd.DataFrame(data)
    return df


def test_get_querying_dates(data):
    querying_dates = _get_querying_dates(data)
    todays_date = datetime.now().date().strftime('%Y-%m-%d')
    assert querying_dates == ("2024-08-01", todays_date)
    
def test_call_pipeline_script(data):
    querying_dates = _get_querying_dates(data)
    _call_pipeline_script(querying_dates[0], querying_dates[1], "dashboard/modules")
    assert os.path.isfile("../data.csv")
    os.remove("../data.csv")
    
def test_load_new_data(data):
    new_data = _load_new_data(data)
    check_file = os.path.isfile("../data.csv")
    assert not check_file, "The csv file should be deleted after loaded"
    assert isinstance(new_data, pd.DataFrame), "The data should be loaded into a pandas data frame"
    
    existing_df = pd.read_csv("./data/data.csv", nrows=1)
    existing_columns = set(existing_df.columns)
    new_columns = set(new_data.columns)
    assert existing_columns == new_columns, "The features in both dataframes should be the same."
    