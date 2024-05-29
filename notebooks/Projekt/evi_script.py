import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *


file_path = (r"data\processed\reference_data.csv")
housing_data = pd.read_csv(file_path)


housing_data = pd.DataFrame(housing_data)


columns_to_analyze = ['Temperatura (2m)', 'Relativna vlaga (2m)', 'Temperatura rosisca (2m)', 'Obcutna temperatura', 'Verjetnost padavin', 'Stevilo nesrec', 'Average speed', 'Free flow speed', 'Current travel time', 'Free flow travel time', 'Confidence', 'Road closure']



for column in columns_to_analyze:
    if housing_data[column].dtype == 'object':  
        housing_data[column] = pd.to_numeric(housing_data[column], errors='coerce')  
    housing_data[column + '_prediction'] = housing_data[column] + np.random.randn(housing_data.shape[0]) * 5





reference = housing_data.sample(n=min(50, len(housing_data)), replace=False)
reference = reference.drop(columns=['Cas'])

current = housing_data.sample(n=min(50, len(housing_data)), replace=False)
current = current.drop(columns=['Cas'])


report = Report(metrics=[
    DataDriftPreset(),
])

report.run(reference_data=reference, current_data=current)


report.save_html(r"data\processed\rift_test.html")

tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference, current_data=current)

tests.save_html(r"data\processed\stability_test.html")
