[![](https://img.shields.io/pypi/v/swperfi-suindara?style=for-the-badge)](https://pypi.org/project/swperfi-suindara) [![](https://img.shields.io/pypi/l/swperfi-suindara?style=for-the-badge)](https://github.com/swperfi-project/swperfi-suindara/blob/main/LICENSE) [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/swperfi-project/swperfi-suindara) [![](https://img.shields.io/badge/-Documentation-fe9c22?style=for-the-badge&link=https%3A%2F%2Fyour_documentation_url)](https://your_documentation_url)

# swperfi-suindara

`swperfi-suindara` is a Python library developed for the **processing and analysis of Android logs** generated after **Call Drop** events. The package provides modules dedicated to:

- **Parsing**: Extracting and processing relevant information from Android AP logs.
- **Prediction**: Making predictions based on the processed data using a pre-trained XGBoost model.
- **Model Details**: Accuracy: 68.12%, Precision: 32.01%, Recall: 48.93%, F1-Score: 38.70%, 
    > Features: [`day_of_week`, `hour_of_day`, `plmn`, `channel`, `band`, `activeRAT`, `disconnectRAT`, `rsrp`, `rsrq`]

This package is part of the  research results from SWPerfI project, which applies advanced **AI techniques** to analyze the performance of software running on mobile devices, particularly through the use of log analysis for call drop prediction and network optimization.

> Note: The **prediction model** is not preloaded within this library. Instead, the model will be available from a separate repository and can be loaded as needed by users. This approach ensures that the library remains lightweight and flexible.

## About SWPERFI Project

The **SWPERFI** project focuses on applying cutting-edge AI techniques to **optimize mobile software performance**. This includes using data mining, machine learning, and deep learning to analyze performance metrics and identify the root causes of software inefficiencies. Within the **SWPERFI** project, the **Intelligent Hardware (IH)** subgroup specifically focuses on the impact of embedded system hardware on software performance.

The objective is to develop tools that enhance the prediction, mitigation, and analysis of issues that affect mobile systems' performance, such as call drops, by leveraging AI models trained on device-side data.

## Installation

To install the library, use the following command:

```bash
pip install swperfi-suindara

```

Alternatively, for installing from the repository:

```
git clone https://github.com/swperfi-project/swperfi-suindara
cd swperfi-suindara
pip install --upgrade pip
pip install .

```

Dependencies
------------

-   pandas >= 2.1.4
-   tqdm >= 4.66.5
-   python-dateutil >= 2.9.0

Usage
-----

### Summary

1.  **DataProcessor**: Responsible for **parsing** Android call drop logs from a single or multiple ZIP files. It outputs processed logs and consolidated data.
2.  **PredictionPipeline**: Uses the parsed data to run predictions, returning the predicted results such as accuracy and the number of correct predictions.
3.  **CallDropPipeline**: A full pipeline that combines **parsing** and **prediction** in one call, handling both **single** and **multiple ZIP files** with a trained model.

Here's how to use the library for **parsing** and **prediction** from **single** and **multiple ZIP** files:

#### 1\. **Using DataProcessor for Parsing a Single ZIP File**

The `DataProcessor` is responsible for parsing a single ZIP file containing AP Android logs (needs main and radio buffers). This is typically the first step before making predictions.

```python
from swperfi_suindara.parsing import DataProcessor

# Path to the single ZIP file containing the AP Android logs
zip_file_path = 'path_to_single_zip_file.zip'

# Instantiate the DataProcessor
dp = DataProcessor(zip_path)


# Display parsed data history, reports, and the consolidated dataframe
print(dp.history)  # Parsed logs history
print(dp.log_parser.report_info)  # Parsing report
print(dp.consolidated_df)  # Consolidated dataframe with processed data

# Optionally save the parsed data to a CSV file for further use
dp.save_to_csv('parsed_data.csv')

```

#### 2\. **Prediction Using a Pretrained Model for a Single ZIP File**

Once the data is parsed, you can use the `PredictionPipeline` to make predictions based on the processed data.

```python
from swperfi_suindara.prediction import PredictionPipeline

...

# Path to the trained model
model_path = 'path_to_your_trained_model/xgboost_model_Android_14.pkl'

# Instantiate the PredictionPipeline
pp = PredictionPipeline(model_path)

# Run prediction pipeline using the processed data from DataProcessor
pp.run_pipeline(dp.consolidated_df, dp.log_parser.zip_file_path)

# Display the prediction results
print(pp.predicted_df)  # Display the predictions
print(f"Total Predictions: {pp.total_predictions}")
print(f"Correct Predictions: {pp.correct_predictions}")
print(f"Accuracy: {pp.accuracy}")

# Optionally save the prediction results to a CSV file
pp.save_to_csv('predicted_data.csv')

```

#### 3\. **Using CallDropPipeline for a Single ZIP File (Full Pipeline)**

The `CallDropPipeline` integrates the **parsing** and **prediction** steps in one function call for a single ZIP file.

```python
from swperfi_suindara import CallDropPipeline

# Initialize the CallDropPipeline with the trained model
pipeline = CallDropPipeline(model_path=r"path_to_your_trained_model/xgboost_model_Android_14.pkl")

# Process a single ZIP file (parsing + prediction)
result_single = pipeline.process_single_zip(r"path_to_single_zip_file.zip")

# Display the result (parsed data + predictions)
print(result_single)

```


#### 4\. **Using CallDropPipeline for Multiple ZIP Files (Full Pipeline)**

For processing multiple ZIP files, you can use the `CallDropPipeline` to process and make predictions for all files in a directory.

```python
from swperfi_suindara import CallDropPipeline

# Initialize the CallDropPipeline with the trained model
pipeline = CallDropPipeline(model_path=r"path_to_your_trained_model/xgboost_model_Android_14.pkl")

# Process multiple ZIP files in a directory (parsing + prediction)
result_multiple = pipeline.process_multiple_zips(r"path_to_directory_with_zips")

# Display the result summary for all processed files
print(result_multiple)

```



Acknowledgments
---------------

The author are part of the **ALGOX research group** at **CNPq** (Algorithms, Optimization, and Computational Complexity) and the **Postgraduate Program in Computer Science (PPGI)** at **ICOMP/UFAM**.

This python package, part of the **SWPERFI - Artificial Intelligence Techniques for Analysis and Optimization of Software Performance** project, was supported by **Motorola Mobility Comércio de Produtos Eletrônicos Ltda** and **Flextronics da Amazônia Ltda** under Agreement No. 004/2021 with **ICOMP/UFAM**, in accordance with **Federal Law No. 8.387/1991**. Additional support was provided by Brazilian agencies **CAPES** (Finance Code 001), **CNPq**, and **FAPEAM** through the **POSGRAD project 2024/2025**.

Authors
-------

-   Pedro Victor dos Santos Matias: Master's Student, PPGI-UFAM

Citation
--------

To cite this project in your research or papers, please use the following reference (to be updated):

```
@software{Matias_2025_SWPERFI,
  author = {Pedro Victor dos Santos Matias},
  title = {swperfi-suindara: A Python Library for Android Call Drop Parsing and Prediction},
  url = {https://github.com/swperfi-project/swperfi-suindara},
  year = {2025},
  doi = {10.5281/zenodo.14057065}  # To be updated
}

```

License
-------

swperfi-suindara is licensed under the **Apache License, Version 2.0**.

```
Copyright 2025 SWPerfI - Universidade Federal do Amazonas (UFAM)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```

## More Information

For more information about the SWPERFI project, visit the official website:

[SWPERFI - Artificial Intelligence Techniques for Software Performance Analysis and Optimization](https://swperfi.icomp.ufam.edu.br)

© SWPERFI - Artificial Intelligence Techniques for Software Performance Analysis and Optimization


