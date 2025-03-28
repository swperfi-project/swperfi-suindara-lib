"""
SWPERFI Call Drop Processing Pipeline
Author: Pedro Matias
Date: 16/02/2025
License: Apache License 2.0

Description:
------------
This module implements an orchestrator for processing multiple ZIP files 
containing call drop logs. It integrates:
- Log parsing (via LogParser)
- Data processing (via DataProcessor)
- Call drop prediction (via PredictionPipeline)

The pipeline can handle single or multiple ZIPs and generates a summary report 
comparing parsed logs with predictions.
"""

import os
import pandas as pd
from tqdm import tqdm
from .parsing.data_processor import DataProcessor
from .prediction.predict import PredictionPipeline
from .utils.config import LoggerSetup
#from calldroploglib import DataProcessor, LogParser, PredictionPipeline, LoggerSetup



class CallDropPipeline:
    """
    Call drop processing pipeline that orchestrates log parsing, data processing, 
    and call drop prediction for single or multiple ZIP files.

    Attributes:
    -----------
    model_path : str
        Path to the trained model file (pickle format).
    logger : Logger
        Configured logger instance.
    summary : list
        Stores processing summary for multiple ZIPs.
    """

    def __init__(self, model_path: str):
        """
        Initializes the pipeline with the path to the trained model and optional logger.

        Parameters:
        -----------
        model_path : str
            Path to the trained model file (pickle format).
        """
        self.logger = LoggerSetup.setup_logger(self.__class__.__name__)
        self.model_path = model_path
        self.summary = []  
        self._data_processor = None  # Inicializado quando o ZIP for processado
        self._prediction_pipeline =  PredictionPipeline(self.model_path)

    
    @property
    def data_processor(self):
        """Getter para acessar o DataProcessor atual."""
        return self._data_processor

    @property
    def prediction_pipeline(self):
        """Getter para acessar o PredictionPipeline atual."""
        return self._prediction_pipeline
    
    def set_model_path(self, new_model_path: str):
        """
        Atualiza o caminho do modelo e reinicializa a PredictionPipeline.

        Parameters:
        -----------
        new_model_path : str
            Novo caminho para o arquivo de modelo (pickle format).
        """
        if self.model_path != new_model_path:
            self.logger.info(f"[MODEL UPDATE] Updating model path from '{self.model_path}' to '{new_model_path}'.")
            self.model_path = new_model_path
            self._prediction_pipeline = PredictionPipeline(self.model_path)
            self.logger.info(f"[MODEL UPDATE] PredictionPipeline reinitialized with the new model path: '{self.model_path}'.")



    def process_single_zip(self, zip_path: str, auto_save: bool = True, output_dir: str = None):
        """
        Processes a single ZIP file, executes parsing, data transformation, 
        and prediction, then saves the results.

        Parameters:
        -----------
        zip_path : str
            Path to the ZIP file containing call drop logs.
        output_dir : str
            Directory where the results will be saved.

        Returns:
        --------
        dict
            Summary of the processing and prediction results for the ZIP file.
        """
        try:
            zip_name = os.path.splitext(os.path.basename(zip_path))[0]
            self.logger.info(f"Processing ZIP: {zip_name}")

            # Step 1: Process logs and consolidate data
            self._data_processor = DataProcessor(zip_path) # Inicializa e armazena o DataProcessor
            if self._data_processor.consolidated_df.empty:
                self.logger.warning(f"ZIP {zip_name} generated no consolidated data.")
                result = {
                    "ZIP": zip_name,
                    "Status": "Empty after processing",
                    "Parsed Calls": 0,
                    "Predicted Calls": 0,
                    "Correct Predictions": 0,
                    "Local Accuracy": "0%"
                }
                self.summary.append(result)
                return result

            # Save consolidated DataFrame (using custom save method)
            if auto_save: 
                self._data_processor.save_to_csv()
                self.logger.info(f"Consolidated DF saved for {zip_name}")

            # Step 2: Run prediction pipeline
            self._prediction_pipeline = PredictionPipeline(self.model_path)
            self._prediction_pipeline.run_pipeline(self._data_processor.consolidated_df, zip_path)
          
            # Save prediction results (using custom save method)
            if auto_save: 
                self._prediction_pipeline.save_to_csv()
                self.logger.info(f"Prediction results saved for {zip_name}")

            # Prepare the result in the requested format
            result = {
                "ZIP": zip_name,
                "Status": "Success",
                "Parsed Calls": len(self._data_processor.consolidated_df),
                "Predicted Calls": self._prediction_pipeline.total_predictions,
                "Correct Predictions": self._prediction_pipeline.correct_predictions,
                "Local Accuracy": str(round(self._prediction_pipeline.accuracy)*100)+"%"
            }

            self.summary.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Unexpected error processing ZIP {zip_path}: {e}")
            result = {
                "ZIP": os.path.basename(zip_path),
                "Status": f"Error: {e}",
                "Parsed Calls": 0,
                "Predicted Calls": 0,
                "Correct Predictions": 0,
                "Local Accuracy": "0%"
            }
            self.summary.append(result)
            return result




    def process_multiple_zips(self, zip_dir: str):
        """
        Processes multiple ZIP files from a directory, executes parsing, 
        transformation, and prediction, and generates a summary report.

        Parameters:
        -----------
        zip_dir : str
            Directory containing multiple ZIP files.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the processing summary for all ZIPs.
        """
        output_dir = os.path.join(zip_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        zip_files = [os.path.join(zip_dir, f) for f in os.listdir(zip_dir) if f.endswith(".zip")]
        if not zip_files:
            self.logger.warning("No ZIP files found in the directory.")
            return []

        self.logger.info(f"Processing {len(zip_files)} ZIP files in directory: {zip_dir}")

        self.progress = 0  # Atributo de progresso iniciado em 0
        self.total_files = len(zip_files)  # Número total de arquivos

        # Processar todos os ZIPs e atualizar o progresso
        for idx, zip_path in enumerate(zip_files, start=1):
            result = self.process_single_zip(zip_path,True, output_dir)
            self.summary.append(result)
            self.progress = int((idx / self.total_files) * 100)  # Atualizar o progresso percentual

        return self.summary
        
    
    
    def to_df(self):
        """
        Converts the summary list to a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            The summary converted to a DataFrame.
        """
        return pd.DataFrame(self.summary)

    def save_to_csv(self, output_path: str):
        """
        Saves the summary as a CSV file by first converting it to a DataFrame.

        Parameters:
        -----------
        output_path : str
            The file path where the CSV will be saved.
        """
        df = self.to_df()
        df.to_csv(output_path, index=False)


