"""
SWPERFI Call Drop Prediction Pipeline
Author: Pedro Matias
Date: 10/02/2025
License: Apache License 2.0

Description:
------------
This module implements an object-oriented prediction pipeline for call drop analysis.
It loads a trained model, prepares data, and runs predictions on the consolidated dataset.
"""

import os
import pickle
import pandas as pd
import logging
import shap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..utils.config import LoggerSetup


class PredictionPipeline:
    """
    A class for handling call drop predictions.

    Attributes:
    -----------
    model_path : str
        Path to the trained model file (pickle format).
    logger : Logger
        Configured logger instance.
    model : object
        Loaded machine learning model.
    required_features : list
        List of required features for prediction.
    """

    def __init__(self, model_path: str):
        """
        Initializes the PredictionPipeline, loading the model and required features.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file (pickle format).
        """
        self.logger = LoggerSetup.setup_logger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO) 
        self.model_path = model_path
        self.log_zip_file_path = None
        self.model, self.required_features = self.load_model()
        self.total_predictions = None 
        self.correct_predictions = None
        self.accuracy  = None
        self.full_predicted_df = pd.DataFrame()
        self.summary_predicted_df = pd.DataFrame()

    def load_model(self):
        """
        Loads the trained model and required features from a pickle file.

        Returns:
        --------
        tuple
            (model, required_features) if successful, otherwise (None, None).
        """
        try:
            with open(self.model_path, 'rb') as file:
                model_data = pickle.load(file)

            # Carrega modelo e features
            model = model_data.get("model")
            required_features = model_data.get("feature_names")

            if model is None or required_features is None:
                self.logger.error("[MODEL LOAD] Model file is missing 'model' or 'feature_names'.")
                return None, None

            # Atribuição de atributos
            self.model = model
            self.required_features = required_features

            # Extração de feature importance automaticamente
            if hasattr(self.model, 'feature_importances_') and hasattr(self.model, 'feature_names_in_'):
                self.feature_importance_dict = dict(
                    zip(self.model.feature_names_in_, self.model.feature_importances_)
                )
                self.logger.info("[MODEL LOAD] Feature importances extracted successfully with %d features.", len(self.feature_importance_dict))
            else:
                self.feature_importance_dict = {}
                self.logger.warning("[MODEL LOAD] Model does not have feature importances or feature names.")

            self.logger.info("[MODEL LOAD] Model and features loaded successfully from '%s'.", self.model_path)
            self.logger.info("[MODEL LOAD] Required features: %s", required_features)

            return model, required_features

        except Exception as e:
            self.logger.error("[MODEL LOAD] Error loading model from '%s': %s", self.model_path, e)
            return None, None
        
    def get_feature_importance_dict(self):
        """
        Returns the preloaded feature importance dictionary.

        Returns:
        --------
        dict
            A dictionary of feature names and their corresponding importances.
        """
        self.logger.info("[FEATURE IMPORTANCE] Retrieving preloaded feature importances.")
        return self.feature_importance_dict if hasattr(self, 'feature_importance_dict') else {}


        
    def filter_by_technology(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filters rows by technology (LTE or NR)."""
        allowed_tech = ['LTE', 'NR']
        if 'Dominant_Technology' in data.columns:
            tech_col = 'Dominant_Technology'
        elif 'activeRAT' in data.columns:
            tech_col = 'activeRAT'
        else:
            self.logger.info("[FILTERING] Neither 'Dominant_Technology' nor 'activeRAT' column found in the data.")
            raise ValueError("Missing technology column in data.")
        
        initial_count = len(data)
        filtered_data = data[data[tech_col].isin(allowed_tech)].copy()
        dropped_count = initial_count - len(filtered_data)
        
        if dropped_count > 0:
            self.logger.info("[FILTERING] Dropped %d rows that are not LTE or NR based on '%s'.", dropped_count, tech_col)
        else:
            self.logger.info("[FILTERING] All rows are LTE or NR based on '%s'.", tech_col)
        return filtered_data
    
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms data by creating necessary columns."""
        data = data.copy()
        # Create 'rsrp'
        if 'rsrp' not in data.columns:
            if 'Dominant_Technology' in data.columns:
                mask_lte = data['Dominant_Technology'] == 'LTE'
                if 'LTE_rsrp' in data.columns:
                    data.loc[mask_lte, 'rsrp'] = data.loc[mask_lte, 'LTE_rsrp']
                    self.logger.info("[TRANSFORMATION] Created 'rsrp' for LTE events from 'LTE_rsrp'.")
                mask_nr = data['Dominant_Technology'] == 'NR'
                if 'NR_ssRsrp' in data.columns:
                    data.loc[mask_nr, 'rsrp'] = data.loc[mask_nr, 'NR_ssRsrp']
                    self.logger.info("[TRANSFORMATION] Created 'rsrp' for NR events from 'NR_ssRsrp'.")
            else:
                mask_lte = data['rat'] == 'LTE'
                if 'LTE_rsrp' in data.columns:
                    data.loc[mask_lte, 'rsrp'] = data.loc[mask_lte, 'LTE_rsrp']
                    self.logger.info("[TRANSFORMATION] Created 'rsrp' for LTE events from 'LTE_rsrp' (using rat).")
                mask_nr = data['rat'] == 'NR'
                if 'NR_rsrp' in data.columns:
                    data.loc[mask_nr, 'rsrp'] = data.loc[mask_nr, 'NR_rsrp']
                    self.logger.info("[TRANSFORMATION] Created 'rsrp' for NR events from 'NR_rsrp' (using rat).")
        
        # Create 'rsrq'
        if 'rsrq' not in data.columns:
            if 'Dominant_Technology' in data.columns:
                mask_lte = data['Dominant_Technology'] == 'LTE'
                if 'LTE_rsrq' in data.columns:
                    data.loc[mask_lte, 'rsrq'] = data.loc[mask_lte, 'LTE_rsrq']
                    self.logger.info("[TRANSFORMATION] Created 'rsrq' for LTE events from 'LTE_rsrq'.")
                mask_nr = data['Dominant_Technology'] == 'NR'
                if 'NR_ssRsrq' in data.columns:
                    data.loc[mask_nr, 'rsrq'] = data.loc[mask_nr, 'NR_ssRsrq']
                    self.logger.info("[TRANSFORMATION] Created 'rsrq' for NR events from 'NR_ssRsrq'.")
            else:
                mask_lte = data['rat'] == 'LTE'
                if 'LTE_rsrq' in data.columns:
                    data.loc[mask_lte, 'rsrq'] = data.loc[mask_lte, 'LTE_rsrq']
                    self.logger.info("[TRANSFORMATION] Created 'rsrq' for LTE events from 'LTE_rsrq' (using rat).")
                mask_nr = data['rat'] == 'NR'
                if 'NR_rsrq' in data.columns:
                    data.loc[mask_nr, 'rsrq'] = data.loc[mask_nr, 'NR_rsrq']
                    self.logger.info("[TRANSFORMATION] Created 'rsrq' for NR events from 'NR_rsrq' (using rat).")
        
        # Create 'plmn'
        if 'plmn' not in data.columns and 'mcc' in data.columns and 'mnc' in data.columns:
            data['plmn'] = data['mcc'].astype(str) + data['mnc'].astype(str)
            self.logger.info("[TRANSFORMATION] Created 'plmn' column by combining 'mcc' and 'mnc'.")
        return data
    
    def remove_invalid_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes rows with missing or invalid (2147483647) values in signal columns."""
        mask_missing = data[self.required_features].isnull().any(axis=1)
        mask_placeholder = (data['rsrp'] == 2147483647) | (data['rsrq'] == 2147483647)
        combined_mask = mask_missing | mask_placeholder
        removed = data[combined_mask]
        data_clean = data[~combined_mask].copy()
        num_removed = len(removed)
        if num_removed > 0:
            if 'call_id' in removed.columns:
                removed_ids = removed['call_id'].unique()
                self.logger.info("[DATA CLEANING] Removed %d rows due to missing or invalid values (2147483647). Removed call_ids: %s", num_removed, removed_ids)
            else:
                self.logger.info("[DATA CLEANING] Removed %d rows due to missing or invalid values (2147483647).", num_removed)
        else:
            self.logger.info("[DATA CLEANING] No rows removed due to missing or invalid values (2147483647).")
        return data_clean

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validates if the required features are present in the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to validate.

        Returns:
        --------
        bool
            True if all required features are present, False otherwise.
        """
        missing_features = set(self.required_features) - set(df.columns)
        if missing_features:
            self.logger.error("[VALIDATION] Missing required features: %s", missing_features)
            return False
        self.logger.info("[VALIDATION] All required features are present.")
        return True

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepares data for prediction by selecting required features.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the call drop data.

        Returns:
        --------
        pd.DataFrame
            The cleaned and prepared DataFrame for prediction.
        """
        self.logger.info("[PREPARATION] Starting data preparation for prediction.")
        df = self.filter_by_technology(df)
        df = self.transform_data(df)
        df = self.remove_invalid_rows(df)

        if not self.validate_features(df):
            raise ValueError("Missing required features for prediction.")
        

        df_prepared = df[self.required_features].copy()

        for col in df_prepared.columns:
            if df_prepared[col].dtype == 'object':
                df_prepared[col] = df_prepared[col].astype('category')
                self.logger.info("[PREPARATION] Converted column '%s' to categorical.", col)

        self.logger.info("[PREPARATION] Data is ready for prediction.")
        return df_prepared,df

    def predict(self, df: pd.DataFrame):
        """
        Runs prediction using the model.

        Parameters:
        -----------
        df : pd.DataFrame
            The prepared DataFrame with required features.

        Returns:
        --------
        list
            List of predicted probabilities for call drops.
        """
        try:
            predictions = list(zip(*self.model.predict_proba(df)))[1]
            self.logger.info("[PREDICTION] Prediction executed successfully.")
            self.logger.info("[PREDICTION] predictions: %s", predictions)
            return predictions

        except Exception as e:
            self.logger.error("[PREDICTION] Error during prediction: %s", e)
            return None
        



    def run_pipeline(self, consolidated_df: pd.DataFrame, zip_file_path:str):
        """
        Runs the full pipeline: data preparation, prediction, and logging results.

        Parameters:
        -----------
        consolidated_df : pd.DataFrame
            The DataFrame containing the consolidated call drop data.
        """
        if consolidated_df.empty or consolidated_df is None:
            self.logger.error("[MAIN] Consolidated DataFrame is empty. Aborting prediction.")
            return
        try:
            df_prepared, df_full = self.prepare_data(consolidated_df) # Preparar os dados antes de rodar a previsão
        except ValueError as ve: self.logger.error("[MAIN] Error preparing data: %s", ve)

        try:
            predictions = self.predict(df_prepared)
            if predictions is None:
                self.logger.error("[MAIN] Prediction failed.")
                return 
            else:
                if self.log_zip_file_path == None:
                    self.log_zip_file_path = zip_file_path
                    
                # Save predictions to DataFrame
                df_full['prediction'] = predictions

                # Criar coluna de acertos (1 se correto, 0 se incorreto)
                df_full['correct_prediction'] = (
                    ((df_full['prediction'] >= 0.5) & (df_full['IDTAG'] == 'CALL_DROP')) |
                    ((df_full['prediction'] < 0.5) & (df_full['IDTAG'] == 'CALL_SUCCESS'))
                ).astype(int)

                # Cálculo da acurácia
                self.total_predictions = len(df_full)
                self.correct_predictions = df_full['correct_prediction'].sum()
                self.accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0

                # Logging dos resultados
                self.logger.info("[PREDICTION] Total predictions: %d", self.total_predictions)
                self.logger.info("[PREDICTION] Correct predictions: %d", self.correct_predictions)
                self.logger.info("[PREDICTION] Accuracy: %.4f", self.accuracy)

                # Criando DataFrame resumido
                summary_data = []

                # Log results
                for idx, pred in zip(df_prepared.index, predictions):
                    call_id = df_full.loc[idx, 'call_id'] if 'call_id' in df_full.columns else 'N/A'
                    disc_time = df_full.loc[idx, 'disc_time'] if 'disc_time' in df_full.columns else 'N/A'
                    idtag = df_full.loc[idx, 'IDTAG'] if 'IDTAG' in df_full.columns else 'N/A'

                    self.logger.info("[RESULT] call_id: %s - disc_time: %s - prediction: %.4f - IDTAG: %s",
                                    call_id, disc_time, pred, idtag)
                    
                for idx, pred in zip(df_prepared.index, predictions):
                    entry = {
                        'call_id': df_full.loc[idx, 'call_id'] if 'call_id' in df_full.columns else 'N/A',
                        'disc_time': pd.to_datetime(df_full.loc[idx, 'disc_time']).strftime('%m-%d %H:%M:%S') if 'disc_time' in df_full.columns else 'N/A',
                    }
                    # Adiciona features requeridas
                    for feature in self.required_features:
                        entry[feature] = df_full.loc[idx, feature] if feature in df_full.columns else 'N/A'
                    entry.update({
                        'prediction': pred,
                        'IDTAG_reference': df_full.loc[idx, 'IDTAG'] if 'IDTAG' in df_full.columns else 'N/A',
                        'disconnect_cause': df_full.loc[idx, 'cause'] if 'cause' in df_full.columns else 'N/A',
                        'correct_prediction': bool(df_full.loc[idx, 'correct_prediction'])
                    })
                    summary_data.append(entry)

                    # Logging de cada predição
                    self.logger.info("[RESULT] call_id: %s - disc_time: %s - prediction: %.4f - IDTAG: %s",
                                    entry['call_id'], entry['disc_time'], pred, entry['IDTAG_reference'])

                # Atributos de predição
                self.full_predicted_df = df_full
                self.summary_predicted_df = pd.DataFrame(summary_data)

                self.logger.info("[MAIN] Prediction pipeline completed successfully.")

                return self.full_predicted_df        
        except Exception as e: self.logger.exception("MAIN", "Unexpected error: %s", e)



    def save_to_csv(self, save_path: str = None):
        """
        Saves the predicted DataFrame (`self.full_predicted_df`) to a CSV file inside 
        a 'prediction_results' folder. The filename is automatically generated based on 
        the model file name.

        Parameters:
        -----------
        save_path : str, optional
            The base directory where the 'prediction_results' folder will be created.
            If None, it saves in the same directory as the model file.
        """
        if self.full_predicted_df.empty:
            self.logger.warning("[SAVE_CSV] Predicted DataFrame is empty. Skipping saving.")
            return

        # Se `save_path` não for fornecido, usar o diretório onde o modelo está armazenado
        if save_path is None:
            save_path = os.path.dirname(self.log_zip_file_path)

        # Criar diretório prediction_results dentro do caminho escolhido
        output_dir = os.path.join(save_path, "prediction_results")
        os.makedirs(output_dir, exist_ok=True)

        # Gerar nome do arquivo baseado no modelo carregado
        zip_name = os.path.basename(self.log_zip_file_path).replace(".zip", "")
        output_file = os.path.join(output_dir, f"{zip_name}_predictions.csv")

        # Salvar DataFrame
        try:
            self.full_predicted_df.to_csv(output_file, index=False)
            self.logger.info(f"[SAVE_CSV] Predicted DataFrame saved successfully: {output_file}")
        except Exception as e:
            self.logger.error(f"[SAVE_CSV] Error saving Predicted DataFrame to CSV: {e}")

    
    def save_summary_to_csv(self, save_path: str = None):
        """
        Saves the predicted DataFrame (`self.summary_predicted_df`) to a CSV file inside 
        a 'prediction_results' folder. The filename is automatically generated based on 
        the model file name.

        Parameters:
        -----------
        save_path : str, optional
            The base directory where the 'prediction_results' folder will be created.
            If None, it saves in the same directory as the model file.
        """
        if self.summary_predicted_df.empty:
            self.logger.warning("[SAVE_SUMMARY_CSV] Summary Predicted DataFrame is empty. Skipping saving.")
            return

        # Se `save_path` não for fornecido, usar o diretório onde o modelo está armazenado
        if save_path is None:
            save_path = os.path.dirname(self.log_zip_file_path)

        # Criar diretório prediction_results dentro do caminho escolhido
        output_dir = os.path.join(save_path, "prediction_results")
        os.makedirs(output_dir, exist_ok=True)

        # Gerar nome do arquivo baseado no modelo carregado
        zip_name = os.path.basename(self.log_zip_file_path).replace(".zip", "")
        output_file = os.path.join(output_dir, f"{zip_name}_summary_predictions.csv")

        # Salvar DataFrame
        try:
            self.summary_predicted_df.to_csv(output_file, index=False)
            self.logger.info(f"[SAVE_SUMMARY_CSV] Summary Predicted DataFrame saved successfully: {output_file}")
        except Exception as e:
            self.logger.error(f"[SAVE_SUMMARY_CSV] Error saving Summary Predicted DataFrame to CSV: {e}")

        
    def generate_local_shap_explanation(self, input_instance: pd.DataFrame):
        """
        Generates a local SHAP explanation for a specific instance and returns a ready-to-use plot.

        Parameters:
        -----------
        input_instance : pd.DataFrame
            A single-row DataFrame corresponding to the instance to explain.

        Returns:
        --------
        tuple
            (shap_values_dict, matplotlib.figure.Figure) ready for PySide6 visualization.
        """
        if self.model is None:
            self.logger.error("[SHAP] Model must be loaded before generating explanations.")
            return None, None

        if input_instance.shape[0] != 1:
            self.logger.error("[SHAP] Input must be a single instance (one row DataFrame).")
            return None, None

        # Criar o explicador SHAP
        explainer = shap.Explainer(self.model)
        shap_values = explainer(input_instance)

        # Criar gráfico sem exibir diretamente
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False, ax=ax)  # Gera o gráfico sem exibir

        # Preparar o retorno dos valores SHAP
        shap_values_dict = dict(zip(input_instance.columns, shap_values.values[0]))

        # Retorna o gráfico pronto para PySide6 e os valores SHAP
        return shap_values_dict, fig

        
       

        

