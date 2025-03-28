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
import numpy as np
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
        self.threshold = 0.5
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
                self.logger.debug(isinstance(model_data, dict))

            # Verifica se o modelo é do tipo dicionário (caso XGBoost)
            if isinstance(model_data, dict):
                self.logger.info("[MODEL LOAD] Model file is a XGBoost.")
                model = model_data.get("model")
                required_features = model_data.get("feature_names")
            else:
                self.logger.info("[MODEL LOAD] Model file is a Catboost.")
                # Caso CatBoost, o modelo é carregado diretamente
                model = model_data
                # Para CatBoost, tentamos acessar 'feature_names_' se existir
                required_features = getattr(model, 'feature_names_', [])

            if model is None or required_features is None:
                self.logger.error("[MODEL LOAD] Model file is missing 'model' or 'feature_names'.")
                return None, None

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
        """Filters rows by technology (LTE or NR) provided by signal parameters."""
        allowed_tech = ['LTE', 'NR']
        if 'Dominant_Technology' in data.columns:
            tech_col = 'Dominant_Technology'
        elif 'disconnectRAT' in data.columns:
            tech_col = 'disconnectRAT'
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
            
        time_columns = ['persist_time', 'voice_reg_time', 'signal_strength_time']
        filtered_data = filtered_data.dropna(subset=time_columns, how='any')
        filtered_data = filtered_data.astype({
            col: 'int64' for col in ['ci', 'tac', 'channel','band','signal_strength_at_end','mcc','mnc'] if col in data.columns
        })
            
        return filtered_data
    
       
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms data by creating necessary columns."""
        data = data.copy()
        model_type = type(self.model).__name__

        # Criação correta de 'rsrp'
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

        # Criação correta de 'rsrq'
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

        # Criação correta de 'sinr_snr'
        if 'sinr_snr' not in data.columns:
            if 'Dominant_Technology' in data.columns:
                mask_lte = data['Dominant_Technology'] == 'LTE'
                if 'LTE_rssnr' in data.columns:
                    data.loc[mask_lte, 'sinr_snr'] = data.loc[mask_lte, 'LTE_rssnr']
                    self.logger.info("[TRANSFORMATION] Created numeric 'sinr_snr' for LTE events from 'LTE_rssnr'.")
                mask_nr = data['Dominant_Technology'] == 'NR'
                if 'NR_ssSinr' in data.columns:
                    data.loc[mask_nr, 'sinr_snr'] = data.loc[mask_nr, 'NR_ssSinr']
                    self.logger.info("[TRANSFORMATION] Created numeric 'sinr_snr' for NR events from 'NR_ssSinr'.")
            else:
                mask_lte = data['rat'] == 'LTE'
                if 'LTE_rssnr' in data.columns:
                    data.loc[mask_lte, 'sinr_snr'] = data.loc[mask_lte, 'LTE_rssnr']
                    self.logger.info("[TRANSFORMATION] Created numeric 'sinr_snr' for LTE events from 'LTE_rssnr' (using rat).")
                mask_nr = data['rat'] == 'NR'
                if 'NR_ssSinr' in data.columns:
                    data.loc[mask_nr, 'sinr_snr'] = data.loc[mask_nr, 'NR_ssSinr']
                    self.logger.info("[TRANSFORMATION] Created numeric 'sinr_snr' for NR events from 'NR_ssSinr' (using rat).")

        # disconnectRAT_mapped apenas para CatBoost
        if model_type == 'CatBoostClassifier':
            if 'disconnectRAT_mapped' not in data.columns and 'disconnectRAT' in data.columns:
                rat_mapping = {
                    13: 'LTE', 14: 'EHRPD', 15: 'HSPAP', 16: 'GSM',
                    17: 'TD_SCDMA', 18: 'IWLAN', 19: 'LTE_CA', 20: 'NR'
                }
                data['disconnectRAT_mapped'] = data['disconnectRAT'].map(rat_mapping).astype('category')
                self.logger.info("[TRANSFORMATION] 'disconnectRAT_mapped' created by mapping 'disconnectRAT'.")

               
        # Criação de 'plmn'
        if 'plmn' not in data.columns and 'mcc' in data.columns and 'mnc' in data.columns:
            data['plmn'] = (
                data['mcc'].astype(str) +
                data['mnc'].astype(str)
            ).astype('category')
            self.logger.info("[TRANSFORMATION] 'plmn' created by combining 'mcc' and 'mnc'.")
    


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
        Validates and adjusts the data types of the required features in the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to validate and adjust.

        Returns:
        --------
        bool
            True if all required features are present and adjusted, False otherwise.
        """
        model_type = type(self.model).__name__
        missing_features = set(self.required_features) - set(df.columns)

        if missing_features:
            self.logger.error("[VALIDATION] Missing required features: %s", missing_features)
            return False

        if model_type == 'CatBoostClassifier':
            expected_types = {
                'ci': 'category',
                'tac': 'category',
                'channel': 'category',
                'plmn': 'category',
                'band': 'category',
                'rsrp': 'float64',
                'rsrq': 'float64',
                'sinr_snr': 'float64',
                'disconnectRAT_mapped': 'category',
                'signal_strength_at_end': 'int32',
                'hour_of_day': 'category',
                'day_of_week': 'category'
            }
        elif model_type == 'XGBClassifier':
            expected_types = {
                'day_of_week': 'int32',
                'hour_of_day': 'int32',
                'plmn': 'category',
                'activeRAT': 'int64',
                'disconnectRAT': 'int64',
                'channel': 'category',
                'band': 'category',
                'rsrp': 'float64',
                'rsrq': 'float64'
            }
        else:
            self.logger.warning(f"[VALIDATION] Model type {model_type} not recognized for type adjustment.")
            return True # if model not recognized, only check for feature presence.

        try:
            df = df.astype(expected_types)
            self.logger.info("[VALIDATION] All required features have been adjusted to correct data types.")
            return True
        except Exception as e:
            self.logger.error(f"[VALIDATION] Error adjusting data types: {e}")
            return False

    
    
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
            predictions = self.model.predict_proba(df)[:, 1]
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

                # Save prediction probabilities to DataFrame
                df_full['call_drop_probability'] = predictions

                # Create boolean column for call drop prediction based on threshold
                df_full['predicted_call_drop'] = (df_full['call_drop_probability'] >= self.threshold)

                # Criar coluna de acertos (1 se correto, 0 se incorreto)
                df_full['correct_prediction'] = (
                    (df_full['predicted_call_drop'] == True) & (df_full['IDTAG'] == 'CALL_DROP') |
                    (df_full['predicted_call_drop'] == False) & (df_full['IDTAG'] == 'CALL_SUCCESS')
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
                summary_data =[]

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
                        'call_drop_probability': pred,
                        'predicted_call_drop': bool(df_full.loc[idx, 'predicted_call_drop']),
                        'IDTAG_reference': df_full.loc[idx, 'IDTAG'] if 'IDTAG' in df_full.columns else 'N/A',
                        'disconnect_cause': df_full.loc[idx, 'cause'] if 'cause' in df_full.columns else 'N/A',
                        'correct_prediction': bool(df_full.loc[idx, 'correct_prediction'])
                    })
                    summary_data.append(entry)

                    # Logging de cada predição
                    self.logger.info("[RESULT] call_id: %s - disc_time: %s - call_drop_probability: %.4f - predicted_call_drop: %s - IDTAG: %s",
                                    entry['call_id'], entry['disc_time'], pred, entry['predicted_call_drop'], entry['IDTAG_reference'])

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

        
    def generate_local_shap_explanation(self, input_instance):
        """
        Gera uma explicação local SHAP para uma instância de entrada com customização nativa.

        Parameters:
        -----------
        input_instance : pd.DataFrame
            Uma instância (linha) para a qual a explicação SHAP será gerada.

        Returns:
        --------
        tuple
            Um dicionário com os valores SHAP e uma figura matplotlib pronta para ser renderizada no PySide.
        """
        # Instanciar o SHAP Explainer com o modelo treinado
        explainer = shap.Explainer(self.model)

        # Calcular os valores SHAP para a instância
        shap_values = explainer(input_instance)

        # Definir tamanho do gráfico com base no número de features
        num_features = len(input_instance.columns)
        fig_width = 10
        fig_height = max(6, num_features * 0.5)  # Altura proporcional ao número de features

        # Configuração inicial do tema escuro
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Gera o gráfico waterfall diretamente na nova figura
        shap.plots.waterfall(shap_values[0], show=False)

        # Customizações visuais do gráfico
        ax.set_facecolor('#2a2d3e')  # Fundo dos eixos
        fig.patch.set_facecolor('#1e1e2f')  # Fundo da figura
        ax.grid(True, linestyle='--', color='#4c5c74', alpha=0.3)  # Linhas de grade com transparência

        # Títulos e labels
        ax.set_title('Local SHAP Explanation', fontsize=14, color='#c0d6f7')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.tight_layout()  # Ajuste automático do layout

        # Converte os valores SHAP em dicionário
        shap_values_dict = dict(zip(input_instance.columns, shap_values.values[0]))

        return shap_values_dict, fig

        
       

        

