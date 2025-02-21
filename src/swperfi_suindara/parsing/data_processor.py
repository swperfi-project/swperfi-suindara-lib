"""
SWPERFI Call Drop Log Parser - Data Processor
Author: Pedro Matias
Date: 28/01/2025
License: Apache License 2.0

Description:
------------
This module implements an object-oriented data processor for call drop logs.
It integrates log parsing (via a LogParser attribute) with further cleaning,
transformation, and consolidation of data. The final output is a consolidated
DataFrame along with a history of intermediate partial DataFrames and processing statuses.
"""

import os
import re
import logging
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser as dt_parser

# Import constants from config.py
from ..utils.config import (
    LoggerSetup,
    CALL_STATE_PATTERNS, DISCONNECT_PATTERNS, IMS_CALL_PATTERNS, IMS_TRACKER_PATTERNS,
    VOICE_REG_PATTERNS, SIGNAL_STRENGTH_PATTERNS, PERSIST_ATOMS_PATTERNS, 
    PERSIST_ATOMS_NUMERIC_FIELDS, FILTER_RAT_COLUMNS, FILTER_RAT_PREFIXES,
    VERSION_MAP, RELEVANT_COLUMNS
)

# Import LogParser from your log_parser module
from .log_parser import LogParser

# --------------------------
# DataProcessor: It focuses on cleaning, transforming and engineering features, preparing the dataset for analysis and prediction.  
# --------------------------
class DataProcessor:
    """
    An object-oriented processor for call drop logs.
    
    It uses a LogParser to load raw log data from a ZIP file and then applies cleaning,
    transformation, and consolidation functions. Intermediate DataFrames and statuses
    are stored in a history dictionary.
    """
    def __init__(self, zip_file_path: str):
        """
        Initializes the class with a given ZIP file path. Sets up logging, parsing, and data storage attributes.
        """
        self.logger = LoggerSetup.setup_logger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)        
        
        # Instantiate LogParser
        self.log_parser = LogParser(zip_file_path)

       
        # The LogParser internally loads data upon instantiation
        # (its own history and raw_logs attributes are already set)
        self.history = {}  # Dictionary for processing status reporting
        self.required_dfs = {}  # Dictionary to hold partial DataFrames
        self.consolidated_df = pd.DataFrame()  # Final consolidated DataFrame
        
        # Trigger the complete processing pipeline
        self.run_processing()
    
    def run_processing(self):
        """
        Runs the complete processing pipeline.
        
        It checks if the LogParser has raw logs loaded (via its get_logs() method or history).
        If raw data exists, it then loads the required partial DataFrames and runs consolidation.
        Otherwise, the process is aborted and a status is recorded.
        """
        self.logger.info("[RUN_PROCESSING] Starting DataProcessor pipeline.")
        # Check raw data from LogParser
        raw_logs = self.log_parser.get_logs()
        if raw_logs.empty:
            msg = "LogParser raw logs are empty. Aborting processing."
            self.history['raw_logs'] = {'status': msg, 'rows': 0}
            self.logger.error("[RUN_PROCESSING] %s", msg)
            return
        self.logger.info(f"[RUN_PROCESSING] Raw logs loaded successfully. Number of rows: {raw_logs.shape[0]}")
        
        # Otherwise, proceed with partial DataFrame extraction.
        self.logger.info("[RUN_PROCESSING] Invoking load_partial_dataframes().")
        self.load_partial_dataframes()
        self.logger.info("[RUN_PROCESSING] Invoking consolidate_call_data().")
        self.consolidate_call_data()



    

    # --- Getter methods that store partial DataFrames in the dictionary ---
    def load_partial_dataframes(self):
        """
        Loads all required partial DataFrames from the raw logs provided by LogParser.
        Each partial DataFrame is stored in self.required_dfs and its status (number of rows)
        is recorded in self.history.
        """
        self.required_dfs = {
            'status_df': self.parse_call_id_status_timestamp(self.log_parser.get_logs()),
            'disconnect_df': self.parse_call_disconnect_causes(self.log_parser.get_logs()),
            'persist_atoms_df': self.parse_persist_atoms_storage_calls(self.log_parser.get_logs()),
            'voice_reg_df': self.parse_voice_registration_state(self.log_parser.get_logs()),
            'signal_df': self.parse_signal_strength_df(self.log_parser.get_logs()),
            'ims_tracker_df': self.parse_imsphone_tracker_logs(self.log_parser.get_logs())
        }
        for key, df in self.required_dfs.items():
            self.history[key] = {
                'rows': len(df),
                'status': "Success" if not df.empty else f"Empty after extraction: {key} is empty."
            }
            self.logger.info("[LOAD_PARTIAL] %s: %s - %d rows.", key.upper(), self.history[key]['status'], len(df))
         


    def get_status_df(self) -> pd.DataFrame:
        """Extracts and stores call status information."""
        self.status_df = self.parse_call_id_status_timestamp(self.raw_logs)
        self.history['status_df'] = {
            'rows': len(self.status_df),
            'status': "Success" if not self.status_df.empty else "Empty after processing"
        }
        return self.status_df
    
    def get_disconnect_df(self) -> pd.DataFrame:
        """Extracts and stores disconnect causes."""
        self.disconnect_df = self.parse_call_disconnect_causes(self.raw_logs)
        self.history['disconnect_df'] = {
            'rows': len(self.disconnect_df),
            'status': "Success" if not self.disconnect_df.empty else "Empty after processing"
        }
        return self.disconnect_df

    def get_signal_df(self) -> pd.DataFrame:
        """Extracts and stores signal strength information."""
        self.signal_df = self.parse_signal_strength_df(self.raw_logs)
        self.history['signal_df'] = {
            'rows': len(self.signal_df),
            'status': "Success" if not self.signal_df.empty else "Empty after processing"
        }
        return self.signal_df

    def get_voice_reg_df(self) -> pd.DataFrame:
        """Extracts and stores voice registration information."""
        self.voice_reg_df = self.parse_voice_registration_state(self.raw_logs)
        self.history['voice_reg_df'] = {
            'rows': len(self.voice_reg_df),
            'status': "Success" if not self.voice_reg_df.empty else "Empty after processing"
        }
        return self.voice_reg_df

    def get_persist_atoms_df(self) -> pd.DataFrame:
        """Extracts and stores persist atoms storage information."""
        self.persist_atoms_df = self.parse_persist_atoms_storage_calls(self.raw_logs)
        self.history['persist_atoms_df'] = {
            'rows': len(self.persist_atoms_df),
            'status': "Success" if not self.persist_atoms_df.empty else "Empty after processing"
        }
        return self.persist_atoms_df
    
    def get_ims_tracker_df(self) -> pd.DataFrame:
        """Extracts and stores IMS tracker information."""
        self.ims_tracker_df = self.parse_imsphone_tracker_logs(self.raw_logs)
        self.history['ims_tracker_df'] = {
            'rows': len(self.ims_tracker_df),
            'status': "Success" if not self.ims_tracker_df.empty else "Empty after processing"
        }
        return self.ims_tracker_df

    def parse_call_id_status_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts call IDs, state transitions, and timestamps.
        
        Uses regex patterns from the configuration.
        """
        self.logger.info("[CALL_ID_STATUS] Extracting call state transitions.")
        # Validate required columns
        for col in ['Log_Tag', 'Tag_Values', 'Timestamp']:
            if col not in df.columns:
                raise ValueError(f"Input DataFrame must contain '{col}'.")
        filtered_df = df[df['Log_Tag'] == 'Telephony:'][['Timestamp', 'Tag_Values']].copy()
        filtered_df['call_id'] = filtered_df['Tag_Values'].str.extract(CALL_STATE_PATTERNS['call_id'])
        filtered_df['from_state'] = filtered_df['Tag_Values'].str.extract(CALL_STATE_PATTERNS['from_state'])
        filtered_df['to_state'] = filtered_df['Tag_Values'].str.extract(CALL_STATE_PATTERNS['to_state'])
        result_df = filtered_df.dropna(subset=['call_id', 'from_state', 'to_state'])
        self.logger.info("[CALL_ID_STATUS] Extracted %d records.", len(result_df))
        return result_df

    def parse_call_disconnect_causes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses disconnect causes from call logs.
        """
        self.logger.info("[DISCONNECT_CAUSES] Parsing disconnect causes.")
        disconnect_logs = df[(df['Log_Tag'] == 'Telephony:') &
                             (df['Tag_Values'].str.contains('onDisconnect'))]
        parsed_data = []
        for _, row in disconnect_logs.iterrows():
            log_line = row['Tag_Values']
            timestamp = row['Timestamp']
            call_id_match = re.search(DISCONNECT_PATTERNS['call_id'], log_line)
            cause_match = re.search(DISCONNECT_PATTERNS['cause'], log_line)
            parsed_data.append({
                'Timestamp': timestamp,
                'call_id': call_id_match.group(1) if call_id_match else None,
                'cause': cause_match.group(1) if cause_match else None
            })
        result_df = pd.DataFrame(parsed_data).dropna(subset=['call_id', 'cause'])
        self.logger.info("[DISCONNECT_CAUSES] Parsed %d disconnect records.", len(result_df))
        return result_df

    def parse_ims_call_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses IMS call logs.
        """
        self.logger.info("[IMS_CALL_LOGS] Parsing IMS call logs.")
        ims_logs = df[df['Log_Tag'].str.contains('ImsCall')]
        call_data = []
        for _, row in ims_logs.iterrows():
            log_line = row['Tag_Values']
            timestamp = row['Timestamp']
            log_data = {'Timestamp': timestamp}
            for field, pattern in IMS_CALL_PATTERNS.items():
                match = re.search(pattern, log_line)
                log_data[field] = match.group(1) if match else None
            call_data.append(log_data)
        call_df = pd.DataFrame(call_data)
        for field in ['reason_code', 'callType', 'networkType', 'objId']:
            if field in call_df.columns:
                call_df[field] = pd.to_numeric(call_df[field], errors='coerce')
        self.logger.info("[IMS_CALL_LOGS] Parsed %d IMS call records.", len(call_df))
        return call_df

    def parse_imsphone_tracker_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses IMS Phone Tracker logs.
        """
        self.logger.info("[IMS_PHONE_TRACKER] Parsing IMS Phone Tracker logs.")
        tracker_logs = df[(df['Log_Tag'].str.contains('ImsPhoneCallTracker')) &
                          (df['Tag_Values'].str.contains('telecomCallID'))]
        tracker_data = []
        for _, row in tracker_logs.iterrows():
            log_line = row['Tag_Values']
            timestamp = row['Timestamp']
            log_data = {'Timestamp': timestamp}
            for field, pattern in IMS_TRACKER_PATTERNS.items():
                match = re.search(pattern, log_line)
                log_data[field] = match.group(1) if match else None
            if log_data.get('ImsTracker_cause') is None:
                log_data['ImsTracker_cause'] = -1
            tracker_data.append(log_data)
        tracker_df = pd.DataFrame(tracker_data)
        for field in ['ImsTracker_cause', 'callType', 'audioQuality', 'audioDirection',
                      'videoQuality', 'videoDirection', 'ims_networkType', 'objId']:
            if field in tracker_df.columns:
                tracker_df[field] = pd.to_numeric(tracker_df[field], errors='coerce')
        if 'wifi_st' in tracker_df.columns:
            tracker_df['wifi_st'] = tracker_df['wifi_st'].map({'Y': True, 'N': False})
        self.logger.info("[IMS_PHONE_TRACKER] Parsed %d IMS phone tracker records.", len(tracker_df))
        return tracker_df

    def parse_voice_registration_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses voice registration state logs.
        """
        self.logger.info("[VOICE_REGISTRATION] Parsing voice registration logs.")
        voice_logs = df[df['Tag_Values'].str.contains('VOICE_REGISTRATION_STATE')]
        parsed_data = []
        for _, row in voice_logs.iterrows():
            log_line = row['Tag_Values']
            timestamp = row['Timestamp']
            log_data = {'Timestamp': timestamp}

            # Extract SIM slot
            sim_slot_match = re.search(r'\[PHONE(\d+)\]', log_line)
            log_data['PhoneSlot'] = int(sim_slot_match.group(1)) if sim_slot_match else None

            # Extract values for each parameter using VOICE_REG patterns
            for field, pattern in VOICE_REG_PATTERNS.items():
                match = re.search(pattern, log_line)
                log_data[field] = match.group(1) if match else None
            
            parsed_data.append(log_data)
        parsed_df = pd.DataFrame(parsed_data)
        self.logger.info("[VOICE_REGISTRATION] Parsed %d records.", len(parsed_df))
        return parsed_df

    def parse_signal_strength_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses signal strength logs and identifies the dominant technology.
        """
        self.logger.info("[SIGNAL_STRENGTH] Parsing signal strength logs.")
        signal_logs = df[df['Tag_Values'].str.contains('SIGNAL_STRENGTH SignalStrength:')]
        parsed_data = []
        
        for _, row in signal_logs.iterrows():
            log_line = row['Tag_Values']
            timestamp = row['Timestamp']
            log_data = {'Timestamp': timestamp}

            # Extract SIM slot
            sim_slot_match = re.search(r'\[PHONE(\d+)\]', log_line)
            log_data['PhoneSlot'] = int(sim_slot_match.group(1)) if sim_slot_match else None

            # Extract values for each RAT using imported patterns
            for rat, rat_patterns in SIGNAL_STRENGTH_PATTERNS.items():
                for param, pattern in rat_patterns.items():
                    match = re.search(pattern, log_line)
                    if match:
                        log_data[f"{rat}_{param}"] = int(match.group(1))

            parsed_data.append(log_data)

        # Convert to DataFrame
        parsed_df = pd.DataFrame(parsed_data)
        # Identify dominant technology per record
        def identify_dominant_tech(row):
            INVALID_VALUE = 2147483647
            for col in row.index:
                if "_" in col:
                    tech, _ = col.split("_", 1)
                    value = row[col]
                    if value != INVALID_VALUE and pd.notna(value):
                        return tech
            return None
        parsed_df['Dominant_Technology'] = parsed_df.apply(identify_dominant_tech, axis=1)
        self.logger.info("[SIGNAL_STRENGTH] Parsed %d records.", len(parsed_df))
        return parsed_df

    def parse_persist_atoms_storage_calls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses PersistAtomsStorage logs to extract voice call session information.
        """
        self.logger.info("[PERSIST_ATOMS] Parsing PersistAtomsStorage logs.")
        voice_logs = df[df['Log_Tag'].str.contains('PersistAtomsStorage')]
        start_points = voice_logs[voice_logs['Tag_Values'].str.contains('Add new voice call session', na=False)]
        aggregated_data = []
        
        for _, start_row in start_points.iterrows():
            timestamp = start_row['Timestamp']
            session_logs = voice_logs[voice_logs['Timestamp'] == timestamp]
            session_data = {"Timestamp": timestamp}
            for _, log_row in session_logs.iterrows():
                log_line = log_row['Tag_Values']
                for field, pattern in PERSIST_ATOMS_PATTERNS.items():
                    match = re.search(pattern, log_line)
                    if match:
                        session_data[field] = match.group(1)
            aggregated_data.append(session_data)
        calls_df = pd.DataFrame(aggregated_data)
        
        # Convert numeric fields to appropriate types
        for field in PERSIST_ATOMS_NUMERIC_FIELDS:
            if field in calls_df.columns:
                calls_df[field] = calls_df[field].apply(lambda x: int(x) if pd.notnull(x) else None)
        for field in ["roam"]:
            if field in calls_df.columns:
                calls_df[field] = calls_df[field].map({'true': True, 'false': False})
        # Avoid to keep None the extra_message
        if 'disconnect_extra_message' not in calls_df.columns:
            calls_df['disconnect_extra_message'] = "N/A"
        calls_df['disconnect_extra_message'] = calls_df['disconnect_extra_message'].fillna("N/A")
        self.logger.info("[PERSIST_ATOMS] Parsed %d records.", len(calls_df))
        return calls_df

    def get_most_recent_record(self, sub_df: pd.DataFrame, cut_time: datetime, phone_slot, TAG="DEBUG") -> dict:
        """
        Retrieves the most recent record from a subset DataFrame based on a cut time and phone slot.
        """
        self.logger.debug(f"[{TAG}] Normalizing PhoneSlot in subset DataFrame.")
        try:
            sub_df['PhoneSlot'] = sub_df['PhoneSlot'].apply(lambda x: int(float(x)) if pd.notnull(x) else None)
        except Exception as e:
            raise ValueError(f"[{TAG}] Error normalizing PhoneSlot: {e}")
        
        self.logger.debug(f"[{TAG}] Filtering records with Timestamp <= {cut_time} and PhoneSlot == {phone_slot}.")
        filtered_df = sub_df[(sub_df['Timestamp'] <= pd.to_datetime(cut_time)) & (sub_df['PhoneSlot'] == phone_slot)]
        self.logger.debug(f"[{TAG}] Found {len(filtered_df)} records after filtering.")

        if not filtered_df.empty:
            return filtered_df.sort_values(by='Timestamp', ascending=False).iloc[0].to_dict()
        else:
            self.logger.warning(f"[{TAG}] No records found; returning None for all fields.")
            return {col: None for col in sub_df.columns}

    def get_signal_strength_record(self, signal_df: pd.DataFrame, start_time: datetime, disc_time: datetime, phone_slot) -> dict:
        """
        Processes signal strength records for a given phone slot and call interval.
        Calculates the mean of signal metrics within the interval or retrieves the closest record.
        """
        TAG = "SIGNAL_STRENGTH"
        self.logger.debug(f"[{TAG}] Processing signal strength for PhoneSlot={phone_slot}, Start={start_time}, Disc={disc_time}.")

        # Garantindo que phone_slot seja um valor escalar
        phone_slot = phone_slot if isinstance(phone_slot, (int, str)) else phone_slot.iloc[0]

        # Filtrar registros por PhoneSlot
        filtered_signals = signal_df[signal_df['PhoneSlot'] == phone_slot].dropna(subset=['Dominant_Technology'])
        
        # Verificar se há registros após filtro de PhoneSlot
        if filtered_signals.empty:
            self.logger.warning(f"[{TAG}] No signals found for PhoneSlot={phone_slot}. Returning empty record.")
            return {col: None for col in signal_df.columns}

        # Garantindo que a coluna Timestamp seja datetime
        try:
            filtered_signals['Timestamp'] = pd.to_datetime(filtered_signals['Timestamp'])
        except Exception as e:
            self.logger.error(f"[{TAG}] Error parsing timestamps: {e}")
            return {col: None for col in signal_df.columns}

        self.logger.debug(f"[{TAG}] Found {len(filtered_signals)} records for PhoneSlot {phone_slot}.")

        # Filtrar registros dentro do intervalo da chamada
        valid_signals = filtered_signals[
            (filtered_signals['Timestamp'] >= start_time) & 
            (filtered_signals['Timestamp'] <= disc_time)
        ]

        self.logger.debug(f"[{TAG}] Found {len(valid_signals)} records within the call interval.")

        if not valid_signals.empty:
            numeric_columns = valid_signals.select_dtypes(include=['number']).columns
            averaged_row = valid_signals[numeric_columns].mean(axis=0).apply(int).to_dict()
            averaged_row['Timestamp'] = valid_signals['Timestamp'].max()
            averaged_row['PhoneSlot'] = phone_slot
            averaged_row['Dominant_Technology'] = valid_signals['Dominant_Technology'].mode()[0]
            self.logger.debug(f"[{TAG}] Calculated interval metrics: {averaged_row}")
            return averaged_row

        # Caso não existam registros no intervalo, buscar registros mais próximos
        before_start = filtered_signals[filtered_signals['Timestamp'] < start_time].sort_values(by='Timestamp', ascending=False)
        after_end = filtered_signals[filtered_signals['Timestamp'] > disc_time].sort_values(by='Timestamp', ascending=True)

        self.logger.debug(f"[{TAG}] Records before start: {len(before_start)}, after end: {len(after_end)}.")

        # Escolher o registro mais próximo
        closest_record = None
        if not after_end.empty:
            closest_record = after_end.iloc[0]
            self.logger.debug(f"[{TAG}] Using record after disconnect: {closest_record['Timestamp']}.")
        elif not before_start.empty:
            closest_record = before_start.iloc[0]
            self.logger.debug(f"[{TAG}] Using record before start: {closest_record['Timestamp']}.")

        if closest_record is not None:
            record_dict = closest_record.to_dict()
            self.logger.debug(f"[{TAG}] Returning closest record: {record_dict}")
            return record_dict

        # Se nenhum registro válido for encontrado, retornar um dicionário vazio
        self.logger.warning(f"[{TAG}] No valid signal records found. Returning empty record.")
        return {col: None for col in signal_df.columns}


    def filter_signal_strength_by_dominant_tech(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorders columns to include only signal-related parameters relevant to the dominant technology.
        """
        self.logger.info("[FILTER_REORDER] Reordering columns based on dominant technology.")
        
        signal_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in FILTER_RAT_PREFIXES)]
        non_signal_columns = [col for col in df.columns if col not in signal_columns]
        filtered_records = []
        for _, row in df.iterrows():
            dominant_tech = row.get('Dominant_Technology')
            relevant_columns = FILTER_RAT_COLUMNS.get(dominant_tech, [])
            filtered_record = row[non_signal_columns + relevant_columns].to_dict()
            filtered_records.append(filtered_record)
        self.logger.debug("[FILTER_REORDER] Reordered columns for %d records.", len(filtered_records))
        return pd.DataFrame(filtered_records)

    def filter_reorder_columns_dynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorders DataFrame columns into a predefined order.
        """
        self.logger.info("[FILTER_REORDER_DYNAMIC] Reordering columns dynamically.")
        event = ["IDTAG", "is_drop_triggered"]
        product_info = ['product_code', 'build_code', 'android_version']
        call_info = ['call_id']
        temporal_info = ['start_time', 'act_time', 'disc_time', 'persist_time', 'voice_reg_time', 'signal_strength_time', 'day_of_week', 'hour_of_day']
        network_info = ['ci', 'tac', 'channel', 'band', 'initialRAT', 'activeRAT', 'disconnectRAT', 'ims_networkType', 'rat_handover']
        voice_reg_info = ['regState', 'roam', 'wifi_st', 'mcc', 'mnc', 'rat']
        disconnection_info = ['cause', 'ImsTracker_cause', 'disconnect_reason_code', 'disconnect_extra_message', 'disconnect_extra_code']
        signal_prefixes = ['LTE_', 'NR_', 'GSM_', 'WCDMA_', 'TDSCDMA_', 'CDMA_']
        signal_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in signal_prefixes)]
        final_order = event + product_info + call_info + temporal_info + voice_reg_info + network_info + signal_columns + disconnection_info
        return df[final_order]

    def label_is_drop_from_dict(self, consolidated_df: pd.DataFrame, report_info: dict, time_tolerance: pd.Timedelta = pd.Timedelta(seconds=90)) -> pd.DataFrame:
        """
        Labels the consolidated DataFrame with 'is_drop_triggered' based on report metadata.
        """
        self.logger.info("[LABEL_DROP] Labeling calls based on report info.")
        consolidated_df['disc_time'] = pd.to_datetime(consolidated_df['disc_time'])
        creation_date = pd.to_datetime(report_info.get('creation_date'), errors='coerce')
        consolidated_df['is_drop_triggered'] = False
        if pd.notna(creation_date):
            matches = consolidated_df.apply(
                lambda row: abs(
                    pd.to_datetime(row['disc_time'].strftime('%m-%d %H:%M:%S'), format='%m-%d %H:%M:%S') -
                    pd.to_datetime(creation_date.strftime('%m-%d %H:%M:%S'), format='%m-%d %H:%M:%S')
                ) <= time_tolerance,
                axis=1
            )
            consolidated_df.loc[matches[matches].index, 'is_drop_triggered'] = True
        self.logger.debug("[LABEL_DROP] Labeling complete.")
        return consolidated_df

    def get_android_version_from_build_id(self, build_id: str) -> int:
        """
        Extracts the Android version from the build_id.
        """
        first_char = build_id[0] if build_id else ''
        return VERSION_MAP.get(first_char, 0)

    def generate_unique_code(self, product_name: str) -> int:
        """
        Generates a unique 12-digit numerical code from the product name using SHA256.
        """
        import hashlib
        hash_object = hashlib.sha256(product_name.encode())
        hash_hex = hash_object.hexdigest()
        unique_code = int(hash_hex[:15], 16) % (10**12)
        return unique_code

    def generate_unique_code_4_bldid(self, build: str) -> int:
        """
        Generates a unique 15-digit numerical code for the build_id using SHA256.
        """
        import hashlib
        hash_object = hashlib.sha256(build.encode())
        hash_hex = hash_object.hexdigest()
        unique_code = int(hash_hex[:20], 16) % (10**15)
        return unique_code

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes and transforms features for analysis or prediction.
        """
        self.logger.info("[PROCESS_FEATURES] Transforming features.")
        df = df.rename(columns={
            'rat_at_start': 'initialRAT',
            'rat_at_connected': 'activeRAT',
            'rat_at_end': 'disconnectRAT',
            'rat_switch_count': 'rat_handover'
        })
        df['rat_handover'] = df['rat_handover'].apply(lambda x: 1 if x > 0 else 0)
        return df
    
    @staticmethod
    def classify_call_status(row):
        """
        Classifies the call status based on the presence of timestamps and disconnect reasons.

        Parameters:
        -----------
        row : Series
            A row of the DataFrame containing call information.
        Returns:
        --------
        str
            The classification of the call status: "success", "call drop", or "origination failure".
        """
        # Lista de causas não relacionadas a falhas (adicionada aqui)
        non_dropped_causes = [
            'NORMAL', 'LOCAL', 'INCOMING_MISSED', 'INCOMING_AUTO_REJECTED',
            'INCOMING_REJECTED', 'IMS_MERGED_SUCCESSFULLY', 'POWER_OFF',
            'INVALID_NUMBER', 'UNOBTAINABLE_NUMBER', 'NUMBER_UNREACHABLE',
            'IMS_SIP_ALTERNATE_EMERGENCY_CALL'
        ]

        #display(row)
        
        if row['cause'] in non_dropped_causes:  # If cause is in non-dropped causes
            return "CALL_SUCCESS"
        elif pd.isna(row['act_time']):  # If activation time is missing, it indicates an origination failure
            return "CALL_ORIGINATION_FAILURE"
        else:  # If activation time is present but cause is not in non-dropped causes, it is likely a drop
            return "CALL_DROP"
    
    
    # -- Consolidation Method --
    def consolidate_call_data(self, tolerance: pd.Timedelta = pd.Timedelta(seconds=30)) -> None:
        """
        Consolidates call data by merging the partial DataFrames stored in self.required_dfs.
        Description:
        ------------
        This function combines parsed data from different logs (e.g., call status, signal strength, 
        registration state, and persist storage) to create a comprehensive DataFrame for each call.

        Parameters:
        -----------
        df_raw : DataFrame
            The raw DataFrame containing all log data.
        tolerance : pd.Timedelta, optional
            Time tolerance for associating PersistAtomsStorage records with calls, by default 30 seconds.

        Returns:
        --------
        DataFrame
            A consolidated DataFrame with detailed information about each call, including:
            - Call start, active, and disconnect times.
            - Disconnect causes and persist storage metrics.
            - Signal strength statistics and registration state.
            - Dominant technology and other relevant parameters.

        Workflow:
        ---------
        1. **Extract Call Status**:
            - Parses call start (`IDLE` to `ACTIVE`), activation, and disconnect times using `get_call_id_status_timestamp`.

        2. **Parse Disconnect Causes**:
            - Extracts disconnect reasons using `parse_call_disconnect_causes`.

        3. **Process PersistAtomsStorage Logs**:
            - Extracts metrics such as signal strength and RAT transitions using `parse_persist_atoms_storage_calls`.

        4. **Merge Registration State Information**:
            - Retrieves the most recent registration state (`parse_voice_registration_state`) for each call.

        5. **Integrate Signal Strength Data**:
            - Uses `get_signal_strength_record` to calculate signal metrics for the call interval or fetch nearby records.

        6. **Combine Data into a Unified Format**:
            - Merges all processed data into a single DataFrame, ensuring timestamps and `call_id` align correctly.

        Notes:
        ------
        - This function acts as the final step in the pipeline, bringing together multiple log sources.
        - It ensures no data is duplicated and prioritizes the most relevant records for each call.

        Example:
        --------
        consolidated_data = consolidate_call_data(df_raw, tolerance=pd.Timedelta(seconds=30))

        Output:
        -------
        | call_id     | start_time          | act_time            | disc_time          | cause | signal_strength | ...
        |-------------|---------------------|---------------------|--------------------|-------|-----------------|---
        | TC@1234_1   | 2024-11-29 10:00:00 | 2024-11-29 10:01:00 | 2024-11-29 10:05:00 | DROP  | {'LTE_rsrp': -90, ...}
        """
        self.logger.info("[CONSOLIDATION] Starting consolidation process.")
        
        try:
            # 🚨 Verificação inicial dos DataFrames
            self.logger.info("[CONSOLIDATION] Checking required DataFrames.")
            for key, df in self.required_dfs.items():
                self.logger.info(f"[CONSOLIDATION] [{key.upper()}] {len(df)} rows, columns: {list(df.columns)}")
                if df.empty:
                    msg = f"Empty after processing: {key} is empty - consolidation terminated."
                    self.history['consolidation'] = {'status': msg, 'call_count_before': 0, 'call_count_after': 0}
                    self.logger.warning("[CONSOLIDATION] %s", msg)
                    self.consolidated_df = pd.DataFrame()
                    return


            # 📥 Extração dos DataFrames parciais necessários
            status_df = self.required_dfs['status_df']
            disconnect_df = self.required_dfs['disconnect_df']
            persist_atoms_df = self.required_dfs['persist_atoms_df']
            voice_reg_df = self.required_dfs['voice_reg_df']
            signal_df = self.required_dfs['signal_df']
            ims_tracker_df = self.required_dfs['ims_tracker_df']

            


            # ⏳ Consolidação dos tempos de chamada
            self.logger.info("[CONSOLIDATION] [CALL STATUS] Consolidating call times.")
            call_ends = status_df[status_df['to_state'] == 'DISCONNECTED'].rename(columns={'Timestamp': 'disc_time'})
            call_ends = call_ends.sort_values(by='disc_time').drop_duplicates(subset=['call_id'], keep='first')

            call_active = status_df[status_df['to_state'] == 'ACTIVE'].rename(columns={'Timestamp': 'act_time'})
            call_active = call_active.sort_values(by='act_time').drop_duplicates(subset=['call_id'], keep='first')

            call_starts = status_df[status_df['from_state'] == 'IDLE'].rename(columns={'Timestamp': 'start_time'})
            call_starts = call_starts.sort_values(by='start_time').drop_duplicates(subset=['call_id'], keep='first')

            calls_df = call_ends[['call_id', 'disc_time']].merge(
                call_active[['call_id', 'act_time']], on='call_id', how='left'
            ).merge(
                call_starts[['call_id', 'start_time']], on='call_id', how='left'
            )
            self.logger.info(f"[CONSOLIDATION] [CALL STATUS] Calls DataFrame after consolidation: {calls_df.shape}")

            # Merge de disconnect_df com calls_df
            self.logger.info("[CONSOLIDATION] [DISCONNECT] Merging disconnect causes.")
            calls_df = calls_df.merge(disconnect_df, on='call_id', how='left')
            


            # 📡 Integração de logs do IMSPhone Tracker
            self.logger.info("[CONSOLIDATION] [IMS TRACKER] Merging IMSPhone Tracker logs.")
            ims_tracker_df['Timestamp'] = pd.to_datetime(ims_tracker_df['Timestamp'])
            ims_tracker_selected = ims_tracker_df[['call_id', 'ImsTracker_cause', 'wifi_st', 'ims_networkType']].drop_duplicates()

            calls_df = calls_df.merge(ims_tracker_selected, on='call_id', how='inner', validate="one_to_one")
            self.logger.info(f"[CONSOLIDATION] [IMS TRACKER] Calls DataFrame after merge: {calls_df.shape}")

            # 🗂️ Adição de informações do PersistAtomsStorage
            self.logger.info("[CONSOLIDATION] [PERSIST ATOMS] Merging PersistAtomsStorage data.")
            persist_atoms_df['Timestamp'] = pd.to_datetime(persist_atoms_df['Timestamp'])
            persist_atoms_df = persist_atoms_df.rename(columns={'Timestamp': 'persist_time'})

            persist_atoms_merged = calls_df.apply(
                lambda row: persist_atoms_df[
                    (persist_atoms_df['persist_time'] <= row['disc_time'] + tolerance) &
                    (persist_atoms_df['persist_time'] >= (row['act_time'] if pd.notna(row['act_time']) else row['start_time']))
                ].sort_values(by='persist_time', ascending=False).head(1),
                axis=1
            )
            persist_atoms_expanded = pd.concat(persist_atoms_merged.tolist(), ignore_index=True)
            calls_df = pd.concat([calls_df, persist_atoms_expanded], axis=1)
            self.logger.info(f"[CONSOLIDATION] [PERSIST ATOMS] Calls DataFrame after merge: {calls_df.shape}")
            
            # 🔗 Adição de informações de Registro de Voz
            self.logger.info("[CONSOLIDATION] [VOICE REGISTRATION] Merging Voice Registration data.")
            voice_reg_df['Timestamp'] = pd.to_datetime(voice_reg_df['Timestamp'])
            voice_reg_merged = calls_df.apply(
                lambda row: self.get_most_recent_record(
                    voice_reg_df,
                    cut_time=row['disc_time'] + tolerance,
                    phone_slot=row.get('PhoneSlot'),
                    TAG="VOICE_REGISTRATION"
                ),
                axis=1
            )
            voice_reg_expanded = pd.DataFrame(voice_reg_merged.tolist())
            if 'Timestamp' in voice_reg_expanded.columns:
                voice_reg_expanded = voice_reg_expanded.rename(columns={'Timestamp': 'voice_reg_time'})
            calls_df = pd.concat([calls_df, voice_reg_expanded], axis=1)
            self.logger.info(f"[CONSOLIDATION] [VOICE REGISTRATION] Calls DataFrame after merge: {calls_df.shape}")

            # 📶 Integração de dados de Força de Sinal
            self.logger.info("[CONSOLIDATION] [SIGNAL STRENGTH] Merging Signal Strength data.")
            signal_df['Timestamp'] = pd.to_datetime(signal_df['Timestamp'])
            signal_strength_merged = calls_df.apply(
                lambda row: self.get_signal_strength_record(
                    signal_df,
                    (row['act_time'] if pd.notna(row['act_time']) else row['start_time']) - tolerance, row['disc_time'] + tolerance,
                    row.get('PhoneSlot')
                ),
                axis=1
            )
            signal_strength_expanded = pd.DataFrame(signal_strength_merged.tolist())
            if 'Timestamp' in signal_strength_expanded.columns:
                signal_strength_expanded = signal_strength_expanded.rename(columns={'Timestamp': 'signal_strength_time'})

            calls_df = pd.concat([calls_df, signal_strength_expanded], axis=1)

            # Garantir que PhoneSlot não seja duplicado
            calls_df = calls_df.loc[:, ~calls_df.columns.duplicated()]  # Remover colunas duplicadas

            # Adicionar colunas de força de sinal relevantes ao DataFrame final
            signal_columns = [col for col in signal_strength_expanded.columns if col not in {'Timestamp', 'PhoneSlot'}] # 'Dominant_Technology'
            

             # 🚨 **Seleção das colunas relevantes**
            self.logger.info("[CONSOLIDATION] Selecting relevant columns.")
            self.consolidated_df = calls_df[RELEVANT_COLUMNS + signal_columns]  # Adicionar colunas de força de sinal


            self.consolidated_df = self.filter_signal_strength_by_dominant_tech(self.consolidated_df)

            # Valor padrão para imputação
            default_value = "2147483647"


            # Para cada tecnologia, verificar quais colunas existem no DataFrame e aplicar fillna
            for tech, cols in FILTER_RAT_COLUMNS.items():
                existing_cols = [col for col in cols if col in self.consolidated_df.columns]
                if existing_cols:
                    self.consolidated_df[existing_cols] = self.consolidated_df[existing_cols].fillna(default_value)


            # 🚨 **Verificação Final**
            call_count_before = len(status_df)
            call_count_after = len(self.consolidated_df)

            # Aplicar a classificação do status da chamada
            self.logger.info("[CONSOLIDATION] [IDTAG] Apply the classification call status based on `disconnect cause`.")
            if not self.consolidated_df.empty:
                self.consolidated_df['IDTAG'] = self.consolidated_df.apply(self.classify_call_status, axis=1)
                missing_call_ids = set(disconnect_df['call_id']) - set(self.consolidated_df['call_id'])
                if missing_call_ids:
                    status_message = f"Success with warnings - Missing call events: {', '.join(map(str, missing_call_ids))}."
                else:
                    status_message = "Success - Consolidation complete."
            else:
                status_message = "Consolidation failed - consolidated_df is empty."
                self.logger.warning("[CONSOLIDATION] %s", status_message)
                self.consolidated_df['IDTAG'] = None

            # Incorporar informações do report_info
            if self.log_parser.report_info:
                self.logger.info("[CONSOLIDATION] Incorporating report information.")

                build_id = self.log_parser.report_info.get('build_id', '')
                if build_id:
                    android_version = self.get_android_version_from_build_id(build_id)
                    build_code = self.generate_unique_code_4_bldid(build_id)
                    self.consolidated_df['android_version'] = android_version
                    self.consolidated_df['build_code'] = build_code
                    self.logger.info(f"[CONSOLIDATION] Build ID: {build_id} | Android Version: {android_version} | Build Code: {build_code}")

                product_name = self.log_parser.report_info.get('product', '')
                if product_name:
                    product_code = self.generate_unique_code(product_name)
                    self.consolidated_df['product_code'] = product_code
                    self.logger.info(f"[CONSOLIDATION] Product Name: {product_name} | Product Code: {product_code}")

                # Aplicar rotulagem de chamadas drop com base nas informações do report
                self.logger.info("[CONSOLIDATION] Labeling dropped calls based on report info.")
                self.consolidated_df = self.label_is_drop_from_dict(self.consolidated_df, self.log_parser.report_info)

                # Resumo final da incorporação
                self.logger.info(f"[CONSOLIDATION] Report info successfully incorporated. {len(self.consolidated_df)} rows updated.")
            else:
                self.logger.warning("[CONSOLIDATION] No report information available. Skipping incorporation.")
            
            # 🕒 Criação de colunas derivadas
            self.logger.info("[CONSOLIDATION] Creating additional time-based columns.")
            self.consolidated_df['disc_time'] = pd.to_datetime(self.consolidated_df['disc_time'], errors='coerce')
            self.consolidated_df['day_of_week'] = self.consolidated_df['disc_time'].dt.dayofweek.astype(int) + 1
            self.consolidated_df['hour_of_day'] = self.consolidated_df['disc_time'].dt.hour.astype(int)


            # 📝 **Atualização do Histórico**
            self.history['consolidation'] = {
                'status': status_message,
                'call_count_before': call_count_before,
                'call_count_after': call_count_after
            }
            self.logger.info("[CONSOLIDATION] %s", status_message)

        except Exception as e:
            status_message = f"Error during consolidation: {e}"
            self.logger.error("[CONSOLIDATION] %s", status_message)
            self.consolidated_df = pd.DataFrame()
    

    def save_to_csv(self, save_path: str = None):
        """
        Saves the consolidated DataFrame (self.consolidated_df) to a CSV file inside 
        a 'parsing_results' folder. The filename is automatically generated based on 
        the ZIP file name.

        Parameters:
        -----------
        save_path : str, optional
            The base directory where the 'parsing_results' folder will be created.
            If None, it saves in the same directory as the ZIP file.
        """
        if self.consolidated_df.empty:
            self.logger.warning("[SAVE_CSV] Consolidated DataFrame is empty. Skipping saving.")
            return

        # Se `save_path` não for fornecido, usar o diretório onde o ZIP está armazenado
        if save_path is None:
            save_path = os.path.dirname(self.log_parser.zip_file_path)

        # Criar diretório parsing_results dentro do caminho escolhido
        output_dir = os.path.join(save_path, "parsing_results")
        os.makedirs(output_dir, exist_ok=True)

        # Gerar nome do arquivo baseado no ZIP
        zip_name = os.path.basename(self.log_parser.zip_file_path).replace(".zip", "")
        output_file = os.path.join(output_dir, f"{zip_name}_parsed.csv")

        # Salvar DataFrame
        try:
            self.consolidated_df.to_csv(output_file, index=False)
            self.logger.info(f"[SAVE_CSV] Consolidated DataFrame saved successfully: {output_file}")
        except Exception as e:
            self.logger.error(f"[SAVE_CSV] Error saving Consolidated DataFrame to CSV: {e}")
    


# End of DataProcessor class

