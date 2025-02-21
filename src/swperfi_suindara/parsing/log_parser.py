"""
SWPERFI Call Drop Log Parser
Author: Pedro Matias
Date: 28/01/2025
License: Apache License 2.0

Description:
------------
This module provides an object-oriented implementation for parsing call drop logs.
It extracts log events from logcat files and JSON report metadata from ZIP files,
returning structured data (pandas DataFrames and dictionaries). The module is divided
into several classes to encapsulate the functionality.
"""

import re
import json
import zipfile
from io import BytesIO
from datetime import datetime
import pandas as pd
from dateutil import parser as dt_parser
from dataclasses import dataclass

from ..utils.config import LoggerSetup, DESIRED_TAGS, DESIRED_STRINGS, PERSIST_VALUES, WHOIS_TIMEZONE_INFO




# --------------------------
# LogEvent: Represents an extracted log entry
# --------------------------
@dataclass
class LogEvent:
    Timestamp: datetime
    PID: int
    TID: int
    Log_Level: str
    Log_Tag: str
    Tag_Values: str

# --------------------------
# LogFile: Represents a single log file and its parsing logic.
# --------------------------
class LogFile:
    """
    Represents a single log file and provides methods to parse its content.
    """
    def __init__(self, file_content: BytesIO):
        """
        Initializes the LogFile with the given file-like object.
        
        Parameters:
            file_content (BytesIO): The content of the log file.
        """
        self.file_content = file_content
        self.logger = LoggerSetup.setup_logger(self.__class__.__name__)
    
    def _detect_year_presence(self) -> bool:
        """
        Determines if the log lines include a four-digit year.
        
        Returns:
            bool: True if a year is detected, False otherwise.
        """
        self.file_content.seek(0)
        for line in self.file_content:
            line = line.decode('iso-8859-1')
            if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}', line):
                return True
        return False

    def parse(self) -> pd.DataFrame:
        """
        Parses the log file content and returns a DataFrame of log events.
        
        Returns:
            pd.DataFrame: DataFrame containing columns:
              - Timestamp
              - PID
              - TID
              - Log Level
              - Log_Tag
              - Tag_Values
        """
        self.logger.info("[PARSING] Starting to parse log file content.")
        year_present = self._detect_year_presence()
        
        # Define regex pattern based on year presence
        if year_present:
            pattern = re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+'
                r'(?P<pid>\d+)\s+'
                r'(?P<tid>\d+)\s+'
                r'(?P<log_level>[VDIWE])\s+'
                r'(?P<log_tag>\S+)\s+'
                r'(?P<tag_values>.*)'
            )
        else:
            pattern = re.compile(
                r'(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+'
                r'(?P<pid>\d+)\s+'
                r'(?P<tid>\d+)\s+'
                r'(?P<log_level>[VDIWE])\s+'
                r'(?P<log_tag>\S+)\s+'
                r'(?P<tag_values>.*)'
            )
        
        
        # Lists to store extracted values
        timestamps, pids, tids, log_levels, log_tags, tag_values = [], [], [], [], [], []
        
        self.file_content.seek(0)
        for line in self.file_content:
            line = line.decode('iso-8859-1')
            match = re.match(pattern, line)
            if match:
                data = match.groupdict()
                if data['log_tag'] in DESIRED_TAGS and any(s in data['tag_values'] for s in DESIRED_STRINGS + PERSIST_VALUES):
                    timestamp = data['timestamp']
                    if not year_present:
                        timestamp = f"{datetime.now().year}-{timestamp}"
                    timestamps.append(timestamp)
                    pids.append(int(data['pid']))
                    tids.append(int(data['tid']))
                    log_levels.append(data['log_level'])
                    log_tags.append(data['log_tag'])
                    tag_values.append(data['tag_values'])
        
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'PID': pids,
            'TID': tids,
            'Log Level': log_levels,
            'Log_Tag': log_tags,
            'Tag_Values': tag_values
        })
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        self.logger.info("[PARSING] Finished parsing log file. Extracted %d rows.", len(df))
        return df

# --------------------------
# ReportInformation: Handles JSON metadata extraction.
# --------------------------
class ReportInformation:
    """
    Handles extraction of metadata from JSON report files.
    """
    def __init__(self, file_contents: list):
        """
        Initializes the ReportInformation with a list of file-like objects (BytesIO)
        containing JSON report data.
        
        Parameters:
            file_contents (list): List of BytesIO objects.
        """
        self.file_contents = file_contents
        self.logger = LoggerSetup.setup_logger(self.__class__.__name__)
    
    def extract(self) -> dict:
        """
        Extracts relevant information from the JSON report files.
        
        Returns:
            dict: A dictionary with keys:
                - creation_date: Report creation date ('YYYY-MM-DD HH:MM:SS').
                - summary: Report summary.
                - category: Report category.
                - event_details: Dictionary of details extracted from the description.
                - build_id: Build id from device information.
                - product: Product name from device information.
        """
        self.logger.info("[REPORT EXTRACTION] Starting extraction from JSON report files.")
        result = {}
        
        for file_content in self.file_contents:
            try:
                data = json.load(file_content)
            except Exception as e:
                self.logger.error("[REPORT EXTRACTION] Error loading JSON: %s", e)
                continue

            # Extract report information
            if "Report Creation Date" in data:
                creation_date_str = data.get("Report Creation Date", "")
                creation_date = None
                if creation_date_str:
                    creation_date = dt_parser.parse(creation_date_str, tzinfos=WHOIS_TIMEZONE_INFO)
                    creation_date = creation_date.strftime('%Y-%m-%d %H:%M:%S')
                summary = data.get("Summary", "")
                category = data.get("Category", "")
                description = data.get("Description", "")
                event_details = {}
                if description:
                    details = description.split(',')
                    for detail in details:
                        key_value = detail.split(':')
                        if len(key_value) == 2:
                            key = key_value[0].strip()
                            value = key_value[1].strip()
                            event_details[key] = value
                result.update({
                    "creation_date": creation_date,
                    "summary": summary,
                    "category": category,
                    "event_details": event_details
                })
            if "Device Information" in data:
                build_id = data["Device Information"]["Build Information"].get("Build id", "")
                product = data["Device Information"]["Build Information"].get("Product", "")
                product = product.split('_')[0]
                result.update({
                    "build_id": build_id,
                    "product": product
                })
        self.logger.info("[REPORT EXTRACTION] Finished extracting report information.")
        return result

# --------------------------
# LogParser: Main orchestrator for processing ZIP files and parsing logs.
# --------------------------
class LogParser:
    """
    Main orchestrator for processing ZIP files and parsing logs.
    """
    def __init__(self, zip_file_path: str):
        """
        Initializes the LogParser with the path to the ZIP file.
        
        Parameters:
            zip_file_path (str): Full path to the ZIP file containing logs and reports.
        """
        self.zip_file_path = zip_file_path
        self.logger = LoggerSetup.setup_logger(self.__class__.__name__)
        self.logs_df = pd.DataFrame()   # To store parsed log events.
        self.report_info = None         # To store extracted report metadata.
        self.history = {}               # To store processing details (e.g., file name, row counts, status).

        # Trigger full processing upon initialization.
        self.run_processing()

    def run_processing(self):
        """
        Runs the full processing pipeline:
          1. Processes the ZIP file.
          2. Loads the parsed logs and report information.
          3. Updates the history with processing details.
        """
        try:
            self.logger.info("[RUN_PROCESSING_PARSER] Starting processing of ZIP file: %s", self.zip_file_path)
            self.process_zip()
            # Retrieve the results using getters:
            self.logs_df = self.get_logs()
            self.report_info = self.get_report_info()
            self.history['raw_logs'] = {
                'file': self.zip_file_path,
                'rows': len(self.logs_df),
                'status': "Loaded successfully" if not self.logs_df.empty else "Empty after processing"
            }
            self.logger.info("[RUN_PROCESSING_PARSER] Loaded %d raw log rows.", len(self.logs_df))
            return
        except Exception as e:
            self.history['raw_logs'] = {'status': f"Error: {e}"}
            self.logger.error("[RUN_PROCESSING_PARSER] Error processing ZIP file: %s", e)

    def process_zip(self):
        """
        Processes the ZIP file:
          - Opens the ZIP.
          - Selects log files (e.g. files ending in '.txt' containing 'radio' or 'main').
          - Selects JSON report files (e.g. files ending with '_information.json').
          - Structure each file and updates self.logs_df and self.report_info.
        """
        self.logger.info("[ZIP PROCESS] Starting processing of ZIP file: %s", self.zip_file_path)
        try:
            with zipfile.ZipFile(self.zip_file_path, 'r') as z:
                file_names = z.namelist()
                # Select log files: .txt files containing 'radio' or 'main'
                log_files = [name for name in file_names if name.endswith(".txt") and ("radio" in name or "main" in name)]
                # Select JSON report files ending with '_information.json'
                report_files = [name for name in file_names if name.endswith("_information.json")]
                
                self.logger.info("[ZIP PROCESS] Found %d log files and %d report files.", len(log_files), len(report_files))
                
                # Process each log file
                for lf in log_files:
                    self.logger.info("[ZIP PROCESS] Processing log file: %s", lf)
                    with z.open(lf) as file_content:
                        # Wrap the content in BytesIO and parse using LogFile
                        log_file = LogFile(BytesIO(file_content.read()))
                        df = log_file.parse()
                        self.logs_df = pd.concat([self.logs_df, df], ignore_index=True)
                
                if not self.logs_df.empty:
                    self.logs_df = self.logs_df.sort_values(by="Timestamp").drop_duplicates()
                    self.logger.info("[ZIP PROCESS] Consolidated %d log records.", len(self.logs_df))
                else:
                    self.logger.warning("[ZIP PROCESS] No log records extracted.")
                
                # Process report information if available
                if report_files:
                    info_contents = [BytesIO(z.read(rf)) for rf in report_files]
                    report_info_instance = ReportInformation(info_contents)
                    self.report_info = report_info_instance.extract()
                else:
                    self.logger.warning("[ZIP PROCESS] No report information files found.")
        except zipfile.BadZipFile:
            self.logger.error("[ZIP PROCESS] The file '%s' is not a valid ZIP file.", self.zip_file_path)
            raise

    def get_logs(self) -> pd.DataFrame:
        """
        Returns the parsed log DataFrame.
        
        Returns:
            pd.DataFrame: The consolidated DataFrame with log events.
        """
        return self.logs_df.reset_index(drop=True)
    

    def get_report_info(self) -> dict:
        """
        Returns the extracted report metadata.
        
        Returns:
            dict: The dictionary with report information, or None if not available.
        """
        return self.report_info