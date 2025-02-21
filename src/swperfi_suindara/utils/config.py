# config.py

"""
SWPERFI Call Drop Log Parser - Configuration File
Author: Pedro Matias
Date: 28/01/2025
License: Apache License 2.0

Description:
------------
This configuration file defines constants and settings used by the SWPERFI Call Drop Log Parser.
It includes lists for desired log tags, strings, persistent values, and a mapping of timezones used
to properly parse dates in JSON report files.
"""

import logging
import os


# --------------------------
# LoggerSetup: Centralized logging configuration
# --------------------------
class LoggerSetup:
    @staticmethod
    def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
        """
        Sets up and returns a logger with a specific format.
        
        Parameters:
            name (str): The name of the logger.
            log_dir (str): Directory where log files will be saved. Default is "logs".
        
        Returns:
            logging.Logger: Configured logger instance.
        """
        # Create the log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create the log file name based on the class/module name
        log_file = os.path.join(log_dir, f"{name}.log")

        # Set up the logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Check if the logger already has handlers, to avoid adding multiple handlers
        if not logger.handlers:
            # Create a console handler with the desired format
            ch = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
            # Create a file handler for saving logs to a file
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

# --------------------------
# --------------------------
# --------------------------
# --------------------------


# List of desired log tags
DESIRED_TAGS = [
    'dumpstate:', 'LocationManagerService:', 'Telephony:',
    'ImsCall', 'RILJ', 'ImsPhoneCallTracker:', 'PersistAtomsStorage:'
]

# List of desired strings to filter within log entries
DESIRED_STRINGS = [
    'Bugreport dir', ', Lat: ', 'GsmConnection: Update state from',
    ': onDisconnect: callId=', 'TelephonyConnectionService: Adding IMS connection to conference controller: ',
    ': processCallTerminated :: reason=ImsReasonInfo ::', '< VOICE_REGISTRATION_STATE',
    'SignalStrength:', '[ImsPhoneConnection', ' cause = ', 'maybeRemapReasonCode : fromCode ', 'isVolteEnabled'
]

# List of persistent values to check in the log entry
PERSIST_VALUES = [
    'Add new voice call session', 'bearer_at_end', 'bearer_at_start', 'carrier_id',
    'codec_bitmask', 'concurrent_call_count_at_end', 'concurrent_call_count_at_start',
    'direction', 'disconnect_extra_code', 'disconnect_extra_message', 'disconnect_reason_code',
    'fold_state', 'is_emergency', 'is_esim', 'is_multi_sim', 'is_multiparty', 'is_roaming',
    'last_known_rat', 'main_codec_quality', 'rat_at_connected', 'rat_at_end', 'rat_at_start',
    'rat_switch_count', 'rtt_enabled', 'setup_begin_millis', 'setup_duration_millis', 'setup_failed',
    'signal_strength_at_end', 'sim_slot_index', 'srvcc_cancellation_count', 'srvcc_completed',
    'srvcc_failure_count', 'video_enabled'
]

# Timezone mapping for parsing dates in JSON report files
WHOIS_TIMEZONE_INFO = {
    "A": 3600, "ACDT": 37800, "ACST": 34200, "ACT": -18000, "ACWST": 31500,
    "ADT": 14400, "AEDT": 39600, "AEST": 36000, "AET": 36000, "AFT": 16200,
    "AKDT": -28800, "AKST": -32400, "ALMT": 21600, "AMST": -10800, "AMT": -14400,
    "ANAST": 43200, "ANAT": 43200, "AQTT": 18000, "ART": -10800, "AST": 10800,
    "AT": -14400, "AWDT": 32400, "AWST": 28800, "AZOST": 0, "AZOT": -3600,
    "AZST": 18000, "AZT": 14400, "AoE": -43200, "B": 7200, "BNT": 28800,
    "BOT": -14400, "BRST": -7200, "BRT": -10800, "BST": 21600, "BTT": 21600,
    "C": 10800, "CAST": 28800, "CAT": 7200, "CCT": 23400, "CDT": -18000,
    "CEST": 7200, "CET": 3600, "CHADT": 49500, "CHAST": 45900, "CHOST": 32400,
    "CHOT": 28800, "CHUT": 36000, "CIDST": -14400, "CIST": -18000, "CKT": -36000,
    "CLST": -10800, "CLT": -14400, "COT": -18000, "CST": -21600, "CT": -21600,
    "CVT": -3600, "CXT": 25200, "ChST": 36000, "D": 14400, "DAVT": 25200,
    "DDUT": 36000, "E": 18000, "EASST": -18000, "EAST": -21600, "EAT": 10800,
    "ECT": -18000, "EDT": -14400, "EEST": 10800, "EET": 7200, "EGST": 0,
    "EGT": -3600, "EST": -18000, "ET": -18000, "F": 21600, "FET": 10800,
    "FJST": 46800, "FJT": 43200, "FKST": -10800, "FKT": -14400, "FNT": -7200,
    "G": 25200, "GALT": -21600, "GAMT": -32400, "GET": 14400, "GFT": -10800,
    "GILT": 43200, "GMT": 0, "GST": 14400, "GYT": -14400, "H": 28800,
    "HDT": -32400, "HKT": 28800, "HOVST": 28800, "HOVT": 25200, "HST": -36000,
    "I": 32400, "ICT": 25200, "IDT": 10800, "IOT": 21600, "IRDT": 16200,
    "IRKST": 32400, "IRKT": 28800, "IRST": 12600, "IST": 19800, "JST": 32400,
    "K": 36000, "KGT": 21600, "KOST": 39600, "KRAST": 28800, "KRAT": 25200,
    "KST": 32400, "KUYT": 14400, "L": 39600, "LHDT": 39600, "LHST": 37800,
    "LINT": 50400, "M": 43200, "MAGST": 43200, "MAGT": 39600, "MART": 34200,
    "MAWT": 18000, "MDT": -21600, "MHT": 43200, "MMT": 23400, "MSD": 14400,
    "MSK": 10800, "MST": -25200, "MT": -25200, "MUT": 14400, "MVT": 18000,
    "MYT": 28800, "N": -3600, "NCT": 39600, "NDT": 9000, "NFT": 39600,
    "NOVST": 25200, "NOVT": 25200, "NPT": 19800, "NRT": 43200, "NST": 12600,
    "NUT": -39600, "NZDT": 46800, "NZST": 43200, "O": -7200, "OMSST": 25200,
    "OMST": 21600, "ORAT": 18000, "P": -10800, "PDT": -25200, "PET": -18000,
    "PETST": 43200, "PETT": 43200, "PGT": 36000, "PHOT": 46800, "PHT": 28800,
    "PKT": 18000, "PMDT": -7200, "PMST": -10800, "PONT": 39600, "PST": -28800,
    "PT": -28800, "PWT": 32400, "PYST": -10800, "PYT": -14400, "Q": -14400,
    "QYZT": 21600, "R": -18000, "RET": 14400, "ROTT": -10800, "S": -21600,
    "SAKT": 39600, "SAMT": 14400, "SAST": 7200, "SBT": 39600, "SCT": 14400,
    "SGT": 28800, "SRET": 39600, "SRT": -10800, "SST": -39600, "SYOT": 10800,
    "T": -25200, "TAHT": -36000, "TFT": 18000, "TJT": 18000, "TKT": 46800,
    "TLT": 32400, "TMT": 18000, "TOST": 50400, "TOT": 46800, "TRT": 10800,
    "TVT": 43200, "U": -28800, "ULAST": 32400, "ULAT": 28800, "UTC": 0,
    "UYST": -7200, "UYT": -10800, "UZT": 18000, "V": -32400, "VET": -14400,
    "VLAST": 39600, "VLAT": 36000, "VOST": 21600, "VUT": 39600, "W": -36000,
    "WAKT": 43200, "WARST": -10800, "WAST": 7200, "WAT": 3600, "WEST": 3600,
    "WET": 0, "WFT": 43200, "WGST": -7200, "WGT": -10800, "WIB": 25200,
    "WIT": 32400, "WITA": 28800, "WST": 50400, "WT": 0, "X": -39600,
    "Y": -43200, "YAKST": 36000, "YAKT": 32400, "YAPT": 36000, "YEKST": 21600,
    "YEKT": 18000, "Z": 0
}


# Regex patterns for call state extraction (for get_call_id_status_timestamp)
CALL_STATE_PATTERNS = {
    'call_id': r'for\s+(TC@\d+_?\d*)',
    'from_state': r'Update state from (\w+) to \w+',
    'to_state': r'Update state from \w+ to (\w+)'
}

# Regex patterns for disconnect causes
DISCONNECT_PATTERNS = {
    'call_id': r'callId=(TC@\d+_\d+)',
    'cause': r'cause=(\w+)'
}

# Regex patterns for IMS call logs
IMS_CALL_PATTERNS = {
    'reason_code': r'reason=ImsReasonInfo\s*::\s*\{(\d+)',
    'reason_string': r'ImsReasonInfo\s*::.*?CODE_\w+,\s\d+,\s(.*?);',
    'userInitiated': r'userInitiated\s*=\s*(\w+)',
    'networkType': r'networkType:(\d+)',
    'wifi_st': r'isWifi:\s*(\w+)',
    'callType': r'callType=(\d+)',
    'objId': r'ImsCall\sobjId:(\d+)',
}

# Regex patterns for IMS Phone Tracker logs
IMS_TRACKER_PATTERNS = {
    'ImsTracker_cause': r'cause\s*=\s*(\d+)',
    'call_id': r'telecomCallID:\s*(TC@\d+_\d+)',
    'ims_networkType': r'networkType:(\d+)',
    'wifi_st': r'isWifi:\s*(\w+)',
    'callType': r'callType=(\d+)',
    'audioQuality': r'audioQuality=(\d+)',
    'audioDirection': r'audioDirection=(\d+)',
    'videoQuality': r'videoQuality=(\d+)',
    'videoDirection': r'videoDirection=(-?\d+)',
    'objId': r'ImsCall\sobjId:(\d+)',
}

# Regex patterns for voice registration state logs
VOICE_REG_PATTERNS = {
    'regState': r'regState\s*[:=]\s*(\w+)',
    'rat': r'rat\s*[:=]\s*(\w+)',
    'mcc': r'mcc\s*[:=]\s*(\d+)',
    'mnc': r'mnc\s*[:=]\s*(\d+)',
    'ci': r'(?:ci|cid)\s*[:=]\s*(\d+)',
    'tac': r'(?:tac|lac)\s*[:=]\s*(\d+)',
    'channel': r'(?:earfcn|nrarfcn|arfcn)\s*[:=]\s*(\d+)',
    'band': r'bands\s*[:=]\s*\[(?:BAND_)?(\d+)'
}

# Regex patterns for signal strength logs 


SIGNAL_STRENGTH_PATTERNS = {
    'CDMA': {
        'cdmaDbm': r'cdmaDbm\s*[:=]\s*(-?\d+)',
        'cdmaEcio': r'cdmaEcio\s*[:=]\s*(-?\d+)',
        'evdoDbm': r'evdoDbm\s*[:=]\s*(-?\d+)',
        'evdoEcio': r'evdoEcio\s*[:=]\s*(-?\d+)',
        'evdoSnr': r'evdoSnr\s*[:=]\s*(-?\d+)',
    },
    'GSM': {
        'rssi': r'rssi\s*[:=]\s*(-?\d+)',
        'ber': r'ber\s*[:=]\s*(-?\d+)',
        'mTa': r'mTa\s*[:=]\s*(-?\d+)',
    },
    'WCDMA': {
        'ss': r'ss\s*[:=]\s*(-?\d+)',
        'ber': r'ber\s*[:=]\s*(-?\d+)',
        'rscp': r'rscp\s*[:=]\s*(-?\d+)',
        'ecno': r'ecno\s*[:=]\s*(-?\d+)',
    },
    'TDSCDMA': {
        'rssi': r'rssi\s*[:=]\s*(-?\d+)',
        'ber': r'ber\s*[:=]\s*(-?\d+)',
        'rscp': r'rscp\s*[:=]\s*(-?\d+)',
    },
    'LTE': {
        'rsrp': r'rsrp\s*[:=]\s*(-?\d+)',
        'rsrq': r'rsrq\s*[:=]\s*(-?\d+)',
        'rssnr': r'rssnr\s*[:=]\s*(-?\d+)',
    },
    'NR': {
        'ssRsrp': r'ssRsrp\s*=\s*(-?\d+)',
        'ssRsrq': r'ssRsrq\s*=\s*(-?\d+)',
        'ssSinr': r'ssSinr\s*=\s*(-?\d+)',
    },
}

# Regex for PERSIST 

PERSIST_ATOMS_PATTERNS = {
            "band_at_end": r'band_at_end:\s*(\d+)',
            "bearer_at_end": r'bearer_at_end:\s*(\d+)',
            "bearer_at_start": r'bearer_at_start:\s*(\d+)',
            "carrier_id": r'carrier_id:\s*(\d+)',
            "direction": r'direction:\s*(\d+)',
            "disconnect_extra_code": r'disconnect_extra_code:\s*(\d+)',
            "disconnect_extra_message": r'disconnect_extra_message:\s*\"([^\"]+)\"',
            "disconnect_reason_code": r'disconnect_reason_code:\s*(\d+)',
            "roam": r'is_roaming:\s*(\w+)',
            "activeRAT": r'rat_at_connected:\s*(\d+)',
            "disconnectRAT": r'rat_at_end:\s*(\d+)',
            "initialRAT": r'rat_at_start:\s*(\d+)',
            "rat_handover": r'rat_switch_count:\s*(\d+)',
            "signal_strength_at_end": r'signal_strength_at_end:\s*(\d+)',
            "PhoneSlot": r'sim_slot_index:\s*(\d+)',
        }

PERSIST_ATOMS_NUMERIC_FIELDS = [
            "band_at_end", "bearer_at_end", "bearer_at_start",
            "carrier_id", "direction", "disconnect_extra_code",
            "disconnect_reason_code", "activeRAT", "disconnectRAT",
            "initialRAT", "rat_handover", "signal_strength_at_end", "PhoneSlot"
        ]


FILTER_RAT_COLUMNS = {
            'CDMA': ['CDMA_cdmaDbm', 'CDMA_cdmaEcio', 'CDMA_evdoDbm', 'CDMA_evdoEcio', 'CDMA_evdoSnr'],
            'GSM': ['GSM_rssi', 'GSM_ber', 'GSM_mTa'],
            'WCDMA': ['WCDMA_ss', 'WCDMA_ber', 'WCDMA_rscp', 'WCDMA_ecno'],
            'TDSCDMA': ['TDSCDMA_rssi', 'TDSCDMA_ber', 'TDSCDMA_rscp'],
            'LTE': ['LTE_rsrp', 'LTE_rsrq', 'LTE_rssnr'],
            'NR': ['NR_ssRsrp', 'NR_ssRsrq', 'NR_ssSinr'],
        }
FILTER_RAT_PREFIXES = ['LTE_', 'NR_', 'GSM_', 'WCDMA_', 'TDSCDMA_', 'CDMA_']

# Default time tolerance values for merging records (can be overridden)
DEFAULT_TOLERANCE = 30  # seconds

# Version mapping for build IDs
VERSION_MAP = {'S': 12, 'T': 13, 'U': 14, 'V':15}

RELEVANT_COLUMNS = [  # Certifique-se de que apenas uma coluna 'PhoneSlot' será incluída
                'call_id', 'start_time', 'act_time', 'disc_time', 'persist_time', 'voice_reg_time', 'signal_strength_time', 'cause',
                'ImsTracker_cause', 'wifi_st', 'ims_networkType',
                'band_at_end', 'bearer_at_end', 'bearer_at_start', 'signal_strength_at_end',
                'roam', 'disconnect_reason_code', 'disconnect_extra_message', 'disconnect_extra_code',
                'direction', 'carrier_id',  'initialRAT', 'activeRAT', 'disconnectRAT',  'rat_handover', # 'call_duration',
                'regState', 'rat', 'mcc', 'mnc', 'ci', 'tac', 'channel', 'band', 'PhoneSlot'
            ]