"""
utils.py - Utility functions for the project.
"""

import os
import socket
from datetime import datetime, timezone, timedelta


def get_israel_timestamp():
    """
    Get current timestamp in Israel timezone (UTC+3).

    Returns:
        str: Timestamp string in format 'dd-HH-MM' (day-hour-minute)
    """
    israel_tz = timezone(timedelta(hours=3))
    return datetime.now(israel_tz).strftime("%d-%H-%M")


def get_project_root():
    """
    Returns the absolute path to the project root directory.
    This function is located in utils/utils.py, so we go up two levels.

    Returns:
        str: Absolute path to the project root
    """
    # Get the directory where this file is located (utils/)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the project root
    project_root = os.path.dirname(utils_dir)
    return project_root


def get_timestamped_logdir(subdir_name="runs"):
    """
    Generate a full log_dir path in main script's directory with Israel timezone timestamp.

    Args:
        subdir_name: Name of the subdirectory to create (default: "runs")

    Returns:
        str: Full path to timestamped log directory with format: {subdir_name}/{timestamp}_{hostname}
    """
    hostname = socket.gethostname()
    # Get timestamp in Israel timezone (UTC+3)
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%b%d_%H-%M-%S")
    # Path to the main script (not where it's called from)
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Combine path
    log_dir = os.path.join(base_path, subdir_name, f"{timestamp}_{hostname}")
    return log_dir

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","1","y","on"): return True
    if v in ("no","false","f","0","n","off"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")