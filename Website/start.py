#!/usr/bin/env python3
"""
Main entry point script for the Smart Home Security System.

This script checks for essential dependencies, changes the working directory
to the script's location, and then imports and runs the main application
logic from the CameraAndWebsite module. It handles basic error reporting
and ensures a clean exit.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import traceback # Keep for error reporting in main

def check_dependencies():
    """
    Checks if essential Python libraries are installed.

    Attempts to import required modules (cv2, dlib, numpy, flask, lgpio).
    Prints status messages indicating success or failure.

    Returns:
        bool: True if all dependencies appear to be installed, False otherwise.
    """
    print("Checking dependencies...")
    missing = []
    try:
        import cv2
        print("- OpenCV (cv2): Found")
    except ImportError:
        missing.append("cv2 (OpenCV)")
    try:
        import dlib
        print("- dlib: Found")
    except ImportError:
        missing.append("dlib")
    try:
        import numpy
        print("- NumPy: Found")
    except ImportError:
        missing.append("numpy")
    try:
        import flask
        print("- Flask: Found")
    except ImportError:
        missing.append("Flask")
    try:
        import lgpio
        print("- lgpio: Found (Required for LCD)")
    except ImportError:
        print("- lgpio: Not Found (LCD functionality will be disabled)")
        # Decide if lgpio is critical or optional
        # missing.append("lgpio") # Uncomment if lgpio is strictly required

    if not missing:
        print("All essential dependencies seem to be installed.")
        return True
    else:
        print("\nERROR: Missing dependencies:")
        for lib in missing:
            print(f"  - {lib}")
        print("\nPlease install the missing libraries.")
        print("Example: pip install opencv-python dlib numpy Flask lgpio")
        return False

def main():
    """
    Main execution function.

    Checks dependencies, changes the current directory to the script's location,
    imports the main application function from CameraAndWebsite, and runs it.
    Handles KeyboardInterrupt for graceful shutdown and prints other exceptions.

    Returns:
        int: 0 on successful execution or graceful shutdown, 1 on error.
    """
    if not check_dependencies():
        return 1

    print("\nStarting Smart Home Security System...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Changing working directory to: {script_dir}")
    os.chdir(script_dir)

    try:
        # Dynamically import after changing directory and checking deps
        from CameraAndWebsite import main as camera_main
        print("Launching main application...")
        camera_main()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down gracefully...")
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        return 1

    print("Application finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())