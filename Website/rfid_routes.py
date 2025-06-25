"""
Manages RFID reader operations and provides Flask routes for interaction.

Defines the RFIDController class to handle reading and writing to an MFRC522
RFID reader using the mfrc522 library. Implements continuous background
scanning, interaction with the DoorController, and thread synchronization.
Also provides Flask routes for manual read/write operations and retrieving
the currently detected card name.
"""

import threading
import time
from flask import jsonify
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import traceback


class RFIDController:
    """
    Handles interactions with the MFRC522 RFID reader.

    Manages read/write operations, continuous scanning in a background thread,
    synchronization via a lock, and communication with the DoorController.

    Attributes:
        reader (SimpleMFRC522): Instance of the RFID reader library.
        reader_lock (threading.Lock): Lock to synchronize access to the reader.
        current_operation (None): Placeholder for potential operation tracking.
        last_result (dict, optional): Stores the result of the last operation.
        is_reading (bool): Flag indicating if an operation is actively using the reader.
        continuous_scan (bool): Flag to control the background scanning thread.
        scan_thread (threading.Thread, optional): Handle for the background thread.
        current_rfid_name (str, optional): Name/data from the currently detected card.
        door_controller (DoorController, optional): Reference to the main door controller.
    """
    def __init__(self):
        """Initializes the RFID reader, lock, and state variables."""
        self.reader = SimpleMFRC522()
        self.reader_lock = threading.Lock()
        self.current_operation = None
        self.last_result = None
        self.is_reading = False
        self.continuous_scan = False
        self.scan_thread = None
        self.current_rfid_name = None
        self.door_controller = None

    def set_door_controller(self, controller):
        """
        Sets the reference to the DoorController instance.

        Args:
            controller (DoorController): The main door controller instance.
        """
        self.door_controller = controller

    def read_card(self):
        """
        Performs a blocking read operation to get data from an RFID card.

        Acquires the reader lock, attempts to read, updates the internal state
        and the DoorController, and stores the result. Handles exceptions.

        Returns:
            dict: A dictionary containing 'success' (bool), and either 'id' (str)
                  and 'data' (str) on success, or 'error' (str) on failure.
        """
        with self.reader_lock:
            try:
                self.is_reading = True
                print("Attempting RFID read...")
                id_val, text = self.reader.read()
                print(f"Read result: ID={id_val}, Text='{text}'")

                current_name = text.strip() if text else None
                self.current_rfid_name = current_name

                if self.door_controller:
                    self.door_controller.set_rfid_card(current_name)

                self.last_result = {
                    'success': True,
                    'id': str(id_val),
                    'data': text.strip() if text else 'No data'
                }
                return self.last_result
            except Exception as e:
                print(f"Error during RFID read: {str(e)}")
                traceback.print_exc()
                self.last_result = {
                    'success': False,
                    'error': f"Read failed: {str(e)}"
                }
                return self.last_result
            finally:
                self.is_reading = False

    def write_card(self, data):
        """
        Performs a blocking write operation to store data on an RFID card.

        Acquires the reader lock, attempts to write the provided data, and
        stores the result. Handles exceptions.

        Args:
            data (str): The string data to write to the card.

        Returns:
            dict: A dictionary containing 'success' (bool), and either
                  'message' (str) on success, or 'error' (str) on failure.
        """
        with self.reader_lock:
            try:
                self.is_reading = True
                print(f"Attempting RFID write with data: '{data}'")

                self.reader.write(data)

                print("RFID write successful.")
                self.last_result = {
                    'success': True,
                    'message': 'Data written successfully'
                }
                return self.last_result
            except IndexError as ie:
                print(f"IndexError during RFID write: {str(ie)}")
                traceback.print_exc()
                self.last_result = {
                    'success': False,
                    'error': f"Write failed: Index Error. Check card or connection. ({str(ie)})"
                }
                return self.last_result
            except Exception as e:
                print(f"Generic error during RFID write: {str(e)}")
                traceback.print_exc()
                self.last_result = {
                    'success': False,
                    'error': f"Write failed: {str(e)}"
                }
                return self.last_result
            finally:
                self.is_reading = False

    def continuous_read(self):
        """
        Target function for the background RFID scanning thread.

        Continuously performs non-blocking reads using the reader lock. Detects
        when a card is presented or removed, updates the internal state
        (`current_rfid_name`), and notifies the DoorController accordingly.
        Runs while `self.continuous_scan` is True.
        """
        print("Starting continuous RFID scanning...")
        last_card_id = None
        card_absent_count = 0
        card_present_id = None
        card_present_text = None
        card_present_time = 0

        while self.continuous_scan:
            try:
                id_val, text = None, None
                with self.reader_lock:
                    id_val, text = self.reader.read_no_block()

                current_time = time.time()

                if id_val:
                    detected_name = text.strip() if text else None
                    card_absent_count = 0

                    if id_val != card_present_id:
                        card_present_id = id_val
                        card_present_text = detected_name
                        card_present_time = current_time
                        self.current_rfid_name = detected_name

                        print(f"RFID card detected: ID={id_val}, Name='{detected_name}'")

                        if self.door_controller:
                            self.door_controller.set_rfid_card(self.current_rfid_name)
                    else:
                         card_present_time = current_time

                else:
                    if card_present_id is not None:
                        card_absent_count += 1

                        if card_absent_count >= 5:
                            print(f"RFID card removed (ID: {card_present_id})")
                            card_present_id = None
                            card_present_text = None
                            self.current_rfid_name = None

                            if self.door_controller:
                                self.door_controller.clear_rfid_card()

                time.sleep(0.2)

            except Exception as e:
                print(f"Error in continuous RFID read: {str(e)}")
                traceback.print_exc()
                time.sleep(1)

        print("Continuous RFID scanning stopped.")

    def start_continuous_read(self):
        """
        Starts the background thread for continuous RFID scanning.

        Sets the `continuous_scan` flag to True and starts the `scan_thread`.

        Returns:
            bool: True if the thread was started, False if already running.
        """
        if self.scan_thread and self.scan_thread.is_alive():
            return False

        self.continuous_scan = True
        self.scan_thread = threading.Thread(target=self.continuous_read)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        return True

    def stop_continuous_read(self):
        """
        Stops the background thread for continuous RFID scanning.

        Sets the `continuous_scan` flag to False and waits for the thread to join.

        Returns:
            bool: Always returns True.
        """
        self.continuous_scan = False
        if self.scan_thread:
            self.scan_thread.join(timeout=2)
        self.scan_thread = None
        return True

    def get_current_rfid_name(self):
        """
        Returns the name/data associated with the currently detected RFID card.

        Returns:
            str or None: The name/data, or None if no card is detected.
        """
        return self.current_rfid_name

    def cleanup(self):
        """Cleans up resources, stops the scan thread, and cleans GPIO."""
        print("Cleaning up RFID Controller...")
        self.stop_continuous_read()
        try:
            GPIO.cleanup()
            print("GPIO cleanup successful.")
        except Exception as e:
            print(f"Error during GPIO cleanup: {e}")

rfid_controller = RFIDController()

def add_rfid_routes(app, door_controller=None):
    """
    Adds RFID-related API endpoints to the provided Flask application.

    Initializes the RFIDController, sets its door controller reference, starts
    the continuous scan, and defines routes for reading, writing, and getting
    the current card name.

    Args:
        app (Flask): The Flask application instance.
        door_controller (DoorController, optional): The main door controller instance.

    Returns:
        function: A cleanup function to be called on application shutdown.
    """
    if door_controller:
        rfid_controller.set_door_controller(door_controller)

    rfid_controller.start_continuous_read()

    @app.route('/rfid/read', methods=['GET'])
    def read_rfid():
        """Handles GET requests to manually read an RFID card."""
        if rfid_controller.is_reading:
            return jsonify({
                'success': False,
                'error': 'RFID operation in progress'
            }), 400

        result = rfid_controller.read_card()
        return jsonify(result)

    @app.route('/rfid/write', methods=['POST'])
    def write_rfid():
        """Handles POST requests to write data to an RFID card."""
        from flask import request

        if rfid_controller.is_reading:
            return jsonify({
                'success': False,
                'error': 'RFID operation in progress'
            }), 400

        data = request.json.get('data', '')
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        result = rfid_controller.write_card(data)
        return jsonify(result)

    @app.route('/current_rfid_name_direct', methods=['GET']) # Renamed to avoid conflict
    def get_current_rfid_name_direct():
        """Handles GET requests to retrieve the current RFID card name (direct from RFID controller)."""
        return jsonify({
            'name': rfid_controller.get_current_rfid_name()
        })

    def cleanup_rfid():
        """Calls the RFID controller's cleanup method."""
        rfid_controller.cleanup()

    return cleanup_rfid