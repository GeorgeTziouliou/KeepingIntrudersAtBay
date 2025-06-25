"""
Controls the state of the door lock and manages interactions with RFID data.

This module defines the DoorController class which handles locking/unlocking,
tracking the currently presented RFID card, matching face recognition results
against RFID data, and triggering an auto-lock mechanism. It can also interact
with an optional LCD controller to display status updates.
"""

import threading
import time


class DoorController:
    """
    Manages the door lock status, RFID card interaction, and auto-locking.

    Attributes:
        door_status (dict): Contains the current lock state {'locked': bool}.
        lcd_controller (LCDController, optional): Reference to the LCD controller.
        auto_lock_timer (threading.Timer, optional): Timer for auto-locking.
        current_rfid_card (str, optional): Data read from the current RFID card.
        face_name_matches (dict): Tracks consistency of face detections for matching.
    """
    def __init__(self, lcd_controller=None):
        """
        Initializes the DoorController.

        Args:
            lcd_controller (LCDController, optional): An instance of the LCD
                                                     controller for status display.
                                                     Defaults to None.
        """
        self.door_status = {"locked": True}
        self.lcd_controller = lcd_controller
        self.auto_lock_timer = None
        self.current_rfid_card = None
        self.face_name_matches = {}

    def set_rfid_card(self, rfid_data):
        """
        Stores the data from the currently presented RFID card.

        Also resets the face match consistency tracker.

        Args:
            rfid_data (str): The data (usually a name or identifier) read
                             from the RFID card.
        """
        self.current_rfid_card = rfid_data
        self.face_name_matches = {}

    def clear_rfid_card(self):
        """Clears the stored RFID card data and face match tracker."""
        self.current_rfid_card = None
        self.face_name_matches = {}

    def get_rfid_name(self):
        """
        Retrieves the name/data from the currently stored RFID card.

        Returns:
            str or None: The stripped string data from the card, or None if no
                         card data is currently stored or it's not a string.
        """
        if self.current_rfid_card and isinstance(self.current_rfid_card, str):
            return self.current_rfid_card.strip()
        return None

    def check_face_match(self, detected_face_name):
        """
        Checks if the detected face name matches the current RFID card data.

        Requires the RFID card data and detected face name (case-insensitive)
        to match for known faces. For "Other" faces, it requires the RFID data
        to also be "Other" (case-insensitive) and for the "Other" face to have
        been detected consistently multiple times.

        Args:
            detected_face_name (str): The name identified by the face recognition.

        Returns:
            bool: True if the face matches the RFID requirements, False otherwise.
        """
        rfid_name = self.get_rfid_name()

        if not rfid_name:
            return False

        if not detected_face_name:
            return False

        detected_face_name = detected_face_name.strip()
        rfid_name = rfid_name.strip()

        if detected_face_name in self.face_name_matches:
            self.face_name_matches[detected_face_name] += 1
        else:
            self.face_name_matches[detected_face_name] = 1

        if detected_face_name.lower() == "other":
            if rfid_name.lower() == "other" and self.face_name_matches[detected_face_name] >= 3:
                return True
            else:
                return False

        if detected_face_name.lower() == rfid_name.lower():
            return True

        return False

    def unlock_door(self, auto_lock_delay=30):
        """
        Unlocks the door and starts an auto-lock timer.

        Updates the internal state, cancels any previous auto-lock timer,
        updates the LCD (if available), and schedules the door to lock
        again after the specified delay.

        Args:
            auto_lock_delay (int): Time in seconds before the door automatically
                                   locks again. Defaults to 30.

        Returns:
            bool: Always returns True.
        """
        if self.auto_lock_timer:
            self.auto_lock_timer.cancel()
            self.auto_lock_timer = None

        self.door_status["locked"] = False

        if self.lcd_controller:
            try:
                self.lcd_controller.display_message("Door Unlocked", "Access Granted")
                time.sleep(2)
                self.lcd_controller.update_door_status(False)
            except Exception as e:
                print(f"LCD update error: {str(e)}")

        self.auto_lock_timer = threading.Timer(auto_lock_delay, self.lock_door)
        self.auto_lock_timer.daemon = True
        self.auto_lock_timer.start()

        return True

    def lock_door(self):
        """
        Locks the door.

        Updates the internal state and updates the LCD (if available). Clears
        any reference to an active auto-lock timer.

        Returns:
            bool: Always returns True.
        """
        self.door_status["locked"] = True

        if self.lcd_controller:
            try:
                self.lcd_controller.update_door_status(True)
            except Exception as e:
                print(f"LCD update error: {str(e)}")

        self.auto_lock_timer = None

        return True

    def is_locked(self):
        """
        Checks the current lock status of the door.

        Returns:
            bool: True if the door is locked, False otherwise.
        """
        return self.door_status["locked"]

    def cleanup(self):
        """
        Performs cleanup actions, primarily cancelling any active auto-lock timer.
        """
        if self.auto_lock_timer:
            self.auto_lock_timer.cancel()
            self.auto_lock_timer = None

door_controller = None

def initialize_door_controller(lcd_controller=None):
    """
    Creates and returns a global instance of the DoorController.

    Args:
        lcd_controller (LCDController, optional): An instance of the LCD controller.

    Returns:
        DoorController: The initialized door controller instance.
    """
    global door_controller
    door_controller = DoorController(lcd_controller)
    return door_controller