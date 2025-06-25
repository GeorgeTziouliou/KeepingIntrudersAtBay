"""
Controls a 16x2 Character LCD display connected via GPIO pins.

Uses the lgpio library for GPIO access. Handles initialization, sending commands
and data, controlling the backlight, and displaying messages asynchronously
using a message queue and a background thread.

Pin Configuration (BCM):
  LCD_RS = 26
  LCD_E  = 19
  LCD_D4 = 13
  LCD_D5 = 6
  LCD_D6 = 5
  LCD_D7 = 21
  LED_ON = 15 (Backlight control)
"""

import lgpio as GPIO
import time
import threading
import queue

LCD_RS = 26
LCD_E = 19
LCD_D4 = 13
LCD_D5 = 6
LCD_D6 = 5
LCD_D7 = 21
LED_ON = 15

LCD_WIDTH = 16
LCD_CHR = True
LCD_CMD = False

LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0

E_PULSE = 0.0005
E_DELAY = 0.0005


class LCDController:
    """
    Manages communication with the 16x2 LCD display.

    Provides methods for initialization, displaying messages via a thread-safe
    queue, controlling the backlight, and cleaning up GPIO resources.
    """
    def __init__(self):
        """Initializes controller state variables."""
        self.handle = None
        self.is_initialized = False
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None

    def initialize(self):
        """
        Initializes the LCD display hardware.

        Opens the GPIO chip, configures pins as outputs, sends the necessary
        initialization commands to set the LCD to 4-bit mode, 2 lines, and
        turns on the display and backlight.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            self.handle = GPIO.gpiochip_open(0)
            for pin in [LCD_E, LCD_RS, LCD_D4, LCD_D5, LCD_D6, LCD_D7, LED_ON]:
                GPIO.gpio_claim_output(self.handle, pin)

            self._lcd_byte(0x33, LCD_CMD)
            self._lcd_byte(0x32, LCD_CMD)
            self._lcd_byte(0x28, LCD_CMD)
            self._lcd_byte(0x0C, LCD_CMD)
            self._lcd_byte(0x06, LCD_CMD)
            self._lcd_byte(0x01, LCD_CMD)
            time.sleep(E_DELAY)

            self.backlight(True)

            self.is_initialized = True
            print("LCD initialized successfully")
            return True
        except Exception as e:
            print(f"LCD initialization error: {str(e)}")
            self.cleanup()
            return False

    def _lcd_byte(self, bits, mode):
        """
        Sends a single byte to the LCD.

        Handles setting the RS pin for command/data mode and sending the byte
        in two 4-bit nibbles (high then low), pulsing the enable pin after each.

        Args:
            bits (int): The 8-bit byte to send.
            mode (bool): LCD_CMD (False) for command, LCD_CHR (True) for data.
        """
        if not self.handle:
            return

        GPIO.gpio_write(self.handle, LCD_RS, mode)

        GPIO.gpio_write(self.handle, LCD_D4, bool(bits & 0x10))
        GPIO.gpio_write(self.handle, LCD_D5, bool(bits & 0x20))
        GPIO.gpio_write(self.handle, LCD_D6, bool(bits & 0x40))
        GPIO.gpio_write(self.handle, LCD_D7, bool(bits & 0x80))
        self._lcd_toggle_enable()

        GPIO.gpio_write(self.handle, LCD_D4, bool(bits & 0x01))
        GPIO.gpio_write(self.handle, LCD_D5, bool(bits & 0x02))
        GPIO.gpio_write(self.handle, LCD_D6, bool(bits & 0x04))
        GPIO.gpio_write(self.handle, LCD_D7, bool(bits & 0x08))
        self._lcd_toggle_enable()

    def _lcd_toggle_enable(self):
        """Pulses the LCD Enable (E) pin to latch data."""
        if not self.handle:
            return

        time.sleep(E_DELAY)
        GPIO.gpio_write(self.handle, LCD_E, True)
        time.sleep(E_PULSE)
        GPIO.gpio_write(self.handle, LCD_E, False)
        time.sleep(E_DELAY)

    def display_message(self, line1="", line2=""):
        """
        Adds a message to the queue for asynchronous display on the LCD.

        Ensures the LCD is initialized before queueing.

        Args:
            line1 (str): Text for the first line (max 16 chars).
            line2 (str): Text for the second line (max 16 chars).

        Returns:
            bool: True if the message was successfully queued, False otherwise.
        """
        if not self.is_initialized:
            if not self.initialize():
                return False

        try:
            self.message_queue.put((line1, line2))
            return True
        except Exception as e:
            print(f"Error adding message to queue: {str(e)}")
            return False

    def _write_to_lcd(self, line1, line2):
        """
        Writes the provided text lines directly to the LCD hardware.

        Pads lines with spaces to clear previous content.

        Args:
            line1 (str): Text for the first line.
            line2 (str): Text for the second line.
        """
        if not self.handle:
            return

        try:
            line1 = line1.ljust(LCD_WIDTH, " ")
            self._lcd_byte(LCD_LINE_1, LCD_CMD)
            for char in line1:
                self._lcd_byte(ord(char), LCD_CHR)

            line2 = line2.ljust(LCD_WIDTH, " ")
            self._lcd_byte(LCD_LINE_2, LCD_CMD)
            for char in line2:
                self._lcd_byte(ord(char), LCD_CHR)
        except Exception as e:
            print(f"Error writing to LCD: {str(e)}")

    def backlight(self, state):
        """
        Turns the LCD backlight on or off.

        Args:
            state (bool): True to turn backlight on, False to turn off.
        """
        if not self.handle:
            return

        GPIO.gpio_write(self.handle, LED_ON, state)

    def start_message_thread(self):
        """Starts the background thread that processes the message queue."""
        if self.thread and self.thread.is_alive():
            return

        self.running = True
        self.thread = threading.Thread(target=self._message_loop)
        self.thread.daemon = True
        self.thread.start()

    def _message_loop(self):
        """
        The target function for the message processing thread.

        Continuously waits for messages in the queue (with a timeout) and
        calls `_write_to_lcd` to display them. Stops when `self.running` is False.
        """
        while self.running:
            try:
                line1, line2 = self.message_queue.get(timeout=1)
                self._write_to_lcd(line1, line2)
                self.message_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in message loop: {str(e)}")

    def update_door_status(self, is_locked):
        """
        Displays the current door lock status on the LCD.

        Convenience method that queues a status message.

        Args:
            is_locked (bool): True if the door is locked, False otherwise.
        """
        status = "LOCKED" if is_locked else "UNLOCKED"
        self.display_message("Door Status:", status)

    def cleanup(self):
        """
        Cleans up resources used by the LCD controller.

        Stops the message thread, clears the LCD display, turns off the
        backlight, and closes the GPIO chip handle.
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        if self.handle:
            try:
                self._lcd_byte(0x01, LCD_CMD)
                self.backlight(False)
                GPIO.gpiochip_close(self.handle)
            except:
                pass
            self.handle = None
            self.is_initialized = False

lcd_controller = LCDController()