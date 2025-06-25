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

# LCD parameters
LCD_WIDTH = 16
LCD_CHR = True
LCD_CMD = False

LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005


class LCDController:
    def __init__(self):
        self.handle = None
        self.is_initialized = False
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None

    def initialize(self):
        """Initialize the LCD display"""
        try:
            self.handle = GPIO.gpiochip_open(0)  # Open GPIO chip 0
            for pin in [LCD_E, LCD_RS, LCD_D4, LCD_D5, LCD_D6, LCD_D7, LED_ON]:
                GPIO.gpio_claim_output(self.handle, pin)  # Set pins as output

            self._lcd_byte(0x33, LCD_CMD)  # Initialize
            self._lcd_byte(0x32, LCD_CMD)  # Set to 4-bit mode
            self._lcd_byte(0x28, LCD_CMD)  # 2-line, 5x7 matrix
            self._lcd_byte(0x0C, LCD_CMD)  # Display on, cursor off
            self._lcd_byte(0x06, LCD_CMD)  # Auto increment cursor
            self._lcd_byte(0x01, LCD_CMD)  # Clear display
            time.sleep(E_DELAY)

            # Turn on backlight
            self.backlight(True)

            self.is_initialized = True
            print("LCD initialized successfully")
            return True
        except Exception as e:
            print(f"LCD initialization error: {str(e)}")
            self.cleanup()
            return False

    def _lcd_byte(self, bits, mode):
        """Send byte to data pins"""
        if not self.handle:
            return

        GPIO.gpio_write(self.handle, LCD_RS, mode)

        # High nibble
        GPIO.gpio_write(self.handle, LCD_D4, bool(bits & 0x10))
        GPIO.gpio_write(self.handle, LCD_D5, bool(bits & 0x20))
        GPIO.gpio_write(self.handle, LCD_D6, bool(bits & 0x40))
        GPIO.gpio_write(self.handle, LCD_D7, bool(bits & 0x80))
        self._lcd_toggle_enable()

        # Low nibble
        GPIO.gpio_write(self.handle, LCD_D4, bool(bits & 0x01))
        GPIO.gpio_write(self.handle, LCD_D5, bool(bits & 0x02))
        GPIO.gpio_write(self.handle, LCD_D6, bool(bits & 0x04))
        GPIO.gpio_write(self.handle, LCD_D7, bool(bits & 0x08))
        self._lcd_toggle_enable()

    def _lcd_toggle_enable(self):
        """Toggle enable pin"""
        if not self.handle:
            return

        time.sleep(E_DELAY)
        GPIO.gpio_write(self.handle, LCD_E, True)
        time.sleep(E_PULSE)
        GPIO.gpio_write(self.handle, LCD_E, False)
        time.sleep(E_DELAY)

    def display_message(self, line1="", line2=""):
        """Display a message on the LCD"""
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
        """Actually write the message to the LCD"""
        if not self.handle:
            return

        try:
            # Display on line 1
            line1 = line1.ljust(LCD_WIDTH, " ")
            self._lcd_byte(LCD_LINE_1, LCD_CMD)
            for char in line1:
                self._lcd_byte(ord(char), LCD_CHR)

            # Display on line 2
            line2 = line2.ljust(LCD_WIDTH, " ")
            self._lcd_byte(LCD_LINE_2, LCD_CMD)
            for char in line2:
                self._lcd_byte(ord(char), LCD_CHR)
        except Exception as e:
            print(f"Error writing to LCD: {str(e)}")

    def backlight(self, state):
        """Control the LCD backlight"""
        if not self.handle:
            return

        GPIO.gpio_write(self.handle, LED_ON, state)

    def start_message_thread(self):
        """Start the thread that processes messages"""
        if self.thread and self.thread.is_alive():
            return

        self.running = True
        self.thread = threading.Thread(target=self._message_loop)
        self.thread.daemon = True
        self.thread.start()

    def _message_loop(self):
        """Thread that processes messages from the queue"""
        while self.running:
            try:
                # Get message with timeout to allow checking running flag
                line1, line2 = self.message_queue.get(timeout=1)
                self._write_to_lcd(line1, line2)
                self.message_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in message loop: {str(e)}")

    def update_door_status(self, is_locked):
        """Update LCD with door status"""
        status = "LOCKED" if is_locked else "UNLOCKED"
        self.display_message("Door Status:", status)

    def cleanup(self):
        """Clean up GPIO resources"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        if self.handle:
            try:
                self._lcd_byte(0x01, LCD_CMD)  # Clear display
                self.backlight(False)
                GPIO.gpiochip_close(self.handle)
            except:
                pass
            self.handle = None
            self.is_initialized = False


lcd_controller = LCDController()