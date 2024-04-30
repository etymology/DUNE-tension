import sys
import termios
import tty
#from app import TensionTestingApp


class UIManager:
    def __init__(self, application):
        self.application = application

    def display_menu(self):
        """Display the main menu options to the user."""
        print("\nAvailable Actions:")
        print("d - Display available sound devices")
        print("r - Record audio and analyze")
        print("w - Go to specific wire")
        print("n - Go to next wire")
        print("u - Go to previous wire")
        print("c - Calibrate")
        print("p - Change parameters")
        print("q - Quit the application")

    def get_user_choice(self):
        """Capture a single character of input from the user."""
        print("\nEnter your choice:")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            choice = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return choice.lower()

    def process_choice(self, choice):
        """Process the user's choice and call the corresponding method in the application."""
        actions = {
            'd': self.application.handle_select_device,
            'r': self.application.handle_auto_record_log,
            'w': self.application.handle_goto_input_wire,
            'n': self.application.handle_goto_next_wire,
            'u': self.application.handle_goto_prev_wire,
            'c': self.application.handle_calibration,
            'p': self.application.handle_change_variables,
            'q': self.application.handle_quit
        }
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid input. Please select a valid option.")

    def run(self):
        """Run the main UI loop."""
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            if choice == 'q':
                break
            self.process_choice(choice)
        self.application.handle_quit()
