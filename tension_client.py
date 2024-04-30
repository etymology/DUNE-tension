import requests
from maestro import Controller

TENSION_SERVER_URL = 'http://192.168.137.1:5000'

IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3

def wiggle_generator():
    i = 0  # Start with the first element index
    while True:  # This makes it an infinite generator
        # Calculate value using the formula derived
        yield (-1)**(i // 2 + 1) * (i // 2 + 1) * 0.1
        i += 1


def read_tag(tag_name, base_url=TENSION_SERVER_URL):
    """ Function to read the value of a given tag from the server. """
    try:
        response = requests.get(f"{base_url}/tags/{tag_name}")
        if response.status_code == 200:
            # Tags are a dictionary keyed by name, value is the second element in the list
            return response.json()[tag_name][1]
        else:
            return {'error': response.json().get('error', 'Unknown error')}
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}



def write_tag(tag_name, value, base_url=TENSION_SERVER_URL):
    """ Function to write a value to a given tag on the server. """
    try:
        response = requests.post(
            f"{base_url}/tags/{tag_name}", json={'value': value})
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': response.json().get('error', 'Unknown error')}
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


class Tensiometer:
    def __init__(self):
        self.maestro = Controller()

    def __exit__(self):
        self.maestro.close()

    def pluck_string(self):
        if not self.maestro.faulted:
            self.maestro.runScriptSub(0)
        else:
            print("Maestro is faulted. Cannot pluck the string.")

    @staticmethod
    def get_xy() -> tuple[float, float]:
        """ Get the current position of the tensioning system. """
        x = read_tag("X_axis.ActualPosition")
        y = read_tag("Y_axis.ActualPosition")
        return x, y

    @staticmethod
    def get_state() -> dict[str, list]:
        """ Get the current state of the tensioning system. """
        state = read_tag("STATE")
        return state

    @staticmethod
    def wait_until_state_movetype(target_state: int, target_move_type: int):
        """ Wait until the tensioning system is in the idle state. """
        while read_tag("STATE") != target_state and read_tag("MOVE_TYPE") != target_move_type:
            pass
        while read_tag("STATE") != IDLE_STATE and read_tag("MOVE_TYPE") != IDLE_MOVE_TYPE:
            pass

    @staticmethod
    def get_movetype() -> int:
        """ Get the current move type of the tensioning system. """
        movetype = read_tag("MOVE_TYPE")
        return movetype

    def goto_xy(self, x_target: float, y_target: float):
        if self.get_state() == IDLE_STATE and self.get_movetype() == IDLE_MOVE_TYPE:
            """ Move the winder to a given position. """
            write_tag("X_POSITION", x_target)
            write_tag("Y_POSITION", y_target)
            write_tag("MOVE_TYPE", XY_MOVE_TYPE)
            self.wait_until_state_movetype(XY_STATE,XY_MOVE_TYPE)
        else:
            print("Cannot move the system while it is in motion.")

    def increment(self, increment_x, increment_y):
        x, y = self.get_xy()
        self.goto_xy(x+increment_x, y+increment_y)


if __name__ == "__main__":
    t = Tensiometer()
    starting_x, starting_y = t.get_xy()
    movetype = t.get_movetype()
    state = t.get_state()
    print(
        f"Current position: x={starting_x}, y={starting_y}\nstate={state}, movetype={movetype}")

    wg = wiggle_generator()  # Create an instance of the generator
    for _ in range(12):  # Get the first 10 elements
        t.goto_xy(0,starting_y+next(wg))
        

