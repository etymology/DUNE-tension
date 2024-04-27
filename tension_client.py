import requests
from time_decorator import timer
from maestro import Controller

TENSION_SERVER_URL = 'http://192.168.137.1:5000'

IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 3


@timer
def read_tag(tag_name, base_url=TENSION_SERVER_URL):
    """ Function to read the value of a given tag from the server. """
    try:
        response = requests.get(f"{base_url}/tags/{tag_name}")
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': response.json().get('error', 'Unknown error')}
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


@timer
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


def get_position() -> tuple[float, float]:
    """ Get the current position of the tensioning system. """
    x = read_tag("X_axis.ActualPosition")
    y = read_tag("Y_axis.ActualPosition")
    return x, y


def get_state() -> int:
    """ Get the current state of the tensioning system. """
    state = read_tag("STATE")
    return state


def wait_until_idle():
    """ Wait until the tensioning system is in the idle state. """
    while get_state() != IDLE_STATE:
        pass


def get_movetype() -> int:
    """ Get the current move type of the tensioning system. """
    movetype = read_tag("MOVE_TYPE")
    return movetype


def goto_position(x: float, y: float):
    state = get_state()
    movetype = get_movetype()
    if state == IDLE_STATE and movetype == IDLE_MOVE_TYPE:
        """ Move the tensioning system to a given position. """
        set_x_result = write_tag("X_POSITION", x)
        set_y_result = write_tag("Y_POSITION", y)
        set_movetype_result = write_tag(
            TENSION_SERVER_URL, "MOVE_TYPE", XY_MOVE_TYPE)
        return set_x_result, set_y_result, set_movetype_result
    else:
        print("Cannot move the system while it is in motion.")


def pluck_string():
    maestro = Controller()
    maestro.runScriptSub(0)


if __name__ == "__main__":
    starting_x, starting_y = get_position()
    print(f"Current position: x={starting_x}, y={starting_y}")
    print("Moving to position x+5, y+5...")
    goto_position(starting_x + 5, starting_y + 5)
    print("Waiting for the system to become idle...")
    wait_until_idle()
    print("returning to starting position...")
    goto_position(starting_x, starting_y)
    wait_until_idle()
    pluck_string()
    print("Done!")
