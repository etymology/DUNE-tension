import requests
import time
from random import gauss

TENSION_SERVER_URL = "http://192.168.137.1:5000"
IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3


def read_tag(tag_name):
    """
    Send a GET request to read the value of a PLC tag.
    """
    url = f"{TENSION_SERVER_URL}/tags/{tag_name}"
    # print(f"Attempting to read from URL: {url}")  # Debugging statement
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # print(response.json())
            return response.json()[tag_name][1]
        else:
            return {
                "error": "Failed to read tag",
                "status_code": response.status_code,
            }
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_xy():
    """Get the current position of the tensioning system."""
    x = read_tag("X_axis.ActualPosition")
    y = read_tag("Y_axis.ActualPosition")
    return x, y


def get_state() -> dict[str, list]:
    """Get the current state of the tensioning system."""
    return read_tag("STATE")


def get_movetype() -> int:
    """Get the current move type of the tensioning system."""
    movetype = read_tag("MOVE_TYPE")
    return movetype


def write_tag(tag_name, value):
    """
    Send a POST request to write a value to a PLC tag.
    """
    url = f"{TENSION_SERVER_URL}/tags/{tag_name}"
    # print(f"Attempting to write to URL: {url}")  # Debugging statement
    payload = {"value": value}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": "Failed to write tag",
                "status_code": response.status_code,
            }
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def goto_xy(x_target: float, y_target: float):
    """Move the winder to a given position."""
    # current_x, current_y = self.get_xy()
    if x_target < 0 or x_target > 7174 or y_target < 0 or y_target > 2680:
        print(
            f"Motion target {x_target},{y_target} out of bounds. Please enter a valid position."
        )
        return False
    current_state = get_state()
    while current_state != IDLE_STATE:
        current_state = get_state()
    write_tag("MOVE_TYPE", IDLE_MOVE_TYPE)
    write_tag("STATE", IDLE_STATE)
    write_tag("X_POSITION", x_target)
    write_tag("Y_POSITION", y_target)
    write_tag("MOVE_TYPE", XY_MOVE_TYPE)

    while get_movetype() == XY_MOVE_TYPE:
        time.sleep(0.001)
    return True


def increment(increment_x, increment_y):
    x, y = get_xy()
    goto_xy(x + increment_x, y + increment_y)


def wiggle(step):
    """Wiggle the winder by a given step size."""
    increment(0, gauss(0, step))


def is_web_server_active():
    """
    Check if a web server is active by sending a HTTP GET request.
    """
    try:
        return 200 <= requests.get(TENSION_SERVER_URL, timeout=3).status_code < 500
    except requests.RequestException as e:
        print(f"An error occurred while checking the server: {e}")
        return False


# ---------------------------------------------------------------------------
# Spoofing utilities
# ---------------------------------------------------------------------------

# Track spoofed position so that movement functions can update it
_SPOOF_XY = [3000.0, 1300.0]


def spoof_get_xy() -> tuple[float, float]:
    """Return the current spoofed XY position."""
    return tuple(_SPOOF_XY)


def spoof_goto_xy(x_target: float, y_target: float) -> bool:
    """Pretend to move the winder and update the spoofed position."""
    # Reuse bounds check from :func:`goto_xy` for consistency
    if x_target < 0 or x_target > 7174 or y_target < 0 or y_target > 2680:
        print(f"[spoof] Motion target {x_target},{y_target} out of bounds.")
        return False

    print(f"[spoof] Moving to {x_target},{y_target}")
    _SPOOF_XY[0] = x_target
    _SPOOF_XY[1] = y_target
    return True


def spoof_wiggle(step: float) -> bool:
    """Pretend to wiggle the winder."""
    print(f"[spoof] Wiggling by ±{step} mm")
    return True
