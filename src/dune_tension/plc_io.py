import requests
import threading
import time
from random import gauss

# Amount of travel, in mm, assumed to be lost when reversing X direction
BACKLASH_DEADZONE = 0.5

# Track our best guess of the true position, accounting for backlash
_TRUE_XY = [6300.0, 200.0]

# Track the last X movement direction and remaining deadzone to take up
_LAST_X_DIR = 0
_X_DEADZONE_LEFT = 0.0

TENSION_SERVER_URL = "http://192.168.137.1:5000"
IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3


def read_tag(tag_name, *, timeout: float = 10.0, retry_interval: float = 0.1) -> float:
    """Read the value of a PLC tag with basic retry logic.

    Occasionally the PLC server returns malformed JSON where the list under
    ``tag_name`` does not contain the expected value at index ``1``.  In this
    situation the function will retry until ``timeout`` seconds have elapsed.
    ``timeout`` and ``retry_interval`` are in seconds.
    """

    url = f"{TENSION_SERVER_URL}/tags/{tag_name}"

    # try:
    #     response = requests.get(url)
    #     if response.status_code == 200:
    #         resp = response.json()[tag_name]
    #         return resp[1]
    #     else:
    #         return {
    #             "error": "Failed to read tag",
    #             "status_code": response.status_code,
    #         }
    # except requests.exceptions.RequestException as e:
    #     return {"error": str(e)}

    end_time = time.monotonic() + timeout
    last_error: dict | None = None

    while time.monotonic() < end_time:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                try:
                    data = response.json()[tag_name]
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        return data[1]
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    last_error = {"error": f"Malformed response: {exc}"}
            else:
                last_error = {
                    "error": "Failed to read tag",
                    "status_code": response.status_code,
                }
        except requests.exceptions.RequestException as exc:
            last_error = {"error": str(exc)}

        time.sleep(retry_interval)

    return last_error if last_error is not None else {"error": "Read timeout"}


def get_xy():
    """Get the current position of the tensioning system."""
    x = read_tag("X_axis.ActualPosition")
    y = read_tag("Y_axis.ActualPosition")
    return x, y


def get_cached_xy() -> tuple[float, float]:
    """Return the internally tracked XY position.

    ``goto_xy`` keeps ``_TRUE_XY`` updated whenever a move command is issued.
    This function exposes that cached value so that callers can avoid an
    unreliable read from the PLC server.
    """

    return tuple(_TRUE_XY)


def get_state() -> int:
    """Get the current state of the tensioning system."""
    return int(read_tag("STATE"))


def get_movetype() -> int:
    """Get the current move type of the tensioning system."""
    movetype = read_tag("MOVE_TYPE")
    return int(movetype)


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


def goto_xy(x_target: float, y_target: float, *,speed=300, deadzone: float = BACKLASH_DEADZONE):
    """Move the winder to a given position.

    When reversing X direction, assume the first ``deadzone`` mm of travel does
    not result in motion and track the true position accordingly.
    """

    global _TRUE_XY, _LAST_X_DIR, _X_DEADZONE_LEFT

    if not (1000 < x_target < 7174 and 0 < y_target < 2680):
        print(
            f"Motion target {x_target},{y_target} out of bounds. Please enter a valid position."
        )
        return False

    # ------------------------------------------------------------
    # Backlash compensation bookkeeping
    # ------------------------------------------------------------
    delta_x = x_target - _TRUE_XY[0]
    direction = 1 if delta_x > 0 else -1 if delta_x < 0 else 0

    if direction != 0 and direction != _LAST_X_DIR:
        _X_DEADZONE_LEFT = deadzone
        _LAST_X_DIR = direction

    move_x = abs(delta_x)
    actual_move_x = 0.0
    if direction != 0:
        if _X_DEADZONE_LEFT > 0:
            if move_x <= _X_DEADZONE_LEFT:
                _X_DEADZONE_LEFT -= move_x
            else:
                actual_move_x = direction * (move_x - _X_DEADZONE_LEFT)
                _X_DEADZONE_LEFT = 0.0
        else:
            actual_move_x = delta_x

    _TRUE_XY[0] += actual_move_x
    _TRUE_XY[1] = y_target

    # ------------------------------------------------------------
    # Command the PLC move
    # ------------------------------------------------------------
    current_state = get_state()
    while current_state != IDLE_STATE:
        time.sleep(0.1)
        current_state = get_state()
    set_speed(speed)
    write_tag("MOVE_TYPE", IDLE_MOVE_TYPE)
    write_tag("STATE", IDLE_STATE)
    write_tag("X_POSITION", x_target)
    write_tag("Y_POSITION", y_target)
    write_tag("MOVE_TYPE", XY_MOVE_TYPE)

    while get_movetype() == XY_MOVE_TYPE:
        time.sleep(0.1)
    return True

def reset_plc():
    """Reset the PLC to its initial state."""
    write_tag("MOVE_TYPE", IDLE_MOVE_TYPE)
    write_tag("STATE", IDLE_STATE)
    set_speed(0)  # Reset speed to a default value

def increment(increment_x, increment_y):
    # Use the cached position to avoid reading tags when possible
    x, y = get_cached_xy()
    goto_xy(x + increment_x, y + increment_y)


def wiggle(step):
    """Wiggle the winder by a given step size in a background thread."""

    def _do_wiggle() -> None:
        y_wiggle = gauss(0, step)
        increment(0, y_wiggle)
        print(f"Wiggling by {y_wiggle} mm")

    threading.Thread(target=_do_wiggle, daemon=True).start()
    return True

def set_speed(speed: float = 300) -> bool:
    """Set the speed of the winder.

    This function writes the desired speed to the PLC server.
    """
    if not (0 <= speed <= 1000):
        print(f"Speed {speed} out of bounds. Please enter a value between 0 and 100.")
        return False

    response = write_tag("XY_SPEED", speed)
    if "error" in response:
        print(f"Failed to set speed: {response['error']}")
        return False

    print(f"Speed set to {speed}")
    return True
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
_SPOOF_XY = [6300.0, 200.0]


def spoof_get_xy() -> tuple[float, float]:
    """Return the current spoofed XY position."""
    return tuple(_SPOOF_XY)


def spoof_goto_xy(x_target: float, y_target: float) -> bool:
    """Pretend to move the winder and update the spoofed position."""
    # Reuse bounds check from :func:`goto_xy` for consistency
    if x_target < 0 or x_target > 7174 or y_target < 0 or y_target > 2680:
        print(f"[spoof] Motion target {x_target},{y_target} out of bounds.")

    print(f"[spoof] Moving to {x_target},{y_target}")
    _SPOOF_XY[0] = x_target
    _SPOOF_XY[1] = y_target
    return True


def spoof_wiggle(step: float) -> bool:
    """Pretend to wiggle the winder."""
    print(f"[spoof] Wiggling by ±{step} mm")
    return True
