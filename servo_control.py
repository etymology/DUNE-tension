from maestro import Controller
ttyStr="/dev/ttyACM1"

maestro = Controller(ttyStr)

def pluck_string():
    if not maestro.faulted:
        maestro.runScriptSub(0)
    else:
        print("Maestro is faulted. Cannot pluck the string.")

def is_maestro_active():   
    try:
        maestro = Controller(ttyStr)
        return not maestro.faulted
    except Exception as e:
        print(f"Failed to initialize Maestro controller: {e}")
        exit(1)