"""
Self-made logger which prints coloured info, warning, error,
success and debug statements to terminal.
"""


def general(message, type, script):
    preset = f"{script}::" if script != "" else ""
    if type == "info":
        print(f"\033[94m{preset}INFO {message} \033[0m")
    elif type == "warning":
        print(f"\033[93m{preset}WARNING {message} \033[0m")
    elif type == "error":
        print(f"\033[91m{preset}ERROR {message} \033[0m")
    elif type == "success":
        print(f"\033[92m{preset}SUCCESS {message} \033[0m")
    elif type == "debug":
        print(f"\033[35m{preset}DEBUG {message} \033[0m")
    else:
        print(f"\033[91m{preset}ERROR Message type does not exist \033[0m")


def info(message, script=""):
    general(message, "info", script)


def warning(message, script=""):
    general(message, "warning", script)


def error(message, script=""):
    general(message, "error", script)


def success(message, script=""):
    general(message, "success", script)


def debug(message, script=""):
    general(message, "debug", script)
