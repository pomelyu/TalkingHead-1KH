from datetime import datetime


def get_datetime() -> str:
    now = datetime.now()
    formatted = now.strftime("%Y_%m_%d_%H_%M_%S")
    return formatted
