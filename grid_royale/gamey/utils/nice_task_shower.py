import time
import datetime as datetime_module

class NiceTaskShower:
    def __init__(self, message: str) -> None:
        self.message = message


    def __enter__(self) -> NiceTaskShower:
        print(f'{self.message}...', end='', flush=True)
        self.start_time = time.monotonic()
        return self


    def dot(self) -> NiceTaskShower:
        print('.', end='', flush=True)

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        end_time = time.monotonic()
        duration = datetime_module.timedelta(seconds=(end_time - self.start_time))
        minutes, seconds = map(int, divmod(duration.total_seconds(), 60))
        word = 'Done' if exc_value is None else 'Failed'
        print(f' {word} in {minutes}:{seconds:02d} seconds.', flush=True)

