import signal
from contextlib import contextmanager
from pathlib import Path
from typing import List


def mkdir(path: str, exist_ok: bool = True, parents: bool = True) -> Path:
    path: Path = Path(path)
    path.mkdir(exist_ok=exist_ok, parents=parents)
    return path

def files_in_folder(folder: str, ext: List[str]) -> List[str]:
    if isinstance(ext, str):
        ext = [ext]
    files = [str(f) for f in Path(folder).iterdir() if f.suffix in ext]
    sorted(files)
    return files

@contextmanager
def non_interruptable():
    original_signit_handler = signal.getsignal(signal.SIGINT)

    def ignore_sigint(signum, frame):
        print("SIGINT received. Ignoring")

    try:
        signal.signal(signal.SIGINT, ignore_sigint)
        yield
    finally:
        signal.signal(signal.SIGINT, original_signit_handler)
