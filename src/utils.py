import os
from typing import Sequence


def filepath(filename: str, subfolder: str = "data/raw") -> str:
    project_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(project_root, subfolder, filename)


def str_floats(sequence: Sequence[float]) -> str:
    if sequence is None:
        return "(None)"
    return f"({', '.join([f'{x:0.02f}' for x in sequence])})"
