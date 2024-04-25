import math 
from typing import Union, Tuple, Callable
from functools import wraps 

from ..base import QuantumSystem

def check_var_validity(
    arg: float,
    argname: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """
    check_valid_variable Checks if the provided variable is a valid numeric variable.

    """
    if not isinstance(arg, float):
        raise ValueError(
            f"The {argname} is expected to be a float, instead got type {type(arg)}."
        )

    if min_value is not None:
        if arg < min_value:
            raise ValueError(
                f"The {argname} must be greater than or equal to {min_value}."
            )
    if max_value is not None:
        if arg > max_value:
            raise ValueError(
                f"The {argname} must be less than or equal to {max_value}."
            )
        
class Fluxonium(QuantumSystem):
    """
    
    
    
    """
