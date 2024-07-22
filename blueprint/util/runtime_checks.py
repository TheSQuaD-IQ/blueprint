from typing import Any, Type


def check_type(var: Any, name: str, *valid_types: Type) -> None:
    """
    confirm_type Checks if the variable is of the expected type.

    Parameters
    ----------
    var : Any
        The variable to be checked.
    name : str
        The name of the variable.
    valid_types : Type
        The expected types of the variable

    Raises
    ------
    ValueError
        If the variable is not of the same types as
        the list of the valid types provided.
    """
    if not isinstance(var, valid_types):
        var_type = type(var)
        valid_typestrings = ", ".join(valid_types)
        raise ValueError(
            f"{name} is expected to be {valid_typestrings}."
            f"Instead, received a variable of type {var_type}."
        )


def check_valid_interval(
    var: float,
    name: str,
    left_bound: float | None = None,
    right_bound: float | None = None,
    right_open: bool = False,
    left_open: bool = False,
) -> None:
    """
    check_valid_interval Checks if the variable is within the specified interval.

    Parameters
    ----------
    var : float
        The variable to be checked.
    name : str
        The name of the variable.
    left_bound : float | None, optional
        The left bound of the interval, by default None corresponding to -inf
    right_bound : float | None, optional
        The right bound of the interval, by default None corresponding to +inf
    right_open : bool, optional
        If the right bound is open, by default False
    left_open : bool, optional
        If the left bound is open, by default False

    Raises
    ------
    ValueError
        If the variable is not within the specified interval.
    """
    if left_bound is not None:
        if var < left_bound:
            comparison_str = "greater than" if left_open else "greater than or equal to"
            raise ValueError(f"{name} is expected to be {comparison_str} {left_bound}.")
        if left_open and var == left_bound:
            raise ValueError(
                f"{name} is expected to be greater than or equal to {left_bound}."
            )

    if right_bound is not None:
        if var > right_bound:
            comparison_str = "smaller than" if left_open else "smaller than or equal to"
            raise ValueError(f"{name} is expected to be {comparison_str} {left_bound}.")
        if right_open and var == right_bound:
            raise ValueError(
                f"{name} is expected to be smaller than or equal to {left_bound}."
            )


def check_positive(var: float, name: str) -> None:
    """
    check_positive Checks if the variable is positive.

    Parameters
    ----------
    var : Any
        The variable to be checked.

    Raises
    ------
    ValueError
        If the variable is not positive.
    """
    check_valid_interval(var, name, left_bound=0.0)
