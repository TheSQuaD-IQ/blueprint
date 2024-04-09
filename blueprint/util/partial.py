from __future__ import annotations
from typing import Any, Callable, Tuple, Dict

from functools import partial
from inspect import signature


class Partial:
    """
    A class that allows for partial application of functions. Compatible with positional only, positional or keyword, keyword only and variable keyword arguments.
    """

    def __init__(self, /, func: Callable, *args: Any, **keywords: Any) -> None:
        self._req_pos_args = []
        self._pos_args = []

        self._max_keywords = 0
        self._keywords = {}
        self._has_var_keywords = False

        sig = signature(func)
        for name, param in sig.parameters.items():
            if param.kind == param.POSITIONAL_ONLY:
                self._req_pos_args.append(name)
                continue

            if param.kind == param.POSITIONAL_OR_KEYWORD:
                self._pos_args.append(name)

            if param.kind == param.VAR_POSITIONAL:
                raise NotImplementedError(
                    "Variable positional arguments are not supported."
                )

            if param.kind == param.KEYWORD_ONLY:
                self._max_keywords += 1

            if param.kind == param.VAR_KEYWORD:
                self._has_var_keywords = True
                continue

            if param.default is param.empty:
                self._keywords[name] = None

            else:
                self._keywords[name] = param.default

        self._max_num_args = len(self._pos_args)

        self._params = set(self._pos_args)
        self._params.update(tuple(self._keywords))
        self._func = func

        self.set_params(*args, **keywords)

    def __copy__(self) -> Partial:
        """
        __copy__ Copies the partial object.

        Returns
        -------
        Partial
            A copy of the partial object.
        """
        return self.__class__(self._func)

    def __repr__(self) -> str:
        func_repr = repr(self._func)
        param_strs = []
        for param in self.params:
            val = self._keywords.get(param)
            if val is not None:
                param_strs.append(f"{param}={val}")
            else:
                param_strs.append(param)

        param_str = ", ".join(param_strs)
        repr_str = f"Parial({func_repr}, {param_str})"
        return repr_str

    @property
    def req_pos_args(self) -> Tuple[str, ...]:
        """
        req_pos_args Returns the required positional-only arguments of the function. These arguments must be provided as positional arguments
        when calling the function.

        Returns
        -------
        Tuple[str, ...]
            The required positional arguments of the function.
        """
        return tuple(self._req_pos_args)

    @property
    def pos_args(self) -> Tuple[str, ...]:
        """
        pos_args Returns the positional-or-keyword arguments of the function. These arguments can be either provided as positional arguments or as keyword arguments.

        Returns
        -------
        Tuple[str, ...]
            The positional arguments of the function.
        """
        return tuple(self._pos_args)

    @property
    def keyword_args(self) -> Tuple[str, ...]:
        """
        keyword_args Returns the keyword arguments of the function.
        Note that this also includes the positional-or-keyword arguments (see `pos_args` for more details).

        Returns
        -------
        Tuple[str, ...]
            The keyword arguments of the function.
        """
        return tuple(self._keywords)

    @property
    def free_keyword_args(self) -> Tuple[str, ...]:
        """
        free_keywords Returns the free keyword arguments of the function. These are the keyword arguments that have not been assigned a value.

        Returns
        -------
        Tuple[str, ...]
            The free keyword arguments of the function.
        """
        names = (name for name, val in self._keywords.items() if val is None)
        return tuple(names)

    @property
    def keywords(self) -> Dict[str, Any]:
        """
        keywords Returns the keyword arguments of the function.

        Returns
        -------
        Tuple[str, ...]
            The keyword arguments of the function.
        """
        return self._keywords

    @property
    def params(self) -> Tuple[str, ...]:
        """
        params Returns the parameters of the function.

        Returns
        -------
        Tuple[str, ...]
            The parameters of the function.
        """
        names = (*self.req_pos_args, *self.keyword_args)
        return names

    @property
    def free_params(self) -> Tuple[str, ...]:
        """
        free_args Returns the free arguments of the function. These are the positional arguments that have not been assigned a value.

        Returns
        -------
        Tuple[str, ...]
            The free arguments of the function.
        """
        free_params = (*self.req_pos_args, *self.free_keyword_args)
        return free_params

    @property
    def func(self) -> Callable:
        """
        func Returns the function that is being partially applied.

        Returns
        -------
        Callable
            The function that is being partially applied.
        """
        return self._func

    def set_keyword(self, /, name: str, value: Any) -> None:
        """
        set_keyword Sets the value of a keyword argument.

        Parameters
        ----------
        name : str
            The name of the keyword argument.
        value : Any
            The value of the keyword argument.
        """
        if name not in self._keywords:
            raise ValueError(f"Got an unexpected keyword argument {name}.")
        self._keywords[name] = value

    def set_params(self, /, *args, **keywords) -> None:
        """
        set_params Sets the parameters of the function.

        Parameters
        ----------
        args : Any
            The values of the positional arguments.
        keywords : Any
            The values of the keyword arguments.
        """
        num_args = len(args)
        max_num_args = len(self._pos_args)

        if num_args > max_num_args:
            raise ValueError(
                f"Expected at most {max_num_args} positional arguments, got {num_args}."
            )

        for ind, val in enumerate(args):
            name = self._pos_args[ind]
            if name in keywords:
                raise ValueError(f"Got multiple values for argument {name}.")

            self._keywords[name] = val

        for name, val in keywords.items():
            if name not in self._keywords and not self._has_var_keywords:
                raise ValueError(f"Got an unexpected keyword argument {name}.")
            self._keywords[name] = val

    def __call__(self, /, *args: Any, **keywords: Any) -> Any:
        num_args = len(args)

        min_num_args = len(self._req_pos_args)

        num_pos_args = len(self._pos_args)
        max_num_args = min_num_args + num_pos_args

        if num_args < min_num_args:
            raise ValueError(
                f"Expected at least {min_num_args} positional arguments, got {num_args}."
            )
        if num_args > max_num_args:
            raise ValueError(
                f"Expected at most {max_num_args} positional arguments, got {num_args}."
            )

        if not self._has_var_keywords:
            for name in keywords:
                if name not in self._keywords:
                    raise ValueError(f"Got an unexpected keyword argument {name}.")

        _args = list(args)
        _keywords = {**self._keywords, **keywords}

        num_set_args = num_args - min_num_args
        for ind, name in enumerate(self._pos_args):
            if ind < num_set_args:
                if name in keywords:
                    raise ValueError(f"Got multiple values for argument {name}.")
                del _keywords[name]
                continue

            try:
                arg = _keywords.pop(name)
                _args.append(arg)
            except KeyError as exc:
                raise ValueError(
                    f"Missing value for the positional or keyword argument {name}."
                ) from exc

        num_keywords = len(_keywords)
        if num_keywords > self._max_keywords and not self._has_var_keywords:
            raise ValueError(
                f"Got more keyword arguments than expected. Expected at most {self._max_keywords}, got {num_keywords}."
            )

        result = self._func(*_args, **_keywords)
        return result

    def finalize(self, /, *args: Any, **keywords: Any) -> partial:
        """
        finalize Returns a partial function with the given arguments and the parameters previously set.

        Parameters
        ----------
        args : Any
            The positional arguments of the function.
        keywords : Any
            The keyword arguments of the function.

        Returns
        -------
        Any
            The result of the function evaluation.
        """
        num_args = len(args)

        min_num_args = len(self._req_pos_args)
        max_num_args = min_num_args + len(self._pos_args)

        if num_args > max_num_args:
            raise ValueError(
                f"Expected at most {max_num_args} positional arguments, got {num_args}."
            )

        if not self._has_var_keywords:
            for name in keywords:
                if name not in self._keywords:
                    raise ValueError(f"Got an unexpected keyword argument {name}.")

        merged_keywords = {**self._keywords, **keywords}
        set_keywords = {}
        for name, val in merged_keywords.items():
            if val is not None:
                set_keywords[name] = val

        if num_args == 0:
            return partial(self._func, **set_keywords)

        if num_args <= min_num_args:
            return partial(self._func, *args, **set_keywords)

        num_set_args = num_args - min_num_args
        for ind in range(num_set_args):
            name = self._pos_args[ind]
            if name in set_keywords:
                del set_keywords[name]
        return partial(self._func, *args, **set_keywords)
