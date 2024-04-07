from __future__ import annotations
from typing import Any, Callable, Tuple

from functools import partial
from inspect import signature


class Partial:
    """
    A class that allows for partial application of functions. Compatible with positional only, positional or keyword, keyword only and variable keyword arguments.
    """

    def __init__(self, func: Callable, /, *args: Any, **keywords: Any) -> None:
        self._min_num_args = 0
        self._pos_args = []

        self._max_keywords = 0
        self._keywords = {}
        self._keyword_args = []
        self._has_var_keywords = False

        sig = signature(func)
        for name, param in sig.parameters.items():
            if param.kind == param.POSITIONAL_ONLY:
                self._pos_args.append(name)
                self._min_num_args += 1
                continue

            if param.kind == param.POSITIONAL_OR_KEYWORD:
                self._pos_args.append(name)

            if param.kind == param.VAR_POSITIONAL:
                raise NotImplementedError(
                    "Variable positional arguments are not supported."
                )

            if param.kind == param.KEYWORD_ONLY:
                self._keyword_args.append(name)
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

    @property
    def pos_args(self) -> Tuple[str, ...]:
        """
        pos_args Returns the positional arguments of the function. Note that this include both the positional-only and positional-or-keyword arguments.

        Returns
        -------
        Tuple[str, ...]
            The positional arguments of the function.
        """
        return tuple(self._pos_args)

    @property
    def free_args(self) -> Tuple[str, ...]:
        """
        free_args Returns the free arguments of the function. These are the positional arguments that have not been assigned a value.

        Returns
        -------
        Tuple[str, ...]
            The free arguments of the function.
        """
        names = []
        for name in self._pos_args:
            val = self._keywords.get(name)
            if val is None:
                names.append(name)

        return tuple(names)

    @property
    def keyword_args(self) -> Tuple[str, ...]:
        """
        keywords Returns the keyword only arguments of the function.

        Returns
        -------
        Tuple[str, ...]
            The keyword arguments of the function.
        """
        return tuple(self._keyword_args)

    @property
    def free_keyword_args(self) -> Tuple[str, ...]:
        """
        free_keywords Returns the free keyword arguments of the function. These are the keyword arguments that have not been assigned a value.

        Returns
        -------
        Tuple[str, ...]
            The free keyword arguments of the function.
        """
        names = []
        for name in self._keyword_args:
            val = self._keywords[name]
            if val is None:
                names.append(name)

        return tuple(names)

    @property
    def params(self) -> Tuple[str, ...]:
        """
        params Returns the parameters of the function.

        Returns
        -------
        Tuple[str, ...]
            The parameters of the function.
        """
        params = (*self._pos_args, *self._keyword_args)
        return params

    @property
    def free_params(self) -> Tuple[str, ...]:
        """
        free_args Returns the free arguments of the function. These are the positional arguments that have not been assigned a value.

        Returns
        -------
        Tuple[str, ...]
            The free arguments of the function.
        """
        free_params = (*self.free_args, *self.free_keyword_args)
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

        pos_keyword_args = self._pos_args[self._min_num_args :]
        max_num_args = len(pos_keyword_args)

        if num_args > max_num_args:
            raise ValueError(
                f"Expected at most {max_num_args} positional arguments, got {num_args}."
            )

        for ind, val in enumerate(args):
            name = pos_keyword_args[ind]
            if name in keywords:
                raise ValueError(f"Got multiple values for argument {name}.")

            self._keywords[name] = val

        for name, val in keywords.items():
            if name not in self._keywords and not self._has_var_keywords:
                raise ValueError(f"Got an unexpected keyword argument {name}.")
            self._keywords[name] = val

    def __call__(self, /, *args: Any, **keywords: Any) -> Any:
        num_args = len(args)

        if num_args < self._min_num_args:
            raise ValueError(
                f"Expected at least {self._min_num_args} positional arguments, got {num_args}."
            )
        if num_args > self._max_num_args:
            raise ValueError(
                f"Expected at most {self._max_num_args} positional arguments, got {num_args}."
            )

        if not self._has_var_keywords:
            for name in keywords:
                if name not in self._keywords:
                    raise ValueError(f"Got an unexpected keyword argument {name}.")

        _args = list(args)
        _keywords = {**self._keywords, **keywords}

        for ind, name in enumerate(self._pos_args):
            if ind < self._min_num_args:
                continue

            if ind < num_args:
                del _keywords[name]
                continue

            if name not in keywords:
                raise ValueError(f"Got multiple values for argument {name}.")
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
        if num_args > self._max_num_args:
            raise ValueError(
                f"Expected at most {self._max_num_args} positional arguments, got {num_args}."
            )

        if not self._has_var_keywords:
            for name in keywords:
                if name not in self._keywords:
                    raise ValueError(f"Got an unexpected keyword argument {name}.")

        _keywords = {**self._keywords, **keywords}
        if num_args == 0:
            print("Keyword only partial.")
            return partial(self._func, **_keywords)

        if num_args <= self._min_num_args:
            print("Positional-only and keyword partial.")
            return partial(self._func, *args, **_keywords)

        for name in self._pos_args[self._min_num_args : num_args]:
            del _keywords[name]
        return partial(self._func, *args, **_keywords)
