from typing import Union, Callable

from jax import Array

from ..base.terms import TimeDependentTerm

Numeric = Union[float, complex]
GenNumeric = Union[Numeric, Callable[..., Numeric]]

class Drive:
    """
    Drive A class for representing a drive in the Hamiltonian.
    """

    def __init__(
        self,
        label: str,
        operator: Array,
        prefactor: GenNumeric,
    ) -> None:
        if not isinstance(label, str):
            raise ValueError(
                f"The label must be a string, instead got type {type(label)}."
            )
        self._label = label

        # TODO: potentially it might be better to provide the operator as a method that returns it, instead of saving it.
        if not isinstance(operator, Array):
            raise ValueError(
                f"The operator must be a jax.Array, instead got type {type(operator)}."
            )

        op_dims = len(operator.shape)
        if len(operator.shape) != 2:
            raise ValueError(
                f"The operator must be a 2D array, instead got a {op_dims}-dimensional array."
            )
        dim, other_dim = operator.shape
        if dim != other_dim:
            raise ValueError(
                f"The operator must be a square matrix, instead got a ({dim}x{other_dim}) matrix."
            )

        self._operator = operator
        self._dim = dim

        self._prefactor = TimeDependentTerm(prefactor)

    @property
    def label(self) -> str:
        """
        label Returns the label of the time-dependent term.

        Returns
        -------
        str
            The label of the time-dependent term.
        """
        return self._label
    
    @property
    def dim(self) -> int:
        """
        dim Returns the dimension of the time-dependent term.

        Returns
        -------
        int
            The dimension of the time-dependent term.
        """
        return self._dim
    
    def set_params(self, **params) -> None:
        """
        set_params Sets the parameters of the time-dependent term.

        Parameters
        ----------
        **params
            The parameters of the time-dependent term.
        """
        self._prefactor.set_params(**params)

    def eval_prefactor(self, **params) -> Numeric:
        """
        eval_prefactor Evaluates the prefactor of the time-dependent term.

        Returns
        -------
        Numeric
            The evaluated prefactor.
        """
        return self._prefactor(**params)

    def _get_hamiltonian(self, *, exclude_prefactor: bool = False, **params) -> Array:
        """
        _get_hamiltonian Evaluates the prefactor and returns the Hamiltonian of the coupling term.

        Parameters
        ----------
        exclude_prefactor : bool, optional
            Whether to include the prefactor in front of the quantum operators, by default False

        Returns
        -------
        Array
            The coupling term Hamiltonian.

        Raises
        ------
        ValueError
            If the evaluated prefactor is not a number (float or complex).
        """
        if exclude_prefactor:
            return self._operator

        prefactor = self._prefactor(**params)
        if not isinstance(prefactor, (float, complex)):
            raise ValueError(
                f"The evaluated prefactor must be a number (float or complex), instead got type {type(prefactor)}."
            )

        return prefactor * self._operator

    def get_hamiltonian(self, *, exclude_prefactor: bool = False, **params) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian of the coupling term.

        Parameters
        ----------
        exclude_prefactor : bool, optional
            Whether to include the prefactor in front of the quantum operators, by default False

        Returns
        -------
        Array
            The coupling term Hamiltonian.
        """
        hamiltonian = self._get_hamiltonian(
            exclude_prefactor=exclude_prefactor, **params
        )
        # TODO: consider if we want this term to also have operator processing methods, like transform or expand. If not, maybe merge this method with _get_hamilotinan.

        return hamiltonian
