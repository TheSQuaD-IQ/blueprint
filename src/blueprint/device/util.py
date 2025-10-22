from typing import Iterator

from ..systems import System


def embed_systems(*systems: System) -> Iterator[System]:
    """
    embed_systems Embeds each system in the list of systems into the
    hilbert space defined by these systems.

    Parameters
    ----------
    systems : Iterable[System]
        The list of systems to embed.

    Returns
    -------
    tuple[System]
        The list of embedded systems.
    """
    dims = tuple((system.dim for system in systems))
    for ind, system in enumerate(systems):
        yield system.embed(ind, dims)
