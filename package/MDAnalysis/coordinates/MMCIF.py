# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
"""
MMCIF structure files in MDAnalysis --- :mod:`MDAnalysis.coordinates.MMCIF`
===========================================================================

.. versionadded:: 2.11.0

MDAnalysis reads coordinates from MMCIF (macromolecular Crystallographic
Information File) files, also known as PDBx/mmCIF format, using the
`gemmi <https://gemmi.readthedocs.io>`_ library as a backend. MMCIF is a
more modern and flexible alternative to the PDB format, capable of storing
detailed structural and experimental data about biological macromolecules.

MMCIF files use a structured, tabular format with key-value pairs to store
both coordinate and atom information. Each model of a multi-model file
(e.g. an NMR ensemble) is read as one frame, with the topology taken from the
first model. If the models have different numbers of atoms, only the first
model is read and a warning is issued.

The reader automatically detects if the structure contains placeholder unit
cell information (usually the case for cryoEM structures, where cell
parameters are (1, 1, 1, 90, 90, 90)) and sets dimensions to ``None``
in that case.

Basic usage
-----------

.. code-block:: python

    import MDAnalysis as mda

    u = mda.Universe("structure.cif")

    # or from a compressed file
    u = mda.Universe("structure.cif.gz")

    # or from a structure already read with gemmi
    import gemmi

    u = mda.Universe(gemmi.read_structure("structure.cif"))

See Also
--------
* `wwPDB MMCIF Resources <http://mmcif.wwpdb.org>`_
* `Gemmi library documentation <https://gemmi.readthedocs.io>`_

Classes
-------

.. autoclass:: MMCIFReader
   :members:
   :inherited-members:

"""

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..lib import util
from ..lib.util import store_init_arguments
from . import base

if TYPE_CHECKING:
    from gemmi import FlatStructure, Structure

try:
    import gemmi

    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

logger = logging.getLogger("MDAnalysis.coordinates.MMCIF")


def _read_gemmi_structure(filename: str | Path) -> "Structure":
    # This function exists because of some lacking methods in the gemmi Python API.
    # Within gemmi in C++, one can call `read_structure` and in-memory, string, and filepath
    # arguments will all be accepted:
    # https://github.com/project-gemmi/gemmi/blob/4416e298f204b7b57bf5b3051d7efd4fe02957cf/include/gemmi/mmread.hpp#L86

    # However, for MDA to similarly accept common input types like streams (open File-like objs and StringIO objs)
    # as well as pathlib.Path() objects, we have to use the Python API methods available currently (as of 0.7.3)
    # with a string as a common target for all input types.
    # For this, we call gemmi.cif.read_string (https://gemmi.readthedocs.io/en/latest/cif.html#reading) to handle CIF
    # strings and gemmi.read_pdb_string to handle PDB strings (no one method can handle both formats currently Py-side)

    # openany() is called instead of passing file paths (when available) differently from streams;
    # even though reading the file into a string is less efficient, this is easier to maintain.

    # If the gemmi Python API is extended, this function can be simplified/removed and replaced with something like
    # gemmi.read_structure
    with util.openany(filename) as f:
        content_as_str = f.read()
    try:
        # String -> Doc -> Block -> Structure
        # making Structure from first Block in Document as is done internally in gemmi:
        # https://github.com/project-gemmi/gemmi/blob/4416e298f204b7b57bf5b3051d7efd4fe02957cf/include/gemmi/mmcif.hpp#L32
        return gemmi.make_structure_from_block(
            gemmi.cif.read_string(content_as_str)[0]
        )
    except IndexError:
        # the document parsed as CIF but contains no data blocks (e.g. an
        # empty or comment-only file), so it cannot be PDB content either
        raise ValueError(
            f"No data blocks found in mmCIF content of {filename}"
        ) from None
    except ValueError as e:
        try:
            return gemmi.read_pdb_string(content_as_str)
        except (ValueError, RuntimeError):
            # gemmi raises RuntimeError for unparseable PDB content;
            # re-raise the mmCIF error since that is the primary format here
            raise e


class _ParsedMMCIF(NamedTuple):
    """Everything the parser and reader need from one gemmi parse"""

    #: atom table of every model, models in file order, strings as bytes
    flat: "FlatStructure"
    #: number of atoms in the first model (its rows come first in `flat`)
    n_atoms: int
    n_models: int
    #: whether every model has `n_atoms` atoms, so models can be frames
    same_size: bool
    #: unit cell as [a, b, c, alpha, beta, gamma]
    cell: np.ndarray


# The most recent (file key, parse) pair. A Universe built from a file runs
# MMCIFParser and then MMCIFReader on it, and this lets both share one parse.
_last_parse = (None, None)


def _is_gemmi_structure(thing) -> bool:
    return HAS_GEMMI and isinstance(thing, gemmi.Structure)


def _file_key(filename) -> tuple | None:
    """Identify an on-disk file by its path, modification time and size;
    ``None`` for streams and gemmi objects, which are never cached"""
    if util.isstream(filename) or not isinstance(filename, (str, os.PathLike)):
        return None
    stat = os.stat(filename)
    return os.path.realpath(filename), stat.st_mtime_ns, stat.st_size


def _parse_mmcif(filename) -> _ParsedMMCIF:
    """Parse a file, stream or ``gemmi.Structure`` into a `_ParsedMMCIF`,
    reusing the previous result if it was for the same, unchanged file"""
    global _last_parse
    key = _file_key(filename)
    if key is not None and _last_parse[0] == key:
        return _last_parse[1]

    if _is_gemmi_structure(filename):
        structure = filename
    else:
        structure = _read_gemmi_structure(filename)
    flat = gemmi.FlatStructure(structure)
    flat.strings_as_numbers = False

    sizes = [model.count_atom_sites() for model in structure]
    parsed = _ParsedMMCIF(
        flat=flat,
        n_atoms=sizes[0] if sizes else 0,
        n_models=len(sizes),
        same_size=len(set(sizes)) == 1,
        cell=np.array(
            [
                getattr(structure.cell, name)
                for name in ("a", "b", "c", "alpha", "beta", "gamma")
            ]
        ),
    )
    if key is not None:
        _last_parse = (key, parsed)
    return parsed


class MMCIFReader(base.ReaderBase):
    """Reads from an MMCIF file using :mod:`gemmi` as a backend.

    The *filename* argument accepts a file path, a compressed ``.cif.gz`` file,
    a stream/file-like object, or a ``gemmi.Structure``.

    Notes
    -----

    Each model of the structure is read as one frame. If the models have
    different numbers of atoms, only the first model is read (and a warning is
    thrown). Also, if the structure has a placeholder "CRYST1" record
    (1, 1, 1, 90, 90, 90), it's set to ``None`` instead.

    .. versionadded:: 2.11.0
    """

    format = ["cif", "cif.gz", "mmcif", "mmcif.gz"]
    units = {"time": None, "length": "Angstrom"}

    @store_init_arguments
    def __init__(self, filename, convert_units=True, **kwargs):
        # initialise the base first, so that its __del__ works if we raise
        super(MMCIFReader, self).__init__(
            filename, convert_units=convert_units, **kwargs
        )
        if not HAS_GEMMI:
            errmsg = "MMCIFReader: To read mmCIF files, please install gemmi"
            raise ImportError(errmsg)
        parsed = _parse_mmcif(filename)
        if parsed.same_size:
            self.n_frames = parsed.n_models
        else:
            self.n_frames = 1
            if parsed.n_models > 1:
                wmsg = (
                    f"File {self.filename} has {parsed.n_models} "
                    "differently-sized models, but only the first one will "
                    "be read"
                )
                warnings.warn(wmsg)
                logger.warning(wmsg)
        self.n_atoms = parsed.n_atoms
        self._positions = np.asarray(parsed.flat.pos, dtype=np.float32)[
            : self.n_frames * self.n_atoms
        ].reshape(self.n_frames, self.n_atoms, 3)

        if np.allclose(parsed.cell, [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]):
            wmsg = (
                "1 A^3 CRYST1 record,"
                " this is usually a placeholder."
                " Unit cell dimensions will be set to None."
            )
            warnings.warn(wmsg)
            logger.warning(wmsg)
            self._dimensions = None
        else:
            self._dimensions = parsed.cell

        self.ts = self._Timestep(self.n_atoms, **self._ts_kwargs)
        self._read_frame(0)

    @staticmethod
    def _format_hint(thing):
        """Can this Reader read *thing*?"""
        return _is_gemmi_structure(thing)

    def _read_next_timestep(self, ts=None):
        return self._read_frame(self.ts.frame + 1)

    def _read_frame(self, frame):
        if not 0 <= frame < self.n_frames:
            raise IOError(
                f"Invalid frame {frame} for trajectory with length {self.n_frames}"
            )
        self.ts.frame = frame
        self.ts.positions = self._positions[frame]
        if self.convert_units:
            self.convert_pos_from_native(self.ts._pos)
        self.ts.dimensions = self._dimensions
        return self.ts

    def _reopen(self):
        self.ts.frame = -1
