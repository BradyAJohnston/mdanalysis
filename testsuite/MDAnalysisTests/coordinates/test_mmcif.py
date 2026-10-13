import glob
import shutil
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.MMCIF import HAS_GEMMI

from MDAnalysisTests.datafiles import MMCIF as MMCIF_FOLDER


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.parametrize(
    "mmcif_filename",
    [
        f
        for f in glob.glob(f"{MMCIF_FOLDER}/*.cif.gz")
        if "invalid" not in f
        and "warning" not in f
        and Path(f).with_suffix("").with_suffix(".pdb.gz").exists()
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_legacy_pdb_vs_mmcif(mmcif_filename):
    u_cif = mda.Universe(mmcif_filename)
    u_pdb = mda.Universe(
        Path(mmcif_filename).with_suffix("").with_suffix(".pdb.gz")
    )
    assert len(u_cif.residues) == len(u_pdb.residues)
    assert len(u_cif.atoms) == len(u_pdb.atoms)


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.parametrize(
    "mmcif_filename",
    [
        f
        for f in glob.glob(f"{MMCIF_FOLDER}/*.cif*")
        if "invalid" not in f and "warning" not in f
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_works_with_explicit_format(mmcif_filename):
    u = mda.Universe(mmcif_filename, format="MMCIF")
    assert u.trajectory.n_atoms > 0


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.parametrize(
    "mmcif_filename",
    [
        f
        for f in glob.glob(f"{MMCIF_FOLDER}/*.cif*")
        if "invalid" not in f and "warning" not in f
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_works_without_explicit_format(mmcif_filename):
    u = mda.Universe(mmcif_filename)
    assert u.trajectory.n_atoms > 0


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.parametrize(
    "mmcif_filename,natoms_protein,natoms_total",
    [
        (f"{MMCIF_FOLDER}/1YJP.cif", 59, 66),
        (f"{MMCIF_FOLDER}/1YJP.cif.gz", 59, 66),
        (f"{MMCIF_FOLDER}/7ETN.cif.gz", 150, 150),
    ],
)
def test_n_atoms(mmcif_filename, natoms_protein, natoms_total):
    u = mda.Universe(mmcif_filename)
    assert len(u.atoms) == natoms_total
    assert len(u.select_atoms("protein").atoms) == natoms_protein


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.parametrize(
    "mmcif_filename,cell",
    [
        (
            f"{MMCIF_FOLDER}/1YJP.cif.gz",
            np.array([21.937, 4.866, 23.477, 90.00, 107.08, 90.00]),
        ),
        (
            f"{MMCIF_FOLDER}/7ETN.cif.gz",
            np.array([5.264, 24.967, 20.736, 90.00, 94.85, 90.00]),
        ),
    ],
)
def test_cell(mmcif_filename, cell):
    assert np.allclose(mda.Universe(mmcif_filename).coord._unitcell, cell)


def test_no_gemmi_raises(monkeypatch):
    monkeypatch.setattr(mda.coordinates.MMCIF, "HAS_GEMMI", False)
    with pytest.raises(ImportError, match="please install gemmi"):
        mda.coordinates.MMCIF.MMCIFReader(f"{MMCIF_FOLDER}/1YJP.cif")


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
def test_multimodel_warning_msg():
    with pytest.warns(
        UserWarning,
        match=r"File .+ has .+ models, but only the first one will be read",
    ):
        mda.coordinates.MMCIF.MMCIFReader(
            f"{MMCIF_FOLDER}/multimodel_warning.cif.gz"
        )


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
def test_cryst1_record_placeholder():
    with pytest.warns(
        UserWarning,
        match=(
            r"CRYST1 record, this is usually a placeholder. "
            r"Unit cell dimensions will be set to"
        ),
    ):
        assert (
            mda.coordinates.MMCIF.MMCIFReader(
                f"{MMCIF_FOLDER}/custom.cif.gz"
            ).ts.dimensions
            is None
        )


@pytest.fixture
def two_model_structure():
    gemmi = pytest.importorskip("gemmi")
    structure = gemmi.read_structure(f"{MMCIF_FOLDER}/1YJP.cif")
    # the copied model keeps the original's number, which must not matter
    structure.add_model(structure[0])
    for chain in structure[1]:
        for residue in chain:
            for atom in residue:
                atom.pos = gemmi.Position(
                    atom.pos.x + 1, atom.pos.y, atom.pos.z
                )
    return structure


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_models_as_frames(two_model_structure):
    u = mda.Universe(two_model_structure)
    assert u.trajectory.n_frames == 2
    first = u.trajectory[0].positions.copy()
    second = u.trajectory[1].positions
    assert np.allclose(second - first, [1, 0, 0], atol=1e-4)
    # the caller's structure is left as it was
    assert len(two_model_structure) == 2


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_universe_from_gemmi_structure():
    import gemmi

    path = f"{MMCIF_FOLDER}/1YJP.cif"
    u_file = mda.Universe(path)
    u_obj = mda.Universe(gemmi.read_structure(path))
    assert np.array_equal(u_obj.atoms.names, u_file.atoms.names)
    assert np.array_equal(u_obj.atoms.positions, u_file.atoms.positions)


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parsed_once_per_universe(monkeypatch):
    MMCIF = mda.coordinates.MMCIF
    monkeypatch.setattr(MMCIF, "_last_parse", (None, None))
    calls = []
    read = MMCIF._read_gemmi_structure

    def counting_read(filename):
        calls.append(filename)
        return read(filename)

    monkeypatch.setattr(MMCIF, "_read_gemmi_structure", counting_read)
    mda.Universe(f"{MMCIF_FOLDER}/1YJP.cif")
    assert len(calls) == 1


@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parse_cache_sees_changed_file(tmp_path):
    path = tmp_path / "structure.cif.gz"
    shutil.copy(f"{MMCIF_FOLDER}/1YJP.cif.gz", path)
    assert mda.Universe(path).atoms.n_atoms == 66
    shutil.copy(f"{MMCIF_FOLDER}/7ETN.cif.gz", path)
    assert mda.Universe(path).atoms.n_atoms == 150
