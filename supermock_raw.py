"""
Describe the producer's raw SuperMock layout in one place.

The three raw files for a skypatch are horizontally aligned, but only the
catalog is grouped by core.  Keeping discovery, ordering, and column expansion
here prevents pipeline stages from independently making plausible but
incompatible assumptions about that alignment.
"""

import json
import os
import re
from collections import namedtuple
from glob import glob

import h5py
import numpy as np

KINDS = {
    "lightcone_catalogs": "lightcone_galaxies",
    "luminosities": "luminosities",
    "photometry": "photometry",
}

DEFAULT_TIME_GRIDS = os.path.join(
    "raw", "SuperMockLoad", "supermockload", "data", "time_grids.npz"
)
_BAND_NAMES_PATH = os.path.join(
    "SuperMockLoad", "supermockload", "data", "band_names.json"
)
_TIME_GRIDS_PATH = os.path.join(
    "SuperMockLoad", "supermockload", "data", "time_grids.npz"
)

ColumnSource = namedtuple("ColumnSource", "kind dataset col_index dtype extra_shape")


class RawLayoutError(RuntimeError):
    """The raw files do not have the layout required for safe row alignment."""


def _module_raw_path(relative):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", relative)


def _data_path(root, relative, path):
    if path is not None:
        return path
    if root is not None:
        candidate = os.path.join(os.fspath(root), relative)
        if os.path.exists(candidate):
            return candidate
    candidate = _module_raw_path(relative)
    if os.path.exists(candidate):
        return candidate
    # This also makes DEFAULT_TIME_GRIDS useful when the module is run from a
    # checkout whose raw directory is the current working directory's child.
    return os.path.join("raw", relative)


def load_band_names(root=None, path=None):
    """Load producer band names, retaining their matrix-column order."""
    data_path = _data_path(root, _BAND_NAMES_PATH, path)
    try:
        with open(data_path) as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise RawLayoutError(f"Band-name file is missing: {data_path}") from exc


# Kept public for callers that need the producer's original column order.
BAND_NAMES = load_band_names()


def patch_path(root, patch, kind):
    """Return the expected raw file path for one skypatch and source kind."""
    try:
        stem = KINDS[kind]
    except KeyError as exc:
        raise RawLayoutError(f"Unknown raw kind {kind!r}") from exc
    return os.path.join(os.fspath(root), kind, f"{stem}_skypatch_{int(patch)}.h5")


def discover_skypatches(root):
    """Return skypatches present in all three required raw source directories."""
    found = {}
    for kind, stem in KINDS.items():
        pattern = os.path.join(os.fspath(root), kind, f"{stem}_skypatch_*.h5")
        patches = set()
        for path in glob(pattern):
            name = os.path.basename(path)
            if ".shard" in name:
                continue
            match = re.fullmatch(rf"{re.escape(stem)}_skypatch_(\d+)\.h5", name)
            if match:
                patches.add(int(match.group(1)))
        found[kind] = patches

    union = set().union(*found.values())
    if not union:
        raise RawLayoutError(f"No raw skypatches found under {root!r}")
    missing = [
        (patch, kind)
        for patch in sorted(union)
        for kind in KINDS
        if patch not in found[kind]
    ]
    if missing:
        pairs = ", ".join(f"({patch}, {kind})" for patch, kind in missing)
        raise RawLayoutError(f"Raw skypatch files are incomplete; missing: {pairs}")
    return sorted(union)


def _with_catalog(path_or_handle, callback):
    if isinstance(path_or_handle, (str, bytes, os.PathLike)):
        with h5py.File(path_or_handle, "r") as handle:
            return callback(handle)
    return callback(path_or_handle)


def core_groups(path_or_handle):
    """Return catalog core groups in the producer's flat-file row order."""

    def get_groups(handle):
        # PLAN §1.3 verified sorted(f.keys()) against luminosity redshifts for
        # all 62 groups. Numeric sorting fails at core_8 (3.33 vs 5.46), silently
        # attaching every following row's photometry to the wrong galaxy.
        return sorted(handle.keys())

    return _with_catalog(path_or_handle, get_groups)


def patch_offsets(path_or_handle):
    """Return groups, row counts, exclusive offsets, and their total row count.

    ``offsets[i]`` is the flat-file starting row for ``groups[i]``; it has the
    same length as ``groups`` rather than including a final sentinel.
    """

    def get_offsets(handle):
        groups = core_groups(handle)
        sizes = np.array(
            [handle[group]["redshift"].shape[0] for group in groups], dtype=np.int64
        )
        offsets = np.empty(len(sizes), dtype=np.int64)
        if len(sizes):
            offsets[0] = 0
            offsets[1:] = np.cumsum(sizes[:-1], dtype=np.int64)
        return groups, sizes, offsets, int(sizes.sum())

    return _with_catalog(path_or_handle, get_offsets)


def _flat_row_count(path, kind):
    with h5py.File(path, "r") as handle:
        names = list(handle.keys())
        if not names:
            raise RawLayoutError(f"{kind} file {path} contains no datasets")
        counts = {name: handle[name].shape[0] for name in names}
    if len(set(counts.values())) != 1:
        raise RawLayoutError(
            f"{kind} file {path} has inconsistent row counts: {counts}"
        )
    return next(iter(counts.values()))


def validate_patch(root, patch):
    """Check core schemas and the catalog-to-flat-file row-count invariant."""
    catalog_path = patch_path(root, patch, "lightcone_catalogs")
    groups, _, _, n_total = patch_offsets(catalog_path)
    if not groups:
        raise RawLayoutError(f"Catalog {catalog_path} contains no core groups")
    with h5py.File(catalog_path, "r") as handle:
        expected = set(handle[groups[0]].keys())
        for group in groups[1:]:
            actual = set(handle[group].keys())
            if actual != expected:
                difference = sorted(actual.symmetric_difference(expected))
                raise RawLayoutError(
                    f"Catalog group {group} differs from {groups[0]}; "
                    f"symmetric difference: {difference}"
                )
    for kind in ("luminosities", "photometry"):
        count = _flat_row_count(patch_path(root, patch, kind), kind)
        if count != n_total:
            raise RawLayoutError(
                f"{kind} patch {patch} has {count:,} rows; catalog has {n_total:,}"
            )


def sanitize_band_name(name):
    """Make a producer band label safe as the suffix of an output dataset name."""
    # Dots are path separators in common h5py idioms, and hyphens are awkward
    # for consumers, so remove only the known trailing text extension then
    # replace those two separators without otherwise changing producer labels.
    if name.endswith(".txt"):
        name = name[:-4]
    return name.replace(".", "_").replace("-", "_")


def _add_column(columns, name, source):
    if name in columns:
        previous = columns[name]
        raise RawLayoutError(
            f"Output column {name!r} collides between "
            f"{previous.kind}/{previous.dataset} and {source.kind}/{source.dataset}"
        )
    columns[name] = source


def _luminosity_bands(dataset, fallback):
    value = dataset.attrs.get("bands")
    if value is None:
        return fallback
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).split()[0].split(",")


def resolve_columns(root, patch, band_names_path=None):
    """Inspect raw files and return output columns in catalog/source-file order."""
    validate_patch(root, patch)
    names = load_band_names(root, band_names_path)
    columns = {}

    with h5py.File(patch_path(root, patch, "lightcone_catalogs"), "r") as catalog:
        first_group = catalog[core_groups(catalog)[0]]
        for dataset_name in first_group:
            dataset = first_group[dataset_name]
            _add_column(
                columns,
                dataset_name,
                ColumnSource(
                    "lightcone_catalogs",
                    dataset_name,
                    None,
                    dataset.dtype,
                    dataset.shape[1:],
                ),
            )

    with h5py.File(patch_path(root, patch, "luminosities"), "r") as luminosities:
        for dataset_name in luminosities:
            dataset = luminosities[dataset_name]
            if dataset_name == "redshift":
                continue
            if dataset_name in ("M_ABS_SDSS", "M_ABS_WISE"):
                bands = _luminosity_bands(dataset, names[f"_LUM_{dataset_name}"])
                if len(bands) != dataset.shape[1]:
                    raise RawLayoutError(
                        f"{dataset_name} has {dataset.shape[1]} columns but {len(bands)} band names"
                    )
                for index, band in enumerate(bands):
                    _add_column(
                        columns,
                        f"{dataset_name}_{band}",
                        ColumnSource(
                            "luminosities", dataset_name, index, dataset.dtype, ()
                        ),
                    )
            else:
                _add_column(
                    columns,
                    dataset_name,
                    ColumnSource(
                        "luminosities",
                        dataset_name,
                        None,
                        dataset.dtype,
                        dataset.shape[1:],
                    ),
                )

    with h5py.File(patch_path(root, patch, "photometry"), "r") as photometry:
        for survey in photometry:
            dataset = photometry[survey]
            survey_names = names[survey]
            if len(survey_names) != dataset.shape[1]:
                raise RawLayoutError(
                    f"{survey} has {dataset.shape[1]} columns but {len(survey_names)} band names"
                )
            output_names = [f"mag_{sanitize_band_name(band)}" for band in survey_names]
            if len(set(output_names)) != len(output_names):
                raise RawLayoutError(f"Sanitized band names collide within {survey}")
            for index, output_name in enumerate(output_names):
                _add_column(
                    columns,
                    output_name,
                    ColumnSource("photometry", survey, index, dataset.dtype, ()),
                )
    return columns


# PLAN §4 measured real columns and found the static policy below within 0.8%
# of a per-column oracle.  For 2-D data zstd5 without shuffle preserves the
# populated low mantissa bits that shuffle harms (sfh: 4.02x vs 1.34x); 1-D
# data benefits from byte shuffle.  Only float64 2-D arrays need width 3
# (sfh full-width falls from 4.06x to 2.02x); other 2-D arrays want full width.
# Roughly 8 MB uncompressed chunks were sufficient: width, not volume, mattered.


def _ensure_hdf5plugin():
    # set_blosc_threads() must be called before this lazy import, otherwise the
    # plugin has already observed BLOSC_NTHREADS and worker processes can
    # oversubscribe their CPUs.
    import hdf5plugin

    return hdf5plugin


def set_blosc_threads(processes):
    """Set Blosc's thread count before any code imports hdf5plugin."""
    processes = int(processes)
    if processes < 1:
        raise ValueError("processes must be >= 1")
    os.environ["BLOSC_NTHREADS"] = str(max(1, (os.cpu_count() or 1) // processes))


def open_output(path, mode="w", chunk_cache_mb=256):
    """Open an output HDF5 file with a cache sized for partial chunk writes."""
    chunk_cache_mb = int(chunk_cache_mb)
    if chunk_cache_mb < 1:
        raise ValueError("chunk_cache_mb must be >= 1")
    # Stage 1 scatters roughly 1,068-row slabs across output bins.  On core_19
    # sfh data, the default 1 MB cache wrote 5,887 rows/s; a 256 MB cache wrote
    # 2.68M rows/s at the identical 4.14 compression ratio.  The large cache
    # keeps the 8 MB chunks resident instead of repeatedly recompressing them.
    # 100003 is prime and comfortably exceeds the 32 8 MB chunks resident at
    # 256 MB; retain it for stage 2's smaller, more numerous chunks.
    return h5py.File(
        path,
        mode,
        rdcc_nbytes=chunk_cache_mb << 20,
        rdcc_nslots=100003,
    )


def storage_opts(dtype, shape, compression="none", level=5):
    """Return chunk geometry and create_dataset keyword arguments for a dataset."""
    shape = tuple(shape)
    if not shape or shape[0] == 0:
        return None, {}
    dtype = np.dtype(dtype)
    ndim = len(shape)
    if ndim > 1:
        width = min(shape[1], 3) if dtype == np.dtype("float64") else shape[1]
        tail = (width,) + shape[2:]
    else:
        tail = ()
    bytes_per_row = max(1, dtype.itemsize * int(np.prod(tail, dtype=np.int64)))
    rows = max(1, min(shape[0], (8 << 20) // bytes_per_row))
    chunks = (rows,) + tail

    if compression == "none":
        return chunks, {}
    if compression == "gzip":
        return chunks, {"compression": "gzip", "compression_opts": level}
    if compression == "lzf":
        return chunks, {"compression": "lzf"}
    if compression == "blosc":
        plugin = _ensure_hdf5plugin()
        shuffle = plugin.Blosc.NOSHUFFLE if ndim > 1 else plugin.Blosc.SHUFFLE
        return chunks, dict(plugin.Blosc(cname="zstd", clevel=level, shuffle=shuffle))
    raise ValueError("compression must be one of: blosc, gzip, lzf, none")


def add_compression_args(parser):
    """Add the compression options shared by all pipeline-stage CLIs."""
    parser.add_argument(
        "--compression",
        choices=("blosc", "gzip", "lzf", "none"),
        default="blosc",
        help="output compression codec (default: blosc)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="compression level where supported (default: 5)",
    )


def load_time_grids(path=None):
    """Load and validate the shared history axes supplied with the raw reader."""
    data_path = _data_path(None, _TIME_GRIDS_PATH, path)
    if not os.path.exists(data_path):
        raise RawLayoutError(f"Time-grid file is missing: {data_path}")
    expected = {
        "sfh_age_gyr": 117,
        "sfh_redshift": 117,
        "mah_age_gyr": 101,
        "mah_redshift": 101,
        "mah_step": 101,
    }
    with np.load(data_path) as grids:
        missing = sorted(set(expected) - set(grids.files))
        if missing:
            raise RawLayoutError(
                f"Time-grid file {data_path} is missing keys: {missing}"
            )
        result = {name: grids[name] for name in expected}
    for name, length in expected.items():
        if result[name].shape != (length,):
            raise RawLayoutError(
                f"Time grid {name} has shape {result[name].shape}; expected ({length},)"
            )
    return result
