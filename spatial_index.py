"""
Spatially order the redshift-partitioned SuperMock catalogs.

``repartition_by_redshift.py`` produces files that are contiguous in redshift
but arbitrarily ordered on the sky.  This script takes those files and rewrites
each one so that its rows are sorted by HEALPix pixel, then attaches the
"index" group described in SPEC.md.

Scheme
------
* The user picks a spatial index *level*.  Objects are binned with
  ``nside = 2 ** level`` in **nested** ordering, using ``ra``/``dec``.
* Rows are sorted by that pixel id, which makes every pixel's rows contiguous.
* Because nested ordering is hierarchical (the parent of pixel ``p`` is
  ``p >> 2``), the same row order is simultaneously contiguous for every
  coarser level, so ``level_0`` through ``level_<level>`` are all written out.

Output layout (SPEC.md)
-----------------------
    /data/<column>          one hdf5 dataset per column, all the same length
    /index                  attrs: index_type = "healpix"
        /level_0/start      length 12 * 4**0, indexed by pixel id
        /level_0/size
        ...
        /level_N/start      length 12 * 4**N
        /level_N/size

Anything that is not a per-object column (the shared time-grid axes) is copied
through unchanged at the root, and the root attributes are preserved.  The
header group that a complete OpenCosmo file also needs is deliberately out of
scope here.

Storage
-------
This stage writes the final artefact, so it is compressed -- but selectively.
Columns that compress well (the 2-D history arrays; small-range integers like
the flags, states and provenance ids) get Blosc/zstd-5.  Continuous 1-D floats
(``ra``/``dec``/``redshift``, positions, velocities, magnitudes) and
high-cardinality identifier integers compress only ~1.1-1.3x, so they are
stored contiguous and unfiltered, which also makes an index-driven slice read a
single seek.  Chunks are sized small (``--chunk-kib``, default 1 MiB
uncompressed) because the smallest expected read is a narrow sky cone;
compression ratio is nearly insensitive to chunk volume (PLAN 4.3).
Reading the Blosc columns back requires ``hdf5plugin``.

Usage
-----
    python spatial_index.py --level 5
    python spatial_index.py --level 5 --processes 8

Each input file is reordered independently, so whole files are scattered
across worker processes.  The pixel permutation is global, so each worker
reads one whole column of its file into RAM, permutes it there, and streams it
back out chunk-aligned -- sequential on both sides, no scattered disk I/O.
Peak memory is therefore roughly (largest column) * (worker processes); the
largest column is ``sfh`` (float64, 117 wide).  Size ``--processes`` to fit.
"""

import argparse
import glob
import os
import shutil
import sys
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import healpy as hp
import numpy as np

from supermock_raw import (
    add_compression_args,
    open_output,
    set_blosc_threads,
    should_compress,
    storage_opts,
)

RA_KEY = "ra"
DEC_KEY = "dec"

# Output storage settings passed as one unit through the worker job tuple.
OutputPolicy = namedtuple(
    "OutputPolicy", "compression compression_level chunk_cache_mb chunk_bytes"
)

# Shared 1-D axes: one value per history bin, not one row per object.
SHARED_DATASETS = (
    "sfh_age_gyr",
    "sfh_redshift",
    "mah_age_gyr",
    "mah_redshift",
    "mah_step",
)

# Units for the columns whose physical meaning is documented in the catalog
# notebook.  Columns not listed here are written without a unit rather than
# with a guessed one.  Strings are astropy.units-parseable, per SPEC.md: note
# that "h" would be read as *hours*, so the reduced Hubble parameter is spelled
# "littleh".  These are exactly the strings astropy produces for, e.g.,
# str(u.solMass / cu.littleh), and they parse back with astropy.cosmology.units
# enabled.
UNITS = {
    "ra": "deg",
    "dec": "deg",
    "stellar_mass": "solMass / littleh",
    "stellar_mass_zobs": "solMass / littleh",
    "peak_mass": "solMass / littleh",
    "x": "Mpc / littleh",
    "y": "Mpc / littleh",
    "z": "Mpc / littleh",
    "vx": "km / s",
    "vy": "km / s",
    "vz": "km / s",
    "sfh": "solMass / yr",
    "mah": "solMass / littleh",
    "mah_host": "solMass / littleh",
    "L_BOL_LSUN": "solLum",
    "L_8_33UM_LSUN": "solLum",
    "sfh_age_gyr": "Gyr",
    "mah_age_gyr": "Gyr",
}
MAG_UNIT = "mag"

# Plain-text "description" attributes (SPEC.md).  Discrete identifiers and flags
# still get one even though they have no unit.
DESCRIPTIONS = {
    "stellar_mass": "Stellar mass of the galaxy",
    "central": "Whether the galaxy is in a central core",
    "merged": "Whether the galaxy's core has merged",
    "core_tag": "Unique identifier of the galaxy core",
    "fof_halo_tag": "Friends-of-friends host halo identifier",
    "core_state": "Current state of the galaxy core",
    "core_state_history": "History of galaxy core states",
    "skypatch": "Source skypatch identifier",
    "source_global_row": "Producer flat row within the source skypatch",
    "source_core": "Source core identifier within the skypatch",
    "t25_a1": "Scale factor when 25 percent of the peak mass was assembled",
    "t50_a1": "Scale factor when 50 percent of the peak mass was assembled",
    "time_infall": "Time when the galaxy entered its host halo",
    "rank_peak_mass": "Rank by peak mass within the host halo",
    "vx": "x-component of the galaxy velocity",
    "vy": "y-component of the galaxy velocity",
    "vz": "z-component of the galaxy velocity",
}


def object_columns(handle, n_objects):
    """Names of the per-object datasets, i.e. those with one row per object."""
    names = []
    for key in handle:
        item = handle[key]
        if not isinstance(item, h5py.Dataset):
            continue
        if key in SHARED_DATASETS:
            continue
        if item.shape and item.shape[0] == n_objects:
            names.append(key)
    return sorted(names)


def unit_for(name):
    if name in UNITS:
        return UNITS[name]
    if name.startswith(("mag_", "M_ABS_")):
        return MAG_UNIT
    if name.endswith("_age_gyr"):
        return "Gyr"
    return None


def annotate(dataset, name):
    """Attach the "unit" and "description" attributes for a column."""
    unit = unit_for(name)
    if unit is not None:
        dataset.attrs["unit"] = unit
    description = DESCRIPTIONS.get(name)
    if description is not None:
        dataset.attrs["description"] = description


def healpix_pixels(ra, dec, level):
    """Nested HEALPix pixel of each object at ``nside = 2 ** level``."""
    nside = 2**level
    return hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)


def build_index(pixels_sorted, level):
    """(start, size) arrays for every level from 0 up to ``level``.

    ``pixels_sorted`` must already be sorted.  Arrays are indexed directly by
    pixel id and therefore have length ``12 * 4**l``, with size 0 for pixels
    that contain no objects.
    """
    levels = {}
    pixels = pixels_sorted
    for lvl in range(level, -1, -1):
        npix = 12 * 4**lvl
        edges = np.searchsorted(pixels, np.arange(npix + 1, dtype=np.int64))
        levels[lvl] = (edges[:-1], np.diff(edges))
        if lvl:
            # Nested ordering: the parent of pixel p at level l is p >> 2.
            # Right-shifting a sorted array keeps it sorted.
            pixels = pixels >> 2
    return levels


def occupancy(levels, level):
    start, size = levels[level]
    filled = size > 0
    n_filled = int(filled.sum())
    if not n_filled:
        return "  no occupied pixels"
    occupied = size[filled]
    return (
        f"  level {level}: {n_filled:,} / {len(size):,} pixels occupied, "
        f"{occupied.min():,} - {occupied.max():,} objects per pixel "
        f"(mean {occupied.mean():,.0f})"
    )


def reorder_file(in_path, out_path, level, block_rows, progress, policy):
    """Rewrite one file, sorted by HEALPix pixel.  Returns lines to log.

    Nothing here prints directly except the transient progress line: with
    several workers running, interleaved partial output would be unreadable,
    so messages are returned and printed by the parent once the file is done.
    """
    log = []
    with (
        h5py.File(in_path, "r") as src,
        open_output(out_path, chunk_cache_mb=policy.chunk_cache_mb) as dst,
    ):
        if RA_KEY not in src or DEC_KEY not in src:
            raise ValueError(f"{os.path.basename(in_path)} has no {RA_KEY}/{DEC_KEY}")

        n = src[RA_KEY].shape[0]
        columns = object_columns(src, n)
        log.append(f"  {n:,} objects, {len(columns)} columns")

        pixels = healpix_pixels(src[RA_KEY][:], src[DEC_KEY][:], level)

        # Stable sort so that, within a pixel, the original (redshift-ordered)
        # row order is preserved.
        order = np.argsort(pixels, kind="stable")
        pixels = pixels[order]

        levels = build_index(pixels, level)
        log.append(occupancy(levels, level))
        del pixels

        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["spatial_index_level"] = level
        dst.attrs["spatial_index_nside"] = 2**level

        data = dst.create_group("data")
        for i, name in enumerate(columns, 1):
            ref = src[name]
            # Whole column into RAM: one sequential read, RAM-speed permute, one
            # sequential write.  Any block-at-a-time scheme would turn the global
            # permutation into scattered I/O on one side or the other.
            values = ref[:]
            chunks, compression_args = storage_opts(
                ref.dtype,
                ref.shape,
                policy.compression,
                policy.compression_level,
                compress=should_compress(values),
                chunk_bytes=policy.chunk_bytes,
            )
            out = data.create_dataset(
                name,
                shape=ref.shape,
                dtype=ref.dtype,
                chunks=chunks,
                **compression_args,
            )
            annotate(out, name)

            step = block_rows or (chunks[0] if chunks else 16 << 20)
            for lo in range(0, n, step):
                hi = min(lo + step, n)
                out[lo:hi] = values[order[lo:hi]]
            del values

            if progress:
                print(
                    f"\r  column {i:>3}/{len(columns)} ({name[:28]:<28})",
                    end="",
                    flush=True,
                )
        if progress:
            print("\r" + " " * 60 + "\r", end="")

        # Persist the exact reorder operation alongside the final catalog.  In
        # particular, this makes the final row -> raw SED row join auditable
        # even after the redshift-partitioned intermediates have been removed.
        provenance = dst.create_group("provenance")
        input_for_output = provenance.create_dataset(
            "input_row_for_output_row", shape=(n,), dtype=np.int64
        )
        step = block_rows or (16 << 20)
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            input_for_output[lo:hi] = order[lo:hi]
        provenance.attrs["permutation_semantics"] = (
            "input_row_for_output_row[final_spatial_row] = repartitioned_input_row"
        )
        provenance.attrs["input_row_order"] = "redshift partition order"
        provenance.attrs["output_row_order"] = "nested HEALPix pixel, stable input order"

        index = dst.create_group("index")
        index.attrs["index_type"] = "healpix"
        for lvl in sorted(levels):
            start, size = levels[lvl]
            group = index.create_group(f"level_{lvl}")
            group.create_dataset("start", data=start.astype(np.int64))
            group.create_dataset("size", data=size.astype(np.int64))

        # Everything that is not a per-object column is passed through as-is.
        for key in src:
            if key in columns:
                continue
            src.copy(key, dst, name=key)
            item = dst[key]
            if isinstance(item, h5py.Dataset):
                annotate(item, key)
    return log


def process_one(job):
    """Worker entry point: reorder a single file into place.

    The output is built at a ".partial" path and only moved into place on
    success, so an interrupted or failed run never leaves a truncated file
    that a later --overwrite-less run would mistake for finished work.
    """
    in_path, out_path, level, block_rows, progress, policy = job
    tmp_path = out_path + ".partial"
    try:
        log = reorder_file(in_path, tmp_path, level, block_rows, progress, policy)
        shutil.move(tmp_path, out_path)
        return os.path.basename(in_path), log, None
    except BaseException as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return os.path.basename(in_path), [], f"{type(exc).__name__}: {exc}"


def run(in_dir, out_dir, level, block_rows, overwrite, processes, policy):
    paths = sorted(glob.glob(os.path.join(in_dir, "*.hdf5")))
    if not paths:
        sys.exit(f"No .hdf5 files found in {in_dir!r}")

    npix = 12 * 4**level
    print(
        f"Spatial index: level {level}, nside {2**level}, {npix:,} pixels "
        f"({hp.nside2resol(2**level, arcmin=True) / 60:.3f} deg resolution)"
    )
    print(f"Found {len(paths)} file(s) in {in_dir!r}\n")

    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for path in paths:
        name = os.path.basename(path)
        out_path = os.path.join(out_dir, name)
        if os.path.abspath(out_path) == os.path.abspath(path):
            sys.exit("Input and output directories must differ")
        if os.path.exists(out_path) and not overwrite:
            print(f"  {name}: exists, skipping (use --overwrite)")
            continue
        jobs.append((path, out_path, level, block_rows, False, policy))

    if not jobs:
        print("\nNothing to do.")
        return

    # Files are independent, so scatter whole files rather than splitting one
    # file's columns: no worker touches another's output and no locking or
    # parallel-HDF5 build is needed.  More processes than files would just sit
    # idle.
    workers = min(processes, len(jobs))
    done = 0
    failures = []

    if workers == 1:
        # Stay in-process when serial: keeps tracebacks intact, allows the
        # in-place per-column progress line, and avoids pickling overhead.
        jobs = [job[:4] + (sys.stdout.isatty(),) + job[5:] for job in jobs]
        results = map(process_one, jobs)
    else:
        print(f"Using {workers} worker process(es)\n")
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(process_one, job) for job in jobs]
        results = (f.result() for f in as_completed(futures))

    try:
        for name, log, error in results:
            done += 1
            print(f"[{done}/{len(jobs)}] {name}")
            for line in log:
                print(line)
            if error:
                failures.append((name, error))
                print(f"  FAILED: {error}")
    finally:
        if workers > 1:
            pool.shutdown()

    if failures:
        sys.exit(
            f"\n{len(failures)} file(s) failed:\n"
            + "\n".join(f"  {name}: {error}" for name, error in failures)
        )

    print(f"\nDone. Spatially ordered files in {out_dir!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Sort redshift-partitioned SuperMock catalogs by nested "
        "HEALPix pixel and write the OpenCosmo spatial index."
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=5,
        help="spatial index level; nside = 2**level (default: 5)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="repartitioned",
        help="directory holding the redshift-partitioned .hdf5 files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="spatially_indexed",
        help="directory for the output files",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=None,
        metavar="N",
        help="write each column in slices of N rows; omit to use the output "
        "chunk row count (default)",
    )
    parser.add_argument(
        "-j",
        "--processes",
        type=int,
        default=1,
        metavar="N",
        help="number of worker processes; files are scattered across them. Peak "
        "memory is about (largest column) * N (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite outputs that already exist",
    )
    parser.add_argument(
        "--chunk-cache-mb",
        type=int,
        default=32,
        help="HDF5 raw chunk cache per output file in MB (default: 32)",
    )
    parser.add_argument(
        "--chunk-kib",
        type=int,
        default=1024,
        metavar="N",
        help="uncompressed bytes per output chunk, in KiB; smaller reads back "
        "small sky cones with less over-read (default: 1024)",
    )
    add_compression_args(parser)
    args = parser.parse_args()

    if not 0 <= args.level <= 13:
        sys.exit("--level must be between 0 and 13")
    if args.block_rows is not None and args.block_rows < 1:
        sys.exit("--block-rows must be >= 1")
    if args.processes < 1:
        sys.exit("--processes must be >= 1")
    if args.chunk_cache_mb < 1:
        sys.exit("--chunk-cache-mb must be >= 1")
    if args.chunk_kib < 1:
        sys.exit("--chunk-kib must be >= 1")

    # Set this before hdf5plugin is imported and before workers start, so child
    # processes inherit the correct per-worker Blosc thread count.
    set_blosc_threads(args.processes)

    policy = OutputPolicy(
        args.compression,
        args.compression_level,
        args.chunk_cache_mb,
        args.chunk_kib << 10,
    )
    run(
        args.input_dir,
        args.output_dir,
        args.level,
        args.block_rows,
        args.overwrite,
        args.processes,
        policy,
    )


if __name__ == "__main__":
    main()
