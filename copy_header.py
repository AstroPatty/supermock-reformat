#!/usr/bin/env python
"""
Copy a "header" group into existing OpenCosmo files, in place.

SPEC.md requires a file holding a single OpenCosmo dataset to also carry a
"header" group describing the cosmology, simulation parameters and per-dataset
metadata.  The catalogs produced by ``repartition_by_redshift.py`` and
``spatial_index.py`` have none, so this script grafts one on from a donor file.

The header is copied verbatim, including every nested group, dataset and
attribute.  Nothing else in the target files is touched.

Usage
-----
    python copy_header.py --source header_donor.hdf5 spatially_indexed/*.hdf5
    python copy_header.py -s donor.hdf5 -d spatially_indexed --replace

The donor header describes one specific redshift shell, so copying it verbatim
would label every partition with the donor's redshift.  Three fields are
therefore rewritten per target:

    header/file/redshift      median of the file's own redshift column
    header/file/step          monotonically increasing, in target order
    header/lightcone/z_range  the file's own (z_min, z_max) attributes

Pass --no-specialise for a strictly verbatim copy.
"""

import argparse
import glob
import os
import sys

import hdf5plugin  # Register Blosc filters before h5py reads compressed datasets.
import h5py
import numpy as np

HEADER = "header"
REDSHIFT_KEY = "redshift"


def collect_targets(paths, directory):
    """Expand the target list from explicit paths and/or a directory."""
    targets = list(paths)
    if directory:
        targets += sorted(glob.glob(os.path.join(directory, "*.hdf5")))
    # Preserve order but drop duplicates, which are easy to produce by passing
    # both a glob and the directory that contains it.
    seen = set()
    unique = []
    for path in targets:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def describe(group):
    """Short summary of what the header contains, for the log."""
    groups = datasets = 0
    attrs = len(group.attrs)

    def count(_, item):
        nonlocal groups, datasets, attrs
        if isinstance(item, h5py.Dataset):
            datasets += 1
        else:
            groups += 1
        attrs += len(item.attrs)

    group.visititems(count)
    return f"{groups} subgroup(s), {datasets} dataset(s), {attrs} attribute(s)"


def redshift_bounds(handle):
    """(z_min, z_max) recorded by repartition_by_redshift, if present."""
    if "z_min" in handle.attrs and "z_max" in handle.attrs:
        return float(handle.attrs["z_min"]), float(handle.attrs["z_max"])
    return None


def redshift_column(handle):
    """The per-object redshifts, wherever this stage of the pipeline puts them.

    ``repartition_by_redshift`` leaves the catalog flat, while
    ``spatial_index`` moves every column under "data" -- so both layouts are
    checked rather than assuming one.
    """
    for path in (REDSHIFT_KEY, f"data/{REDSHIFT_KEY}"):
        if path in handle and isinstance(handle[path], h5py.Dataset):
            return handle[path]
    return None


def specialise(header, handle, step, name):
    """Overwrite the per-file fields of a freshly copied header.

    The donor header describes one particular redshift shell, so copying it
    verbatim would label every partition with the donor's redshift.  These
    three fields are the ones that genuinely vary per file.  Returns the log
    lines describing what changed.
    """
    log = []

    column = redshift_column(handle)
    if column is None:
        log.append(
            f"  {name}: WARNING no {REDSHIFT_KEY!r} column, "
            "file/redshift left at the donor value"
        )
    elif "file" not in header:
        log.append(f"  {name}: WARNING header has no 'file' group")
    else:
        # The median is read from the actual objects rather than taken as the
        # midpoint of the bin edges: the redshift distribution within a
        # partition is not uniform, so the two are not the same number.
        median = float(np.median(column[:]))
        header["file"].attrs["redshift"] = np.float64(median)
        header["file"].attrs["step"] = np.int64(step)
        log.append(f"  {name}: file/redshift = {median:.6f}, file/step = {step}")

    bounds = redshift_bounds(handle)
    if "lightcone" not in header:
        log.append(f"  {name}: WARNING header has no 'lightcone' group")
    elif bounds is None:
        log.append(
            f"  {name}: WARNING no z_min/z_max attrs, "
            "lightcone/z_range left at the donor value"
        )
    else:
        header["lightcone"].attrs["z_range"] = np.array(bounds, dtype=np.float64)
        log.append(f"  {name}: lightcone/z_range = [{bounds[0]:g}, {bounds[1]:g}]")

    return log


def copy_header(source_path, targets, replace, specialise_fields, first_step):
    with h5py.File(source_path, "r") as src:
        if HEADER not in src:
            sys.exit(f"{os.path.basename(source_path)} has no {HEADER!r} group")
        if not isinstance(src[HEADER], h5py.Group):
            sys.exit(f"{os.path.basename(source_path)}: {HEADER!r} is not a group")

        print(f"Source: {source_path}")
        print(f"  header: {describe(src[HEADER])}")
        source_key = os.path.abspath(source_path)

        copied = skipped = 0
        for path in targets:
            name = os.path.basename(path)

            if os.path.abspath(path) == source_key:
                print(f"  {name}: is the source, skipping")
                skipped += 1
                continue

            # Opened "r+": the file is modified in place and must already exist.
            with h5py.File(path, "r+") as dst:
                if HEADER in dst:
                    if not replace:
                        print(f"  {name}: already has a header, skipping "
                              "(use --replace)")
                        skipped += 1
                        continue
                    # HDF5 does not reclaim the freed space on delete, so the
                    # old header's bytes stay in the file as dead space.  A
                    # header is tiny, but repeated --replace runs will slowly
                    # grow the file; h5repack reclaims it if that ever matters.
                    del dst[HEADER]

                src.copy(HEADER, dst, name=HEADER)
                print(f"  {name}: header written")

                if specialise_fields:
                    for line in specialise(
                        dst[HEADER], dst, first_step + copied, name
                    ):
                        print(line)

                copied += 1

    print(f"\nDone. {copied} file(s) updated, {skipped} skipped.")
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Copy a 'header' group from a donor file into existing "
        "HDF5 files, modifying them in place."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="files to add the header to",
    )
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        help="file containing the 'header' group to copy",
    )
    parser.add_argument(
        "-d",
        "--target-dir",
        help="also add every .hdf5 file in this directory to the targets",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="overwrite a header that is already present",
    )
    parser.add_argument(
        "--no-specialise",
        action="store_true",
        help="copy the header verbatim, without overwriting the per-file "
        "redshift/step/z_range fields",
    )
    parser.add_argument(
        "--first-step",
        type=int,
        default=0,
        metavar="N",
        help="step number given to the first file; later files count up "
        "from it (default: 0)",
    )
    args = parser.parse_args()

    targets = collect_targets(args.targets, args.target_dir)
    if not targets:
        sys.exit("No target files given; pass paths and/or --target-dir")

    missing = [p for p in targets if not os.path.isfile(p)]
    if missing:
        sys.exit("Target file(s) not found:\n" + "\n".join(f"  {p}" for p in missing))

    if not os.path.isfile(args.source):
        sys.exit(f"Source file not found: {args.source}")

    copy_header(
        args.source,
        targets,
        args.replace,
        not args.no_specialise,
        args.first_step,
    )


if __name__ == "__main__":
    main()
