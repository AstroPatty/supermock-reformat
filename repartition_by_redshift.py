#!/usr/bin/env python
"""
Re-partition the SuperMock catalogs by redshift.

The raw catalogs are split into simulation *core* groups, while their
luminosity and photometry companions are flat files whose rows follow the
catalog groups in lexicographic order.  This script reshuffles every object
from every available skypatch into global, contiguous and disjoint redshift
bins, retaining provenance so the horizontal join remains traceable.

No objects are filtered out: this is a lossless reorganisation.  The invalid
magnitude cleaning in ``load_supermock.load_and_clean_single_catalog`` is
deliberately NOT applied here, since it is redshift dependent and is better
left to downstream analysis.
"""

import argparse
import os
import sys

import h5py
import numpy as np

from supermock_raw import (
    DEFAULT_TIME_GRIDS,
    RawLayoutError,
    add_compression_args,
    core_groups,
    discover_skypatches,
    load_time_grids,
    open_output,
    patch_offsets,
    patch_path,
    resolve_columns,
    set_blosc_threads,
    storage_opts,
    validate_patch,
)

REDSHIFT_KEY = "redshift"


def compute_edges(redshifts, n_partitions, decimals=2):
    """Equal-count quantile edges, rounded to `decimals` decimal places.

    Rounding is what keeps the bounds human-readable, but it can collapse two
    edges onto the same value where the distribution is very dense.  Any such
    collapse is repaired by nudging the edge up one ulp of the rounding grid,
    which guarantees strictly increasing, non-empty bins.
    """
    step = 10.0 ** (-decimals)
    quantiles = np.quantile(redshifts, np.linspace(0.0, 1.0, n_partitions + 1))
    edges = np.round(quantiles, decimals)

    # Widen the outer edges so the extreme objects cannot fall outside.
    edges[0] = np.floor(redshifts.min() / step) * step
    edges[-1] = np.ceil(redshifts.max() / step) * step

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + step
    edges = np.round(edges, decimals)

    if edges[-1] < redshifts.max():
        edges[-1] = np.ceil(redshifts.max() / step) * step

    return edges


def report_balance(redshifts, edges):
    counts = np.histogram(redshifts, bins=edges)[0]
    # np.histogram's last bin is closed, matching the assignment in pass 2.
    ideal = len(redshifts) / len(counts)
    print(f"\n  {'bin':>4}  {'z range':>18}  {'objects':>10}  {'vs ideal':>9}")
    for i, count in enumerate(counts):
        print(
            f"  {i:>4}  [{edges[i]:>7.2f},{edges[i + 1]:>7.2f})  "
            f"{count:>10,}  {count / ideal - 1.0:>+8.1%}"
        )
    print(f"\n  ideal per partition: {ideal:,.0f}")
    print(
        f"  spread: {counts.min():,} - {counts.max():,} "
        f"({(counts.max() - counts.min()) / ideal:.1%} of ideal)"
    )
    return counts


def assign_bins(z, edges):
    """Bin index for each redshift; the final bin includes its upper edge."""
    idx = np.digitize(z, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def core_id(group):
    """Return the integer producer core id from a ``core_<n>`` group name."""
    try:
        return int(group.removeprefix("core_"))
    except ValueError as exc:
        raise RawLayoutError(f"Unexpected catalog core group name: {group!r}") from exc


def verify(raw_root):
    """Check the cheap scalar invariants that establish raw-file alignment."""
    patches = discover_skypatches(raw_root)
    print(f"Skypatch parity: PASS ({', '.join(map(str, patches))})")
    passed = failed = 0
    for patch in patches:
        validate_patch(raw_root, patch)
        print(f"Patch {patch}: row counts: PASS")
        catalog_path = patch_path(raw_root, patch, "lightcone_catalogs")
        luminosity_path = patch_path(raw_root, patch, "luminosities")
        groups, sizes, offsets, _ = patch_offsets(catalog_path)
        with (
            h5py.File(catalog_path, "r") as catalog,
            h5py.File(luminosity_path, "r") as luminosities,
        ):
            for group, size, offset in zip(groups, sizes, offsets):
                catalog_z = catalog[group][REDSHIFT_KEY]
                flat_z = luminosities[REDSHIFT_KEY]
                # Catalog redshifts are f8 and luminosity redshifts are f4, so
                # equality means their f4 representations agree, not bit equality.
                catalog_bounds = np.asarray(
                    [catalog_z[0], catalog_z[int(size) - 1]], dtype=flat_z.dtype
                )
                flat_bounds = np.asarray(
                    [flat_z[int(offset)], flat_z[int(offset + size - 1)]]
                )
                ok = np.array_equal(catalog_bounds, flat_bounds)
                print(f"Patch {patch} {group}: {'PASS' if ok else 'FAIL'}")
                passed += int(ok)
                failed += int(not ok)
    print(f"Core redshift checks: {passed} passed, {failed} failed")
    if failed:
        raise RawLayoutError("Raw core-to-flat redshift alignment verification failed")


def gather_redshifts(raw_root, patches):
    """Pass 1: read catalog redshifts in the verified flat-file group order."""
    chunks = []
    patch_counts = {}
    for patch in patches:
        path = patch_path(raw_root, patch, "lightcone_catalogs")
        patch_chunks = []
        with h5py.File(path, "r") as catalog:
            for group in core_groups(catalog):
                z = catalog[group][REDSHIFT_KEY][:]
                patch_chunks.append(z)
                print(
                    f"  patch {patch:>3} {group:>8}: {z.size:>9,} objects  "
                    f"z = [{z.min():.4f}, {z.max():.4f}]"
                )
        values = np.concatenate(patch_chunks)
        chunks.append(values)
        patch_counts[patch] = values
    return np.concatenate(chunks), patch_counts


def create_outputs(
    out_dir,
    prefix,
    edges,
    counts,
    columns,
    grids,
    patches,
    compression,
    level,
    chunk_cache_mb=256,
):
    """Create pre-sized output files and every dataset with the shared policy."""
    os.makedirs(out_dir, exist_ok=True)
    handles = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        path = os.path.join(out_dir, f"{prefix}_z_{lo:05.2f}_{hi:05.2f}.hdf5")
        dst = open_output(path, chunk_cache_mb=chunk_cache_mb)
        total = int(counts[i])
        dst.attrs["z_min"] = lo
        dst.attrs["z_max"] = hi
        dst.attrs["partition_index"] = i
        dst.attrs["n_partitions"] = len(edges) - 1
        dst.attrs["n_objects"] = total
        dst.attrs["redshift_edges"] = edges
        dst.attrs["skypatches"] = np.asarray(patches, dtype=np.int32)
        for name, source in columns.items():
            shape = (total,) + tuple(source.extra_shape)
            chunks, kwargs = storage_opts(source.dtype, shape, compression, level)
            dst.create_dataset(
                name, shape=shape, dtype=source.dtype, chunks=chunks, **kwargs
            )
        for name in ("skypatch", "source_core"):
            chunks, kwargs = storage_opts(np.int32, (total,), compression, level)
            dst.create_dataset(
                name, shape=(total,), dtype=np.int32, chunks=chunks, **kwargs
            )
        for name, values in grids.items():
            chunks, kwargs = storage_opts(
                values.dtype, values.shape, compression, level
            )
            dst.create_dataset(name, data=values, chunks=chunks, **kwargs)
        handles.append(dst)
    return handles


def patch_starts(patch_redshifts, edges):
    """Reserve each patch's contiguous span within every global output bin."""
    starts = {}
    cursor = np.zeros(len(edges) - 1, dtype=np.int64)
    counts_by_patch = {}
    for patch, z in patch_redshifts.items():
        counts = np.bincount(assign_bins(z, edges), minlength=len(cursor))
        starts[patch] = cursor.copy()
        counts_by_patch[patch] = counts
        cursor += counts
    return starts, counts_by_patch


def scatter(outputs, bins, cursors, names):
    """Scatter a slab, advancing each bin cursor exactly once for this source."""
    for b in np.unique(bins):
        b = int(b)
        mask = bins == b
        lo = int(cursors[b])
        hi = lo + int(mask.sum())
        for name, values in names:
            outputs[b][name][lo:hi] = values[mask]
        cursors[b] = hi


def repartition(
    raw_root,
    out_dir,
    n_partitions,
    prefix,
    decimals,
    time_grids,
    compression,
    level,
    chunk_cache_mb=256,
):
    patches = discover_skypatches(raw_root)
    for patch in patches:
        validate_patch(raw_root, patch)

    print("\nPass 1/2: reading redshifts")
    redshifts, patch_redshifts = gather_redshifts(raw_root, patches)
    print(f"\n  total objects: {len(redshifts):,}")
    print(f"  redshift range: {redshifts.min():.4f} - {redshifts.max():.4f}")
    edges = compute_edges(redshifts, n_partitions, decimals)
    counts = report_balance(redshifts, edges)
    starts, counts_by_patch = patch_starts(patch_redshifts, edges)
    del redshifts, patch_redshifts

    columns = resolve_columns(raw_root, patches[0])
    for patch in patches[1:]:
        if resolve_columns(raw_root, patch) != columns:
            raise RawLayoutError(
                f"Patch {patch} has a different resolved column layout"
            )
    grids = load_time_grids(time_grids)
    print(f"\n  output columns: {len(columns)}")
    set_blosc_threads(1)
    outputs = create_outputs(
        out_dir,
        prefix,
        edges,
        counts,
        columns,
        grids,
        patches,
        compression,
        level,
        chunk_cache_mb,
    )
    final_cursors = np.zeros(len(counts), dtype=np.int64)
    try:
        for patch in patches:
            print(f"\nPass 2/2: patch {patch}")
            catalog_path = patch_path(raw_root, patch, "lightcone_catalogs")
            groups, sizes, offsets, _ = patch_offsets(catalog_path)

            # The catalog pass also establishes provenance.  Every dataset is
            # slabbed, including wide histories in large core groups.
            for name, source in columns.items():
                if source.kind != "lightcone_catalogs":
                    continue
                cursors = starts[patch].copy()
                with h5py.File(catalog_path, "r") as catalog:
                    for group in groups:
                        dataset = catalog[group][source.dataset]
                        bins = assign_bins(catalog[group][REDSHIFT_KEY][:], edges)
                        values = dataset[:]
                        scatter(outputs, bins, cursors, [(name, values)])
                if not np.array_equal(cursors, starts[patch] + counts_by_patch[patch]):
                    raise AssertionError(
                        f"Catalog cursor mismatch for {name}, patch {patch}"
                    )

            cursors = starts[patch].copy()
            with h5py.File(catalog_path, "r") as catalog:
                for group in groups:
                    dataset = catalog[group][REDSHIFT_KEY]
                    bins = assign_bins(dataset[:], edges)
                    values = np.full(len(dataset), patch, dtype=np.int32)
                    cores = np.full(len(dataset), core_id(group), dtype=np.int32)
                    scatter(
                        outputs,
                        bins,
                        cursors,
                        [("skypatch", values), ("source_core", cores)],
                    )
            if not np.array_equal(cursors, starts[patch] + counts_by_patch[patch]):
                raise AssertionError(f"Provenance cursor mismatch for patch {patch}")
            final_cursors += cursors - starts[patch]

            for kind in ("luminosities", "photometry"):
                path = patch_path(raw_root, patch, kind)
                sources = {}
                for name, source in columns.items():
                    if source.kind == kind:
                        sources.setdefault(source.dataset, []).append((name, source))
                with (
                    h5py.File(path, "r") as flat,
                    h5py.File(catalog_path, "r") as catalog,
                ):
                    for dataset_name, names in sources.items():
                        dataset = flat[dataset_name]
                        cursors = starts[patch].copy()
                        for group, size, offset in zip(groups, sizes, offsets):
                            z = catalog[group][REDSHIFT_KEY]
                            bins = assign_bins(z[:], edges)
                            slab = dataset[int(offset) : int(offset) + len(z)]
                            values = [
                                (
                                    name,
                                    slab
                                    if source.col_index is None
                                    else slab[:, source.col_index],
                                )
                                for name, source in names
                            ]
                            scatter(outputs, bins, cursors, values)
                        if not np.array_equal(
                            cursors, starts[patch] + counts_by_patch[patch]
                        ):
                            raise AssertionError(
                                f"Flat cursor mismatch for {dataset_name}, patch {patch}"
                            )
            print(f"  patch {patch}: columns written")
    finally:
        for dst in outputs:
            dst.close()

    expected = np.asarray(counts, dtype=np.int64)
    # Provenance is written once per object and follows the same reservations
    # as every data column, so these are the actual final output cursors.
    cursors = final_cursors
    if not np.array_equal(cursors, expected):
        sys.exit(f"Row count mismatch!\n  expected: {expected}\n  written:  {cursors}")
    print(
        f"\nDone. {int(expected.sum()):,} objects across {len(outputs)} partition(s) in {out_dir!r}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Re-partition raw SuperMock catalogs into equal-population redshift bins."
    )
    parser.add_argument(
        "--raw-root",
        default="raw",
        help="root directory containing raw/ catalog kinds (default: raw)",
    )
    parser.add_argument(
        "-n",
        "--n-partitions",
        type=int,
        default=16,
        help="number of redshift partitions (default: 16)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="repartitioned", help="directory for output files"
    )
    parser.add_argument(
        "-p", "--prefix", default="SuperMock_v3", help="output filename prefix"
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="decimal places for redshift bounds (default: 2)",
    )
    parser.add_argument(
        "--chunk-cache-mb",
        type=int,
        default=256,
        help=(
            "HDF5 raw chunk cache per output file in MB (default: 256); total cache is "
            "chunk_cache_mb * n_partitions (defaults: 16 * 256 MB = 4 GB), the "
            "dominant memory cost of this stage"
        ),
    )
    parser.add_argument(
        "--time-grids", default=DEFAULT_TIME_GRIDS, help="path to time_grids.npz"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify raw alignment without writing outputs",
    )
    add_compression_args(parser)
    args = parser.parse_args()
    if args.n_partitions < 1:
        sys.exit("--n-partitions must be >= 1")
    if args.chunk_cache_mb < 1:
        sys.exit("--chunk-cache-mb must be >= 1")
    try:
        if args.verify:
            verify(args.raw_root)
        else:
            repartition(
                args.raw_root,
                args.output_dir,
                args.n_partitions,
                args.prefix,
                args.decimals,
                args.time_grids,
                args.compression,
                args.compression_level,
                args.chunk_cache_mb,
            )
    except RawLayoutError as exc:
        sys.exit(f"Raw layout error: {exc}")


if __name__ == "__main__":
    main()
