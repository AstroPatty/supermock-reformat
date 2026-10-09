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

This is an intermediate stage: its outputs are consumed and rewritten by later
stages, so the datasets are written **uncompressed and contiguous**.  That
removes the filter pipeline, chunk cache and chunk B-tree from the write path,
leaving each bin fill as a single positioned write -- essentially a raw byte
copy from the source slab to its reserved span in the output file.
"""

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import h5py
import numpy as np

from supermock_raw import (
    Progress,
    RawLayoutError,
    core_groups,
    discover_skypatches,
    open_output,
    patch_offsets,
    patch_path,
    resolve_columns,
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


def _selected_patches(raw_root, patches):
    """Return requested patches after checking they exist in every raw kind."""
    available = discover_skypatches(raw_root)
    if patches is None:
        return available
    selected = sorted(set(map(int, patches)))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise RawLayoutError(f"Requested skypatches are unavailable: {missing}")
    return selected


def verify(raw_root, patches=None):
    """Check the cheap scalar invariants that establish raw-file alignment."""
    patches = _selected_patches(raw_root, patches)
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


def create_outputs(out_dir, prefix, edges, counts, columns, patches):
    """Create pre-sized output files with uncompressed, contiguous datasets.

    Contiguous layout means every dataset occupies one flat run of bytes, so a
    write to ``dst[name][lo:hi]`` is a single ``pwrite`` at a known offset with
    no chunk cache, chunk B-tree or filter pipeline involved.
    """
    os.makedirs(out_dir, exist_ok=True)
    handles = []
    n_bins = len(edges) - 1
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        path = os.path.join(out_dir, f"{prefix}_z_{lo:05.2f}_{hi:05.2f}.hdf5")
        # A contiguous dataset never touches the raw chunk cache, so keep it tiny.
        dst = open_output(path, chunk_cache_mb=1)
        total = int(counts[i])
        dst.attrs["z_min"] = lo
        dst.attrs["z_max"] = hi
        dst.attrs["partition_index"] = i
        dst.attrs["n_partitions"] = n_bins
        dst.attrs["n_objects"] = total
        dst.attrs["redshift_edges"] = edges
        dst.attrs["skypatches"] = np.asarray(patches, dtype=np.int32)
        dst.attrs["source_global_row_scope"] = "within source skypatch"
        dst.attrs["source_row_order"] = (
            "producer flat row order: lexicographically sorted core groups"
        )
        for name, source in columns.items():
            shape = (total,) + tuple(source.extra_shape)
            dst.create_dataset(name, shape=shape, dtype=source.dtype)
        for name in ("skypatch", "source_core"):
            dst.create_dataset(name, shape=(total,), dtype=np.int32)
        dst.create_dataset("source_global_row", shape=(total,), dtype=np.int64)
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


# --- Parallel decompression -------------------------------------------------
#
# gzip inflate is CPU-bound and single-threaded, and h5py serialises every HDF5
# call behind one global lock, so a *thread* pool gives no speedup (measured:
# 0%).  The reads below are therefore fanned out across worker *processes*, each
# with its own read-only file handle, decompressing disjoint row ranges straight
# into a shared-memory buffer the parent owns.  No decompressed array is ever
# pickled back -- the workers return only a row count.

_OPEN_FILES = {}


def _worker_file(path):
    """Return a per-worker cached read-only handle for ``path``.

    Workers persist across tasks, so caching turns thousands of opens of a large
    catalog into one per worker.  Read-only handles are never shared across
    processes, so this is safe.
    """
    handle = _OPEN_FILES.get(path)
    if handle is None:
        handle = h5py.File(path, "r")
        _OPEN_FILES[path] = handle
    return handle


def _gather_worker(args):
    """Decompress one source row range into its slot in the shared buffer."""
    path, h5path, src_lo, src_hi, dst_lo, shm_name, shape, dtype_str = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        buf = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        buf[dst_lo : dst_lo + (src_hi - src_lo)] = _worker_file(path)[h5path][
            src_lo:src_hi
        ]
    finally:
        shm.close()
    return src_hi - src_lo


def _chunk_aligned_spans(handle, sources, n_total, processes):
    """Split ``sources`` into ~``4*processes`` chunk-aligned work pieces.

    ``sources`` is a list of ``(h5path, n_rows, dst_offset)``.  Splitting inside
    the large core groups (the biggest is ~10% of the catalog) is what lets the
    pool balance past ~10x; aligning every boundary to the dataset's chunk rows
    keeps each gzip chunk owned by exactly one worker, so none is inflated twice.
    """
    target = max(1, math.ceil(n_total / max(1, 4 * processes)))
    items = []
    for h5path, n_rows, dst_off in sources:
        if n_rows <= 0:
            continue
        chunks = handle[h5path].chunks
        chunk_rows = chunks[0] if chunks else n_rows
        step = max(chunk_rows, (target // chunk_rows) * chunk_rows or chunk_rows)
        for lo in range(0, n_rows, step):
            hi = min(lo + step, n_rows)
            items.append((h5path, lo, hi, dst_off + lo))
    return items


def gather(pool, processes, path, sources, shape, dtype):
    """Parallel-decompress ``sources`` into one array of ``shape``.

    Returns ``(array, close)``.  ``array`` is backed by shared memory when a pool
    is supplied; ``close()`` releases it and must be called once the caller has
    finished reading (drop the array reference first).  With ``pool is None`` the
    read runs serially in-process and ``close`` is a no-op.
    """
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize

    if pool is None or nbytes == 0:
        array = np.empty(shape, dtype=dtype)
        with h5py.File(path, "r") as handle:
            for h5path, n_rows, dst_off in sources:
                if n_rows:
                    array[dst_off : dst_off + n_rows] = handle[h5path][:n_rows]
        return array, (lambda: None)

    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        with h5py.File(path, "r") as handle:
            spans = _chunk_aligned_spans(handle, sources, shape[0], processes)
        work = [
            (path, h5path, lo, hi, dst, shm.name, shape, dtype.str)
            for (h5path, lo, hi, dst) in spans
        ]
        for _ in pool.map(_gather_worker, work):
            pass
    except BaseException:
        shm.close()
        shm.unlink()
        raise

    def close():
        shm.close()
        shm.unlink()

    return array, close


def write_patch(
    patch, raw_root, columns, edges, outputs, starts, counts_by_patch, pool, processes
):
    """Pass 2: place one skypatch's objects into their global redshift bins.

    Each object is read once.  A single stable ``argsort`` of the bin index
    permutes the patch into bin order, and every output dataset region is then
    filled with one contiguous slab per bin -- one positioned write, no
    per-core-group scatter.  The per-column reads are decompressed in parallel by
    ``gather`` (see above); the permute-and-write stays in this process.
    """
    n_bins = len(edges) - 1
    catalog_path = patch_path(raw_root, patch, "lightcone_catalogs")
    groups, sizes, offsets, n_total = patch_offsets(catalog_path)

    with h5py.File(catalog_path, "r") as catalog:
        z = np.concatenate([catalog[group][REDSHIFT_KEY][:] for group in groups])

    bins = assign_bins(z, edges)
    # Stable so that, within a bin, objects keep their raw core-group order and
    # every source file (which shares that order) permutes identically.
    order = np.argsort(bins, kind="stable")
    counts_pb = np.bincount(bins, minlength=n_bins).astype(np.int64)
    if not np.array_equal(counts_pb, counts_by_patch[patch]):
        raise AssertionError(
            f"Patch {patch} bin counts changed between pass 1 and pass 2"
        )
    bin_ptr = np.empty(n_bins + 1, dtype=np.int64)
    bin_ptr[0] = 0
    np.cumsum(counts_pb, out=bin_ptr[1:])
    base = starts[patch]

    def emit(name, values):
        for b in range(n_bins):
            s, e = int(bin_ptr[b]), int(bin_ptr[b + 1])
            if e == s:
                continue
            lo = int(base[b])
            outputs[b][name][lo : lo + (e - s)] = values[order[s:e]]

    emit("skypatch", np.full(z.shape[0], patch, dtype=np.int32))
    # This is the join key for very large auxiliary products (notably hi-res
    # SEDs).  It is the row in the producer's flat, lexicographic-core stream,
    # not a core id or a row in the redshift-partitioned output.
    emit("source_global_row", np.arange(n_total, dtype=np.int64))
    emit(
        "source_core",
        np.concatenate(
            [np.full(int(n), core_id(g), np.int32) for g, n in zip(groups, sizes)]
        ),
    )

    progress = Progress(f"patch {patch} columns", len(columns))
    written_columns = 0

    for name, source in columns.items():
        if source.kind != "lightcone_catalogs":
            continue
        # One shared-memory slot per group, at that group's flat-file offset.
        sources = [
            (f"{group}/{source.dataset}", int(size), int(offset))
            for group, size, offset in zip(groups, sizes, offsets)
        ]
        shape = (n_total,) + tuple(source.extra_shape)
        column, close = gather(
            pool, processes, catalog_path, sources, shape, source.dtype
        )
        try:
            emit(name, column)
        finally:
            del column
            close()
        written_columns += 1
        progress.update(written_columns)

    # Luminosities are stored uncompressed and contiguous, so decompression is
    # not the cost there; read them serially.  Photometry is gzip-compressed and
    # large, so its single flat dataset is split by chunk-aligned row range.
    lum_by_dataset = {}
    for name, source in columns.items():
        if source.kind == "luminosities":
            lum_by_dataset.setdefault(source.dataset, []).append((name, source))
    if lum_by_dataset:
        with h5py.File(patch_path(raw_root, patch, "luminosities"), "r") as flat:
            for dataset_name, members in lum_by_dataset.items():
                full = flat[dataset_name][:]
                for name, source in members:
                    col = (
                        full if source.col_index is None else full[:, source.col_index]
                    )
                    emit(name, col)
                    written_columns += 1
                    progress.update(written_columns)
                del full

    phot_by_dataset = {}
    for name, source in columns.items():
        if source.kind == "photometry":
            phot_by_dataset.setdefault(source.dataset, []).append((name, source))
    if phot_by_dataset:
        phot_path = patch_path(raw_root, patch, "photometry")
        with h5py.File(phot_path, "r") as flat:
            widths = {ds: flat[ds].shape[1] for ds in phot_by_dataset}
            dtypes = {ds: flat[ds].dtype for ds in phot_by_dataset}
        for dataset_name, members in phot_by_dataset.items():
            shape = (n_total, widths[dataset_name])
            full, close = gather(
                pool,
                processes,
                phot_path,
                [(dataset_name, n_total, 0)],
                shape,
                dtypes[dataset_name],
            )
            try:
                for name, source in members:
                    col = (
                        full if source.col_index is None else full[:, source.col_index]
                    )
                    emit(name, col)
                    written_columns += 1
                    progress.update(written_columns)
            finally:
                del full
                close()

    progress.done(written_columns)
    return counts_pb


def repartition(raw_root, out_dir, n_partitions, prefix, decimals, processes, patches=None):
    patches = _selected_patches(raw_root, patches)
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
    print(f"\n  output columns: {len(columns)}")

    outputs = create_outputs(out_dir, prefix, edges, counts, columns, patches)
    written = np.zeros(len(counts), dtype=np.int64)
    pool = ProcessPoolExecutor(max_workers=processes) if processes > 1 else None
    print(f"  decompression workers: {processes}")
    try:
        for patch in patches:
            print(f"\nPass 2/2: patch {patch}")
            written += write_patch(
                patch,
                raw_root,
                columns,
                edges,
                outputs,
                starts,
                counts_by_patch,
                pool,
                processes,
            )
            print(f"  patch {patch}: columns written")
    finally:
        if pool is not None:
            pool.shutdown()
        for dst in outputs:
            dst.close()

    expected = np.asarray(counts, dtype=np.int64)
    if not np.array_equal(written, expected):
        sys.exit(f"Row count mismatch!\n  expected: {expected}\n  written:  {written}")
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
        "--processes",
        type=int,
        default=os.cpu_count() or 1,
        help="worker processes for parallel decompression (default: all cores; "
        "1 disables the pool and reads serially)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify raw alignment without writing outputs",
    )
    args = parser.parse_args()
    if args.n_partitions < 1:
        sys.exit("--n-partitions must be >= 1")
    if args.processes < 1:
        sys.exit("--processes must be >= 1")
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
                args.processes,
            )
    except RawLayoutError as exc:
        sys.exit(f"Raw layout error: {exc}")


if __name__ == "__main__":
    main()
