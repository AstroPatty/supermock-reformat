"""Append sharded hi-res SEDs to completed spatial OpenCosmo catalogs.

The normal catalog pipeline first freezes each final row's source identity in
``/data/skypatch`` and ``/data/source_global_row``.  This program then joins
the corresponding SED rows in two phases:

1. *Gather*: worker processes each read one slice of one raw shard in large
   contiguous slabs (the rows of a redshift slice are dense in source order)
   and scatter the needed rows to their final positions in an uncompressed
   row buffer -- a file in ``/dev/shm`` when it fits, else on local scratch.
2. *Write*: the buffer is streamed into ``/data/SED`` in chunk-aligned blocks,
   so every compressed chunk is written exactly once, with Blosc using all
   cores.

Reading raw rows in final (spatial) order instead turns the join into
millions of scattered ~45-KB reads, which is ~50x slower on GPFS.
"""

import argparse
import glob
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

from supermock_raw import Progress, RawLayoutError, _ensure_hdf5plugin, discover_sed_shards

SED_CODECS = ("blosc-lz4", "blosc-zstd", "gzip", "lzf", "none")
DEFAULT_CHUNK_ROWS = 64
DEFAULT_BLOCK_MIB = 512
# Read a whole raw span when at least 1/32 of its rows are needed; otherwise
# point reads are cheaper.  One GPFS point read costs about as much as
# streaming ~70 contiguous rows.
_DENSE_SPAN_FACTOR = 32
_BUFFER_HEADROOM = 1.05


def _output_paths(paths, directory):
    result = list(paths)
    if directory:
        result.extend(sorted(glob.glob(os.path.join(directory, "*.hdf5"))))
    result = list(dict.fromkeys(result))
    if not result:
        raise RawLayoutError("Specify one or more catalogs or --catalog-dir")
    return result


class SedSources:
    """Raw SED shard metadata per patch, plus lazily opened handles for checks."""

    def __init__(self, sed_dir, patches):
        self.shards = {patch: discover_sed_shards(sed_dir, patch)[patch] for patch in patches}
        self.handles = {}
        self.wave_rest = None
        self.width = None
        for patch, shards in self.shards.items():
            with h5py.File(shards[0].path, "r") as source:
                wave = source["wave_rest"][:]
                width = source["SED"].shape[1]
            if self.wave_rest is None:
                self.wave_rest, self.width = wave, width
            elif self.width != width or not np.array_equal(self.wave_rest, wave):
                raise RawLayoutError(f"SED wavelength grid differs for patch {patch}")

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def _handle(self, path):
        handle = self.handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self.handles[path] = handle
        return handle


def sed_storage_opts(codec, level, n_rows, width, chunk_rows):
    """Chunk geometry and filter kwargs for ``/data/SED``.

    SEDs are only ~1.3x compressible; byte shuffle is what buys that ratio, and
    lz4 decodes several times faster than zstd for a negligible size cost.
    Small row chunks keep single-galaxy reads cheap.
    """
    if n_rows == 0 or codec == "none":
        return None, {}
    chunks = (max(1, min(int(chunk_rows), n_rows)), width)
    if codec in ("blosc-lz4", "blosc-zstd"):
        plugin = _ensure_hdf5plugin()
        cname = codec.split("-", 1)[1]
        return chunks, dict(plugin.Blosc(cname=cname, clevel=level, shuffle=plugin.Blosc.SHUFFLE))
    if codec == "gzip":
        return chunks, {"compression": "gzip", "compression_opts": level, "shuffle": True}
    if codec == "lzf":
        return chunks, {"compression": "lzf", "shuffle": True}
    raise ValueError(f"SED codec must be one of: {', '.join(SED_CODECS)}")


def _buffer_dir(requested, fallback, nbytes):
    """Pick where the uncompressed gather buffer lives.

    RAM-backed ``/dev/shm`` is preferred; local scratch (SSD) is the bounded-
    memory fallback for slices larger than free shared memory.
    """
    candidates = [requested] if requested else ["/dev/shm", fallback]
    need = int(nbytes * _BUFFER_HEADROOM)
    for directory in candidates:
        if directory and os.path.isdir(directory) and shutil.disk_usage(directory).free >= need:
            return directory
    raise RawLayoutError(
        f"No SED buffer location with {need / 2**30:,.1f} GiB free among {candidates}"
    )


def _gather_tasks(sources, patches, rows, n_tasks):
    """Split the join into (shard path, sorted local rows, output rows) pieces."""
    order = np.lexsort((rows, patches))
    sorted_patches = patches[order]
    sorted_rows = rows[order]
    total = len(rows)
    piece = max(1, math.ceil(total / max(1, n_tasks)))
    tasks = []
    for patch in np.unique(sorted_patches):
        if int(patch) not in sources.shards:
            raise RawLayoutError(f"Catalog references SED patch {int(patch)}, which was not supplied")
        lo, hi = np.searchsorted(sorted_patches, [patch, patch + 1], side="left")
        prow, pout = sorted_rows[lo:hi], order[lo:hi]
        shards = sources.shards[int(patch)]
        if prow[0] < 0 or prow[-1] >= shards[0].n_global:
            raise RawLayoutError(f"Catalog references out-of-range SED rows for patch {int(patch)}")
        # A patch contributes each raw row only once to a redshift output,
        # therefore local indices are strictly increasing as HDF5 requires.
        if np.any(prow[1:] == prow[:-1]):
            raise RawLayoutError(f"Catalog references duplicate SED rows for patch {int(patch)}")
        for shard in shards:
            left, right = np.searchsorted(prow, [shard.row_offset, shard.row_offset + shard.n_rows])
            for start in range(left, right, piece):
                stop = min(start + piece, right)
                tasks.append((shard.path, prow[start:stop] - shard.row_offset, pout[start:stop]))
    return tasks


def _gather_worker(args):
    """Copy the requested raw rows of one shard into their buffer positions."""
    path, local, out, buf_path, shape, slab_rows = args
    buf = np.memmap(buf_path, dtype=np.float32, mode="r+", shape=shape)
    try:
        with h5py.File(path, "r") as source:
            sed = source["SED"]
            i = 0
            while i < len(local):
                first = int(local[i])
                j = int(np.searchsorted(local, first + slab_rows, side="left"))
                stop = int(local[j - 1]) + 1
                if stop - first <= _DENSE_SPAN_FACTOR * (j - i):
                    slab = sed[first:stop]
                    buf[out[i:j]] = slab[local[i:j] - first]
                else:
                    buf[out[i:j]] = sed[local[i:j]]
                i = j
    finally:
        del buf
    return len(local)


def _gather(tasks, buf_path, shape, slab_rows, processes, progress):
    work = [(path, local, out, buf_path, shape, slab_rows) for path, local, out in tasks]
    done = 0
    if processes <= 1:
        for item in work:
            done += _gather_worker(item)
            progress.update(done)
        return
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for future in as_completed([pool.submit(_gather_worker, item) for item in work]):
            done += future.result()
            progress.update(done)


def _create_sed(data, n_rows, width, codec, level, chunk_rows):
    chunks, options = sed_storage_opts(codec, level, n_rows, width, chunk_rows)
    return data.create_dataset("SED.partial", shape=(n_rows, width), dtype=np.float32, chunks=chunks, **options)


def _write_sed_header(dst, wave_rest):
    """Store the shared SED coordinate and its semantics in SuperMock header.

    This is deliberately a single header dataset rather than a per-object data
    column.  OpenCosmo-side support for these SuperMock extension fields is
    maintained separately.
    """
    if "header/supermock" not in dst:
        raise RawLayoutError(
            "SED append requires /header/supermock; copy a header donor before appending SEDs"
        )
    supermock = dst["header/supermock"]
    name = "sed_wave_rest"
    if name in supermock:
        wave = supermock[name]
        if not np.array_equal(wave[:], wave_rest):
            raise RawLayoutError("header/supermock/sed_wave_rest differs from SED source grid")
    else:
        wave = supermock.create_dataset(name, data=wave_rest)
    wave.attrs["unit"] = "Angstrom"
    wave.attrs["description"] = "Rest-frame wavelength grid shared by data/SED axis 1"
    supermock.attrs["sed_dataset_path"] = "/data/SED"
    supermock.attrs["sed_wavelength_path"] = "/header/supermock/sed_wave_rest"
    supermock.attrs["sed_wavelength_axis"] = np.int64(1)
    supermock.attrs["sed_flux_density"] = "fnu"
    supermock.attrs["sed_flux_unit"] = "Jy"
    supermock.attrs["sed_wavelength_relation"] = "lambda_obs = wave_rest * (1+z)"
    supermock.attrs["sed_redshift_path"] = "/data/redshift"


def append_one(path, sources, *, codec="blosc-lz4", level=5, chunk_rows=DEFAULT_CHUNK_ROWS,
               block_mib=DEFAULT_BLOCK_MIB, processes=1, buffer_dir=None, scratch_dir=None,
               overwrite=False, verify_samples=16):
    """Append SEDs in place; incomplete work remains visibly named SED.partial."""
    with h5py.File(path, "r+") as dst:
        if "data" not in dst or "skypatch" not in dst["data"] or "source_global_row" not in dst["data"]:
            raise RawLayoutError(f"{path} lacks /data/skypatch or /data/source_global_row; rerun the catalog pipeline with provenance enabled")
        data = dst["data"]
        n_rows = data["skypatch"].shape[0]
        if data["source_global_row"].shape != (n_rows,):
            raise RawLayoutError(f"{path} has invalid source_global_row shape")
        if "SED" in data and not overwrite:
            raise RawLayoutError(f"{path} already has /data/SED (use --overwrite)")
        if "SED" in data:
            del data["SED"]
        if "SED.partial" in data:
            del data["SED.partial"]
        if "wave_rest" in dst:
            raise RawLayoutError(
                f"{path} has legacy /wave_rest; migrate it to /header/supermock/sed_wave_rest first"
            )
        _write_sed_header(dst, sources.wave_rest)

        width = sources.width
        sed = _create_sed(data, n_rows, width, codec, level, chunk_rows)
        sed.attrs["unit"] = "Jy"
        sed.attrs["sed_format"] = "fnu_jy_obs_on_restgrid"
        sed.attrs["wave_convention"] = "lambda_obs = wave_rest * (1+z)"
        sed.attrs["source_identity"] = "(skypatch, source_global_row)"
        data.attrs["sed_append_complete"] = False

        if n_rows:
            patches = data["skypatch"][:].astype(np.int64)
            rows = data["source_global_row"][:].astype(np.int64)
            shape = (n_rows, width)
            nbytes = n_rows * width * np.dtype(np.float32).itemsize
            directory = _buffer_dir(buffer_dir, scratch_dir or os.path.dirname(os.path.abspath(path)), nbytes)
            buf_path = os.path.join(directory, f".sed_buffer_{os.getpid()}_{os.path.basename(path)}.f32")
            row_bytes = width * np.dtype(np.float32).itemsize
            block_rows = max(1, (int(block_mib) << 20) // row_bytes)
            print(f"  SED buffer: {nbytes / 2**30:,.1f} GiB in {directory}", flush=True)
            try:
                np.memmap(buf_path, dtype=np.float32, mode="w+", shape=shape).flush()
                tasks = _gather_tasks(sources, patches, rows, 4 * processes)
                del patches, rows
                _gather(tasks, buf_path, shape, block_rows, processes,
                        Progress(f"SED gather ({os.path.basename(path)})", n_rows))

                # Chunk-aligned blocks: each compressed chunk is written once.
                if sed.chunks:
                    block_rows = max(sed.chunks[0], block_rows // sed.chunks[0] * sed.chunks[0])
                buf = np.memmap(buf_path, dtype=np.float32, mode="r", shape=shape)
                previous_threads = os.environ.get("BLOSC_NTHREADS")
                os.environ["BLOSC_NTHREADS"] = str(os.cpu_count() or 1)
                try:
                    progress = Progress(f"SED write ({os.path.basename(path)})", n_rows)
                    for lo in range(0, n_rows, block_rows):
                        hi = min(lo + block_rows, n_rows)
                        sed[lo:hi] = buf[lo:hi]
                        progress.update(hi)
                    progress.done()
                finally:
                    if previous_threads is None:
                        os.environ.pop("BLOSC_NTHREADS", None)
                    else:
                        os.environ["BLOSC_NTHREADS"] = previous_threads
                    del buf
            finally:
                if os.path.exists(buf_path):
                    os.remove(buf_path)

        # Validate identity data without re-reading spectra.  The catalog fields
        # have already undergone the identical redshift/spatial permutations.
        if verify_samples and "core_tag" in data and "redshift" in data:
            samples = np.unique(np.linspace(0, n_rows - 1, min(verify_samples, n_rows), dtype=np.int64))
            patches = data["skypatch"][samples]
            rows = data["source_global_row"][samples]
            for patch in np.unique(patches):
                positions = np.flatnonzero(patches == patch)
                # Read identity fields directly from the appropriate shards.
                for pos in positions:
                    row = int(rows[pos])
                    shard = next(item for item in sources.shards[int(patch)] if item.row_offset <= row < item.row_offset + item.n_rows)
                    local = row - shard.row_offset
                    source = sources._handle(shard.path)
                    if source["core_tag"][local] != data["core_tag"][samples[pos]] or not np.isclose(source["redshift"][local], data["redshift"][samples[pos]], rtol=0, atol=1e-6):
                        raise RawLayoutError(f"{path} failed SED identity validation at final row {samples[pos]}")
        data.move("SED.partial", "SED")
        data.attrs["sed_append_complete"] = True
        dst.attrs["sed_append_source"] = "raw sharded hi-res SEDs"
        dst.flush()
    print(f"appended SEDs: {path}")


def main():
    parser = argparse.ArgumentParser(description="Append raw sharded hi-res SEDs to completed spatial catalogs.")
    parser.add_argument("catalog", nargs="*", help="completed spatial catalog(s)")
    parser.add_argument("--catalog-dir", help="directory containing completed spatial .hdf5 catalogs")
    parser.add_argument("--sed-dir", default="hires", help="directory containing raw SED shards (default: hires)")
    parser.add_argument("--processes", type=int, default=1, help="parallel shard readers for the gather phase (default: 1)")
    parser.add_argument("--block-mib", type=int, default=DEFAULT_BLOCK_MIB,
                        help=f"uncompressed MiB per raw read slab and per output write (default: {DEFAULT_BLOCK_MIB})")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"rows per output SED chunk (default: {DEFAULT_CHUNK_ROWS})")
    parser.add_argument("--compression", choices=SED_CODECS, default="blosc-lz4",
                        help="SED codec (default: blosc-lz4, byte-shuffled)")
    parser.add_argument("--compression-level", type=int, default=5, help="compression level where supported (default: 5)")
    parser.add_argument("--buffer-dir", help="directory for the uncompressed gather buffer (default: /dev/shm, else the catalog's directory)")
    parser.add_argument("--verify-samples", type=int, default=16, help="identity samples per catalog (default: 16; 0 disables)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing /data/SED")
    args = parser.parse_args()
    if args.block_mib < 1 or args.chunk_rows < 1 or args.processes < 1 or args.verify_samples < 0:
        sys.exit("--block-mib, --chunk-rows and --processes must be >= 1 and --verify-samples must be >= 0")
    try:
        paths = _output_paths(args.catalog, args.catalog_dir)
        # Scan provenance in bounded slabs, then require only patches actually
        # represented in the selected final catalog files.
        patches = set()
        for path in paths:
            with h5py.File(path, "r") as handle:
                if "data/skypatch" not in handle:
                    raise RawLayoutError(f"{path} lacks /data/skypatch")
                skypatch = handle["data/skypatch"]
                for lo in range(0, skypatch.shape[0], 1 << 20):
                    patches.update(map(int, np.unique(skypatch[lo:lo + (1 << 20)])))
        sources = SedSources(args.sed_dir, sorted(patches))
        try:
            for path in paths:
                append_one(path, sources, codec=args.compression, level=args.compression_level,
                           chunk_rows=args.chunk_rows, block_mib=args.block_mib, processes=args.processes,
                           buffer_dir=args.buffer_dir, overwrite=args.overwrite,
                           verify_samples=args.verify_samples)
        finally:
            sources.close()
    except RawLayoutError as exc:
        sys.exit(f"Raw layout error: {exc}")


if __name__ == "__main__":
    main()
