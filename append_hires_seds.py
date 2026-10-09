"""Append sharded hi-res SEDs to completed spatial OpenCosmo catalogs.

The normal catalog pipeline first freezes each final row's source identity in
``/data/skypatch`` and ``/data/source_global_row``.  This program then joins
the corresponding SED rows in bounded output-row blocks.  It does *not* build
an intermediate 1.5-TB flat SED copy.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

from supermock_raw import RawLayoutError, add_compression_args, discover_sed_shards, storage_opts


def _output_paths(paths, directory):
    result = list(paths)
    if directory:
        result.extend(sorted(glob.glob(os.path.join(directory, "*.hdf5"))))
    result = list(dict.fromkeys(result))
    if not result:
        raise RawLayoutError("Specify one or more catalogs or --catalog-dir")
    return result


class SedSources:
    """Lazily opened raw SED shards, addressed by (patch, producer row)."""

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

    def read(self, patch, rows):
        """Read global producer rows in caller order using sorted shard reads."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty((0, self.width), dtype=np.float32)
        shards = self.shards[int(patch)]
        if np.any(rows < 0) or np.any(rows >= shards[0].n_global):
            raise RawLayoutError(f"Catalog references out-of-range SED rows for patch {patch}")
        result = np.empty((len(rows), self.width), dtype=np.float32)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        restore = np.empty(len(rows), dtype=np.int64)
        restore[order] = np.arange(len(rows))
        for shard in shards:
            lo, hi = shard.row_offset, shard.row_offset + shard.n_rows
            left = np.searchsorted(sorted_rows, lo, side="left")
            right = np.searchsorted(sorted_rows, hi, side="left")
            if left == right:
                continue
            local = sorted_rows[left:right] - lo
            # A patch contributes each raw row only once to a redshift output,
            # therefore local indices are strictly increasing as HDF5 requires.
            values = self._handle(shard.path)["SED"][local]
            result[order[left:right]] = values
        return result


def _create_sed(data, n_rows, width, compression, level, block_rows):
    chunks, options = storage_opts(
        np.float32, (n_rows, width), compression, level, True,
        block_rows * width * np.dtype("float32").itemsize,
    )
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


def append_one(path, sources, block_mib, compression, level, overwrite, verify_samples):
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

        block_rows = max(1, (int(block_mib) << 20) // (sources.width * 4))
        sed = _create_sed(data, n_rows, sources.width, compression, level, block_rows)
        sed.attrs["unit"] = "Jy"
        sed.attrs["sed_format"] = "fnu_jy_obs_on_restgrid"
        sed.attrs["wave_convention"] = "lambda_obs = wave_rest * (1+z)"
        sed.attrs["source_identity"] = "(skypatch, source_global_row)"
        data.attrs["sed_append_complete"] = False

        for lo in range(0, n_rows, block_rows):
            hi = min(lo + block_rows, n_rows)
            patches = data["skypatch"][lo:hi]
            source_rows = data["source_global_row"][lo:hi]
            values = np.empty((hi - lo, sources.width), dtype=np.float32)
            for patch in np.unique(patches):
                positions = np.flatnonzero(patches == patch)
                if int(patch) not in sources.shards:
                    raise RawLayoutError(f"{path} references SED patch {int(patch)}, which was not supplied")
                values[positions] = sources.read(int(patch), source_rows[positions])
            sed[lo:hi] = values

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
    parser.add_argument("--block-mib", type=int, default=512, help="uncompressed SED payload per output block in MiB (default: 512)")
    parser.add_argument("--verify-samples", type=int, default=16, help="identity samples per catalog (default: 16; 0 disables)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing /data/SED")
    add_compression_args(parser)
    args = parser.parse_args()
    if args.block_mib < 1 or args.verify_samples < 0:
        sys.exit("--block-mib must be >= 1 and --verify-samples must be >= 0")
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
                append_one(path, sources, args.block_mib, args.compression, args.compression_level, args.overwrite, args.verify_samples)
        finally:
            sources.close()
    except RawLayoutError as exc:
        sys.exit(f"Raw layout error: {exc}")


if __name__ == "__main__":
    main()
