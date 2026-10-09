"""Convert complete sharded hi-res SuperMock SEDs with bounded memory.

Outputs retain producer global-row order as a companion product.  This avoids
the normal spatial indexer's whole-column RAM strategy, which is unsuitable for
a roughly 1.5-TB SED matrix.
"""

import argparse
import os
import sys

import h5py
import numpy as np

from supermock_raw import (
    RawLayoutError, add_compression_args, discover_sed_shards, open_output,
    patch_offsets, set_blosc_threads, storage_opts,
)


def _companion_path(input_dir, patch, stem):
    """Accept both flat hires files and ordinary raw/<kind> files."""
    flat = os.path.join(input_dir, f"{stem}_skypatch_{patch}.h5")
    if os.path.exists(flat):
        return flat
    kinds = {"lightcone_galaxies": "lightcone_catalogs", "luminosities": "luminosities"}
    nested = os.path.join(input_dir, kinds[stem], os.path.basename(flat))
    if os.path.exists(nested):
        return nested
    raise RawLayoutError(f"Missing hi-res companion file: {flat}")


def _flat_count(path, label):
    with h5py.File(path, "r") as handle:
        counts = {name: value.shape[0] for name, value in handle.items()}
    if not counts or len(set(counts.values())) != 1:
        raise RawLayoutError(f"{label} {path} has inconsistent row counts: {counts}")
    return next(iter(counts.values()))


def _catalog_values(catalog, groups, offsets, rows, dataset):
    """Read selected global rows without flattening the grouped catalog."""
    values = []
    offsets = np.asarray(offsets)
    for row in rows:
        group_index = int(np.searchsorted(offsets, row, side="right") - 1)
        values.append(catalog[groups[group_index]][dataset][int(row - offsets[group_index])])
    return np.asarray(values)


def validate_patch(input_dir, patch, shards, samples_per_shard=16):
    """Validate counts and sampled SED/catalog/luminosity alignment."""
    catalog_path = _companion_path(input_dir, patch, "lightcone_galaxies")
    luminosity_path = _companion_path(input_dir, patch, "luminosities")
    groups, _, offsets, n_catalog = patch_offsets(catalog_path)
    n_luminosity = _flat_count(luminosity_path, "Luminosity")
    n_global = shards[0].n_global
    if n_catalog != n_global or n_luminosity != n_global:
        raise RawLayoutError(
            f"Patch {patch} row counts disagree: catalog={n_catalog:,}, "
            f"luminosity={n_luminosity:,}, SED={n_global:,}"
        )
    if samples_per_shard < 1:
        return n_global
    with h5py.File(catalog_path, "r") as catalog, h5py.File(luminosity_path, "r") as luminosities:
        if "redshift" not in luminosities:
            raise RawLayoutError(f"Luminosity file {luminosity_path} lacks redshift")
        for shard in shards:
            local = np.unique(np.linspace(0, shard.n_rows - 1, min(samples_per_shard, shard.n_rows), dtype=np.int64))
            rows = shard.row_offset + local
            with h5py.File(shard.path, "r") as source:
                catalog_tags = _catalog_values(catalog, groups, offsets, rows, "core_tag")
                if not np.array_equal(source["core_tag"][local], catalog_tags):
                    raise RawLayoutError(f"SED core_tag alignment failed for {shard.path}")
                sed_z = source["redshift"][local]
                catalog_z = _catalog_values(catalog, groups, offsets, rows, "redshift")
                luminosity_z = luminosities["redshift"][rows]
                if not (np.allclose(sed_z, catalog_z, rtol=0, atol=1e-10) and np.allclose(sed_z, luminosity_z, rtol=0, atol=1e-6)):
                    raise RawLayoutError(f"SED redshift alignment failed for {shard.path}")
    return n_global


def _create_dataset(parent, name, shape, dtype, compression, level, chunk_bytes):
    chunks, options = storage_opts(dtype, shape, compression, level, True, chunk_bytes)
    return parent.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, **options)


def convert_patch(input_dir, output_dir, patch, shards, block_mib, compression, level,
                  chunk_cache_mb, overwrite, samples_per_shard):
    n_rows = validate_patch(input_dir, patch, shards, samples_per_shard)
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"hires_seds_skypatch_{patch}.h5")
    partial_path = final_path + ".partial"
    if os.path.exists(final_path) and not overwrite:
        raise RawLayoutError(f"Output exists (use --overwrite): {final_path}")
    if os.path.exists(partial_path):
        os.remove(partial_path)
    width = shards[0].width
    block_rows = max(1, (int(block_mib) << 20) // (width * np.dtype("float32").itemsize))
    try:
        with open_output(partial_path, chunk_cache_mb=chunk_cache_mb) as dst:
            data = dst.create_group("data")
            sed = _create_dataset(data, "SED", (n_rows, width), np.float32, compression, level, block_rows * width * 4)
            tags = _create_dataset(data, "core_tag", (n_rows,), np.int64, compression, level, block_rows * 8)
            redshift = _create_dataset(data, "redshift", (n_rows,), np.float64, compression, level, block_rows * 8)
            sed.attrs["unit"] = "Jy"
            sed.attrs["sed_format"] = "fnu_jy_obs_on_restgrid"
            sed.attrs["wave_convention"] = "lambda_obs = wave_rest * (1+z)"
            dst.attrs["skypatch"] = patch
            dst.attrs["n_objects"] = n_rows
            dst.attrs["source_row_order"] = "producer global flat row order"
            wave = None
            for shard in shards:
                with h5py.File(shard.path, "r") as source:
                    source_wave = source["wave_rest"][:]
                    if wave is None:
                        wave = dst.create_dataset("wave_rest", data=source_wave)
                        wave.attrs["description"] = "Rest-frame wavelength grid shared by all SED rows"
                    elif not np.array_equal(source_wave, wave[:]):
                        raise RawLayoutError(f"SED wavelength grids disagree: {shard.path}")
                    for start in range(0, shard.n_rows, block_rows):
                        end = min(start + block_rows, shard.n_rows)
                        out = slice(shard.row_offset + start, shard.row_offset + end)
                        sed[out] = source["SED"][start:end]
                        tags[out] = source["core_tag"][start:end]
                        redshift[out] = source["redshift"][start:end]
        # Keep a prior completed output intact until every source shard has
        # copied successfully.  On the local scratch filesystem this is an
        # atomic replacement, not a second multi-terabyte copy.
        os.replace(partial_path, final_path)
    except BaseException:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise
    print(f"patch {patch}: wrote {n_rows:,} SEDs to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Bounded-memory conversion of sharded hi-res SuperMock SEDs.")
    parser.add_argument("--input-dir", default="hires", help="flat hires input directory (default: hires)")
    parser.add_argument("--output-dir", default="hires_converted", help="directory for one output per skypatch")
    parser.add_argument("--patch", type=int, action="append", help="patch to convert (repeatable; default: all found)")
    parser.add_argument("--block-mib", type=int, default=512, help="maximum uncompressed SED block payload in MiB (default: 512)")
    parser.add_argument("--chunk-cache-mb", type=int, default=32, help="HDF5 raw chunk cache in MiB (default: 32)")
    parser.add_argument("--verify-samples", type=int, default=16, help="catalog identity samples per shard (0 disables; default: 16)")
    parser.add_argument("--overwrite", action="store_true", help="replace completed outputs")
    parser.add_argument("--verify", action="store_true", help="validate inputs only; create no output")
    add_compression_args(parser)
    args = parser.parse_args()
    if args.block_mib < 1 or args.chunk_cache_mb < 1 or args.verify_samples < 0:
        sys.exit("--block-mib and --chunk-cache-mb must be >= 1; --verify-samples must be >= 0")
    try:
        requested = sorted(set(args.patch)) if args.patch is not None else None
        # Scope discovery when the caller requested particular patches: an
        # unrelated incomplete transfer must not prevent converting a complete
        # requested skypatch.
        all_shards = ({patch: discover_sed_shards(args.input_dir, patch)[patch]
                       for patch in requested}
                      if requested is not None else discover_sed_shards(args.input_dir))
        patches = requested if requested is not None else list(all_shards)
        set_blosc_threads(1)
        for patch in patches:
            if args.verify:
                rows = validate_patch(args.input_dir, patch, all_shards[patch], args.verify_samples)
                print(f"patch {patch}: verified {rows:,} rows")
            else:
                convert_patch(input_dir=args.input_dir, output_dir=args.output_dir, patch=patch,
                              shards=all_shards[patch], block_mib=args.block_mib,
                              compression=args.compression, level=args.compression_level,
                              chunk_cache_mb=args.chunk_cache_mb, overwrite=args.overwrite,
                              samples_per_shard=args.verify_samples)
    except RawLayoutError as exc:
        sys.exit(f"Raw layout error: {exc}")


if __name__ == "__main__":
    main()
