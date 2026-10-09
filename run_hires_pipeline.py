"""Run the SuperMock catalog pipeline, optionally including hi-res SEDs."""

import os
import shutil

import click

from append_hires_seds import DEFAULT_BLOCK_MIB, DEFAULT_CHUNK_ROWS, SED_CODECS, SedSources, append_one
from repartition_by_redshift import repartition
from spatial_index import OutputPolicy, run as spatial_run
from supermock_raw import RawLayoutError, set_blosc_threads


def _link_sources(run_dir, catalog_dir, luminosity_dir, photometry_dir):
    """Build the raw-root shape expected by the catalog stages without copying."""
    root = os.path.join(run_dir, "raw_sources")
    os.makedirs(root, exist_ok=True)
    for name, source in (
        ("lightcone_catalogs", catalog_dir),
        ("luminosities", luminosity_dir),
        ("photometry", photometry_dir),
    ):
        target = os.path.join(root, name)
        if os.path.lexists(target):
            if os.path.realpath(target) != os.path.realpath(source):
                raise RawLayoutError(f"Scratch source link already exists with another target: {target}")
            continue
        os.symlink(os.path.abspath(source), target)
    return root


def _final_paths(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".hdf5")
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--catalog-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True), required=True,
              help="Folder containing lightcone_galaxies_skypatch_<P>.h5 files.")
@click.option("--luminosity-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True), required=True,
              help="Folder containing luminosities_skypatch_<P>.h5 files.")
@click.option("--photometry-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True), required=True,
              help="Folder containing photometry_skypatch_<P>.h5 files.")
@click.option("--sed-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True),
              help="Optional folder containing complete sharded hi-res SED inputs.")
@click.option("--header-source", type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="Header donor copied into final catalogs; required with --sed-dir.")
@click.option("--output-dir", type=click.Path(file_okay=False, resolve_path=True), required=True,
              help="Destination for completed OpenCosmo HDF5 catalogs.")
@click.option("--scratch-dir", type=click.Path(file_okay=False, resolve_path=True), required=True,
              help="Local SSD working directory; intermediates are retained here.")
@click.option("--patch", "patches", type=int, multiple=True,
              help="Restrict to this skypatch; repeat option for multiple patches.")
@click.option("--n-partitions", type=click.IntRange(1, None), default=16, show_default=True)
@click.option("--prefix", default="SuperMock_v3", show_default=True)
@click.option("--decimals", type=click.IntRange(0, None), default=2, show_default=True)
@click.option("--spatial-level", type=click.IntRange(0, None), default=5, show_default=True)
@click.option("--processes", type=click.IntRange(1, None), default=1, show_default=True,
              help="Worker processes for catalog stages and for the SED gather.")
@click.option("--sed-block-mib", type=click.IntRange(1, None), default=DEFAULT_BLOCK_MIB, show_default=True,
              help="Uncompressed MiB per raw SED read slab and per SED output write.")
@click.option("--sed-chunk-rows", type=click.IntRange(1, None), default=DEFAULT_CHUNK_ROWS, show_default=True,
              help="Rows per /data/SED chunk; small chunks keep single-galaxy reads cheap.")
@click.option("--sed-compression", type=click.Choice(SED_CODECS), default="blosc-lz4", show_default=True,
              help="Codec for /data/SED (byte-shuffled); independent of --compression.")
@click.option("--sed-compression-level", type=int, default=5, show_default=True)
@click.option("--sed-buffer-dir", type=click.Path(file_okay=False, resolve_path=True),
              help="Where to hold one uncompressed redshift slice of SEDs; default /dev/shm, else --scratch-dir.")
@click.option("--spatial-block-rows", type=click.IntRange(1, None), default=None,
              help="Rows per spatial output write; default follows each output chunk.")
@click.option("--compression", type=click.Choice(["blosc", "gzip", "lzf", "none"]), default="blosc", show_default=True)
@click.option("--compression-level", type=int, default=5, show_default=True)
@click.option("--overwrite", is_flag=True, help="Replace existing scratch and final output files.")
@click.option("--keep-intermediates/--remove-intermediates", default=True, show_default=True,
              help="Keep/remove scratch repartitioned files after success.")
def main(catalog_dir, luminosity_dir, photometry_dir, sed_dir, header_source, output_dir, scratch_dir,
         patches, n_partitions, prefix, decimals, spatial_level, processes,
         sed_block_mib, sed_chunk_rows, sed_compression, sed_compression_level, sed_buffer_dir,
         spatial_block_rows, compression, compression_level,
         overwrite, keep_intermediates):
    """Build final spatial OpenCosmo catalogs, optionally appending hi-res SEDs."""
    run_dir = os.path.join(scratch_dir, "supermock_hires_pipeline")
    repartitioned = os.path.join(run_dir, "repartitioned")
    spatial = os.path.join(run_dir, "spatially_indexed")
    try:
        if sed_dir is not None and header_source is None:
            raise RawLayoutError("--header-source is required when --sed-dir is supplied")
        os.makedirs(run_dir, exist_ok=True)
        raw_root = _link_sources(run_dir, catalog_dir, luminosity_dir, photometry_dir)
        if not overwrite and (_final_paths(repartitioned) or _final_paths(spatial)):
            raise RawLayoutError("Scratch stage outputs exist; use --overwrite or select a new --scratch-dir")
        if overwrite:
            for directory in (repartitioned, spatial):
                if os.path.exists(directory):
                    shutil.rmtree(directory)

        stage_count = 3 if sed_dir is not None else 2
        click.echo(f"Stage 1/{stage_count}: repartitioning galaxies, luminosities, and photometry")
        repartition(raw_root, repartitioned, n_partitions, prefix, decimals, processes,
                    patches=patches or None)

        click.echo(f"Stage 2/{stage_count}: spatial indexing normal catalog columns")
        policy = OutputPolicy(compression, compression_level, 32, 1 << 20)
        set_blosc_threads(processes)
        spatial_run(repartitioned, spatial, spatial_level, spatial_block_rows, True,
                    processes, policy)

        paths = _final_paths(spatial)
        if not paths:
            raise RawLayoutError("Spatial indexing produced no final catalogs")
        if sed_dir is not None:
            # copy_header imports hdf5plugin to read donor headers from existing
            # Blosc catalogs; keep that optional dependency out of catalog-only
            # pipeline invocation.
            from copy_header import copy_header
            click.echo("Writing and specializing OpenCosmo headers")
            copy_header(header_source, paths, replace=overwrite, specialise_fields=True, first_step=0)
            # The selected redshift partitions can mix patches; discover the
            # actual source patches from the already-finished final files.
            import h5py
            import numpy as np
            source_patches = set()
            for path in paths:
                with h5py.File(path, "r") as handle:
                    skypatch = handle["data/skypatch"]
                    for lo in range(0, skypatch.shape[0], 1 << 20):
                        source_patches.update(map(int, np.unique(skypatch[lo:lo + (1 << 20)])))
            click.echo(f"Stage 3/{stage_count}: gathering and appending hi-res SEDs")
            sources = SedSources(sed_dir, sorted(source_patches))
            try:
                for path in paths:
                    append_one(path, sources, codec=sed_compression, level=sed_compression_level,
                               chunk_rows=sed_chunk_rows, block_mib=sed_block_mib, processes=processes,
                               buffer_dir=sed_buffer_dir, scratch_dir=run_dir,
                               overwrite=True, verify_samples=16)
            finally:
                sources.close()
        os.makedirs(output_dir, exist_ok=True)
        for path in paths:
            destination = os.path.join(output_dir, os.path.basename(path))
            if os.path.exists(destination) and not overwrite:
                raise RawLayoutError(f"Final output exists (use --overwrite): {destination}")
            if os.path.exists(destination):
                os.remove(destination)
            shutil.move(path, destination)
        if not keep_intermediates:
            shutil.rmtree(repartitioned)
        click.echo(f"Done: {len(paths)} final catalog(s) in {output_dir}")
    except RawLayoutError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
