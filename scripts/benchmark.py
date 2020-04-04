#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import czifile
import imageio
import numpy as np
import psutil
import tifffile
from dask_jobqueue import SLURMCluster
from distributed import Client
from tqdm import tqdm

import aicsimageio
from aicsimageio import dask_utils

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################
# Args


class Args(argparse.Namespace):
    def __init__(self):
        self.__parse()

    def __parse(self):
        # Setup parser
        p = argparse.ArgumentParser(
            prog="benchmark_aicsimageio",
            description=(
                "Run read time benchmarks for aicsimageio against other common image "
                "readers. The benchmark dataset can be downloaded using the "
                "download_test_resources script with the specific hash: "
                "{INSERT HASH HERE}"
            )
        )

        # Arguments
        p.add_argument(
            "--save_path",
            default="benchmark/results.json",
            type=Path,
            help="Path to save the generated benchmark JSON file.",
        )
        p.add_argument(
            "--distributed",
            action="store_true",
            help="Run distributed cluster benchmarks.",
        )
        p.add_argument(
            "--debug",
            action="store_true",
            help="Show traceback if the script were to fail.",
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################

def _run_benchmark(
    resources_dir: Path,
    extensions: List[str],
    non_aicsimageio_reader: List[Callable],
    iterations: int = 5,
):
    # Collect files matching the extensions provided
    files = []
    for ext in extensions:
        files += list(resources_dir.glob(ext))

    # Run reads for each file and store details in results
    per_file_results = []
    for file in files:
        info_read = aicsimageio.AICSImage(file)
        yx_planes = np.prod(info_read.size("STCZ"))
        for reader in [aicsimageio.imread, non_aicsimageio_reader]:
            reader_path = f"{reader.__module__}.{reader.__name__}"
            read_durations = []
            for i in tqdm(range(iterations), desc=f"{reader_path}: {file.name}"):
                start = datetime.utcnow()
                reader(str(file))
                read_durations.append((datetime.utcnow() - start).total_seconds())

            # Append average read time and other info
            per_file_results.append([{
                "file_name": file.name,
                "file_size_gb": file.stat().st_size / 10e8,
                "reader": "aicsimageio" if "aicsimageio" in reader_path else "other",
                "yx_planes": int(yx_planes),
                "read_duration": read_duration,
            } for read_duration in read_durations])

    # Unpack per file results
    results = []
    for per_file_result in per_file_results:
        results += per_file_result
    return results


def _run_benchmark_suite(resources_dir: Path):
    # Default reader / imageio imread tests
    default_reader_single_image_results = _run_benchmark(
        resources_dir=resources_dir,
        extensions=["*.png", "*.jpg", "*.bmp"],
        non_aicsimageio_reader=imageio.imread,
    )

    # Default reader / imageio mimread tests
    default_reader_many_image_results = _run_benchmark(
        resources_dir=resources_dir,
        extensions=["*.gif"],
        non_aicsimageio_reader=imageio.mimread,
    )

    # Tiff reader / tifffile imread tests
    tiff_reader_results = _run_benchmark(
        resources_dir=resources_dir,
        extensions=["*.tiff"],
        non_aicsimageio_reader=tifffile.imread,
    )

    # CZI reader / czifile imread tests
    czi_reader_results = _run_benchmark(
        resources_dir=resources_dir,
        extensions=["*.czi"],
        non_aicsimageio_reader=czifile.imread,
    )

    return [
        *default_reader_single_image_results,
        *default_reader_many_image_results,
        *tiff_reader_results,
        *czi_reader_results,
    ]


def run_benchmarks(args: Args):
    # Try running the benchmarks
    try:
        # Get benchmark resources dir
        resources_dir = Path().parent.parent / "aicsimageio" / "tests" / "resources"

        # Store machine config
        machine_config = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "cpu_total_count": psutil.cpu_count(),
            "cpu_current_utilization": psutil.cpu_percent(),
            "memory_total_gb": psutil.virtual_memory().total / 10e8,
            "memory_available_gb": psutil.virtual_memory().available / 10e8,
        }

        # Store python config
        pyversion = sys.version_info
        python_config = {
            "python_version": f"{pyversion.major}.{pyversion.minor}.{pyversion.micro}",
            "aicsimageio": aicsimageio.__version__,
            "czifile": czifile.__version__,
            "imageio": imageio.__version__,
            "tifffile": tifffile.__version__,
        }

        # Run tests
        #######################################################################

        log.info(f"Running tests (no cluster)...")
        log.info(f"=" * 80)

        no_cluster_results = _run_benchmark_suite(resources_dir=resources_dir)

        #######################################################################

        log.info(f"Running tests (local cluster)...")
        log.info(f"=" * 80)

        with dask_utils.cluster_and_client() as (cluster, client):
            # Store cluster configuration
            worker_spec = cluster.worker_spec[0]["options"]
            local_cluster_config = {
                "workers": len(cluster.workers),
                "per_worker_thread_allocation": worker_spec["nthreads"],
                "per_worker_memory_allocation_gb": worker_spec["memory_limit"] / 10e8,
            }

            local_cluster_results = _run_benchmark_suite(resources_dir=resources_dir)

        #######################################################################

        if args.distributed:
            log.info(f"Running tests (distributed cluster)...")
            log.info(f"=" * 80)

            # Create or get log dir
            # Do not include ms
            log_dir_name = datetime.now().isoformat().split(".")[0]
            log_dir = Path(f"~/.dask_logs/aicsimageio/{log_dir_name}").expanduser()
            # Log dir get or create
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create SLURM Cluster
            cluster = SLURMCluster(
                        cores=1,
                        memory="2GB",
                        queue="aics_cpu_general",
                        walltime="1:00:00",
                        local_directory=str(log_dir),
                        log_directory=str(log_dir),
                    )

            # Scale workers
            cluster.scale(64)

            # Create client connection
            client = Client(cluster)

            # Store cluster configuration
            worker_spec = cluster.worker_spec[0]["options"]
            distributed_cluster_config = {
                "workers": len(cluster.workers),
                "per_worker_core_allocation": worker_spec["cores"],
                "per_worker_memory_allocation_gb": worker_spec["memory"],
            }

            distributed_cluster_results = _run_benchmark_suite(
                resources_dir=resources_dir
            )

            client.shutdown()
            cluster.close()

        else:
            distributed_cluster_config = None
            distributed_cluster_results = []

        #######################################################################

        log.info(f"Completed all tests")
        log.info(f"=" * 80)

        # Store results in a single JSON
        all_results = {
            "machine_config": machine_config,
            "python_config": python_config,
            "local_cluster_config": local_cluster_config,
            "distributed_cluster_config": distributed_cluster_config,
            "no_cluster_results": no_cluster_results,
            "local_cluster_results": local_cluster_results,
            "distributed_cluster_results": distributed_cluster_results,
        }

        # Ensure save dir exists and save results
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_path, "w") as write_out:
            json.dump(all_results, write_out)

    # Catch any exception
    except Exception as e:
        log.error("=============================================")
        if args.debug:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)

###############################################################################
# Runner


def main():
    args = Args()
    run_benchmarks(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
