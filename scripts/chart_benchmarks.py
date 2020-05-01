#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from pathlib import Path

import altair as alt
import pandas as pd
from quilt3 import Package

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
            prog="generate_benchmark_charts",
            description=(
                "Generate charts from benchmark result data. See benchmark.py"
            )
        )

        # Arguments
        p.add_argument(
            "--benchmark_file",
            default=None,
            help=(
                "Path to a previously generated benchmark JSON file. "
                "Default: retrieve latest file from Quilt."
            ),
        )
        p.add_argument(
            "--save_dir",
            default="benchmark/charts/",
            type=Path,
            help="Path to where the benchmark charts should be saved.",
        )
        p.add_argument(
            "--debug",
            action="store_true",
            help="Show traceback if the script were to fail.",
        )

        # Parse
        p.parse_args(namespace=self)

###############################################################################

SELECTED_CLUSTERS_TO_VISUALIZE = [
    "no-cluster",
    "small-local-cluster-replica",
    "many-small-worker-distributed-cluster",
]


def _generate_chart(results: pd.DataFrame, sorted: bool = False):
    if sorted:
        column = alt.Column("config:O", sort=SELECTED_CLUSTERS_TO_VISUALIZE)
    else:
        column = "config:O"

    return alt.Chart(results).mark_circle().encode(
        x="yx_planes:Q",
        y="read_duration:Q",
        color="reader:N",
        column=column,
    )


def chart_benchmarks(args: Args):
    # Check save dir exists or create
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Get file
    if args.benchmark_file is None:
        benchmark_filepath = Path("benchmark_results.json")
        p = Package.browse(
            "aicsimageio/benchmarks",
            "s3://aics-modeling-packages-test-resources"
        )
        p["results.json"].fetch(benchmark_filepath)
    else:
        benchmark_filepath = args.benchmark_file

    # Read results file
    with open(benchmark_filepath, "r") as read_in:
        all_results = json.load(read_in)

    # Generate charts for each config
    per_cluster_results = []
    selected_cluster_results = []
    for config_name, results in all_results.items():
        results = pd.DataFrame(results)
        results["config"] = config_name

        # Add to all
        per_cluster_results.append(results)

        # Add to primary viz
        if config_name in SELECTED_CLUSTERS_TO_VISUALIZE:
            selected_cluster_results.append(results)

        chart = _generate_chart(results)
        chart.save(str(args.save_dir / f"{config_name}.png"))

    # Generate unified chart
    all_results = pd.concat(per_cluster_results)
    unified_chart = _generate_chart(all_results)
    unified_chart.save(str(args.save_dir / "all.png"))

    # Generate unified primary chart
    primary_results = pd.concat(selected_cluster_results)
    unified_chart = _generate_chart(primary_results, sorted=True)
    unified_chart.configure_header(
        labelBaseline="top"
    )
    unified_chart.save(str(args.save_dir / "primary.png"))

###############################################################################
# Runner


def main():
    args = Args()
    chart_benchmarks(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
