#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from pathlib import Path

import altair as alt
import pandas as pd

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
            default="benchmark/results.json",
            type=Path,
            help="Path to a previously generated benchmark JSON file.",
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


def _generate_chart(results: pd.DataFrame):
    return alt.Chart(results).mark_circle().encode(
        x="yx_planes:Q",
        y="read_duration:Q",
        color="reader:N",
        column="config:N",
    )


def chart_benchmarks(args: Args):
    # Check save dir exists or create
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Read results file
    with open(args.benchmark_file, "r") as read_in:
        all_results = json.load(read_in)

    # Load all datasets
    no_cluster_results = pd.DataFrame(all_results["no_cluster_results"])
    no_cluster_results["config"] = "No Cluster"

    local_cluster_results = pd.DataFrame(all_results["local_cluster_results"])
    local_cluster_results["config"] = "Local Cluster"

    distributed_cluster_results = pd.DataFrame(
        all_results["distributed_cluster_results"]
    )
    if len(distributed_cluster_results) > 0:
        distributed_cluster_results["config"] = "Distributed Cluster"

    # Generate charts
    no_cluster_chart = _generate_chart(no_cluster_results)
    no_cluster_chart.configure_header
    no_cluster_chart.save(str(args.save_dir / "no_cluster.png"))
    local_cluster_chart = _generate_chart(local_cluster_results)
    local_cluster_chart.save(str(args.save_dir / "local_cluster.png"))

    # Only generate distributed if it was ran
    if len(distributed_cluster_results) > 0:
        distributed_cluster_chart = _generate_chart(
            distributed_cluster_results
        )
        distributed_cluster_chart.save(str(args.save_dir / "distributed_cluster.png"))

    # Save version of chart with all combined
    all_results = pd.concat([
        no_cluster_results, local_cluster_results, distributed_cluster_results,
    ])
    all_results_chart = _generate_chart(all_results)
    all_results_chart.save(str(args.save_dir / "all.png"))


###############################################################################
# Runner


def main():
    args = Args()
    chart_benchmarks(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
