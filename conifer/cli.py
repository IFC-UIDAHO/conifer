"""Command-line interface for CONIFER.

A thin, honest wrapper so a fit can be run without writing Python — in the spirit of
pyeer's ``geteerinf``. Point it at three CSV matrices and it writes the estimate.

    conifer fit --counts counts.csv --area area.csv --aux aux.csv --out s_hat.csv

Contract (v0.1, deliberately simple):
  counts.csv : m rows x K columns  — DBH-class counts per area
  area.csv   : m rows x 1 column   — effective sampled area per area
  aux.csv    : m rows x p columns  — auxiliary covariates per area
Rows must align across the three files.
"""
from __future__ import annotations
import argparse
import sys

import numpy as np


def _load(path):
    return np.loadtxt(path, delimiter=",")


def _cmd_fit(args) -> int:
    counts = np.atleast_2d(_load(args.counts))
    area = np.ravel(_load(args.area)).astype(float)
    aux = np.atleast_2d(_load(args.aux))
    if not (counts.shape[0] == area.shape[0] == aux.shape[0]):
        sys.stderr.write(
            f"row mismatch: counts={counts.shape[0]}, area={area.shape[0]}, aux={aux.shape[0]}\n"
        )
        return 2
    from . import DiameterDistribution

    est = DiameterDistribution(seed=args.seed).fit(counts, area, aux)
    header = ",".join(f"class_{k+1}" for k in range(est.s_hat_.shape[1]))
    np.savetxt(args.out, est.s_hat_, delimiter=",", header=header, comments="")
    sys.stdout.write(
        f"CONIFER: fit {counts.shape[0]} areas x {counts.shape[1]} classes -> {args.out}\n"
        f"mean stem density by class: {np.round(est.s_hat_.mean(0), 2).tolist()}\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = argparse.ArgumentParser(prog="conifer", description="CONIFER — compositional forest small-area estimation.")
    p.add_argument("--version", action="version", version=f"conifer {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit DiameterDistribution and write the estimate")
    f.add_argument("--counts", required=True, help="m x K DBH-class counts CSV")
    f.add_argument("--area", required=True, help="m x 1 effective-area CSV")
    f.add_argument("--aux", required=True, help="m x p covariate CSV")
    f.add_argument("--out", default="conifer_s_hat.csv", help="output CSV for the estimate")
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=_cmd_fit)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
