
import argparse
import functools
import logging
import os
import sys
import time
from typing import List, Sequence, Union

from termcolor import colored


def _sampling_scope(sampling_by_batch: bool) -> str:
    return "batch-wise" if sampling_by_batch else "graph-wise"


def build_strategy_parts(sampling_flag: bool,
                         sampling_by_batch: bool,
                         if_crossover: bool,
                         if_mutation: bool,
                         if_supervised: bool) -> List[str]:
    if not sampling_flag:
        return ["full-loader"]

    if not if_crossover:
        return ["sampling-only", _sampling_scope(sampling_by_batch)]

    return ["sampling-crossover",
            _sampling_scope(sampling_by_batch),
            f"mutation-{'on' if if_mutation else 'off'}",
            f"supervised-{'on' if if_supervised else 'off'}"]


def build_leaf_name(task_type: str,
                    epochs: int,
                    lr: float,
                    batch_size: int,
                    num_classes: int,
                    n_layers: int,
                    heads: int,
                    sampling_flag: bool,
                    if_crossover: bool,
                    if_mutation: bool,
                    if_supervised: bool,
                    flag_for_supervised: str,
                    flag_for_unsupervised: str) -> str:
    parts = [f"T-{task_type}",
             f"E-{epochs}",
             f"lr-{lr}",
             f"BS-{batch_size}",
             f"NC-{num_classes}",
             f"NL-{n_layers}",
             f"H-{heads}"]

    if sampling_flag and not if_crossover:
        parts.extend([
            f"Mut-{'on' if if_mutation else 'off'}",
            f"Sup-{'on' if if_supervised else 'off'}",
        ])

    if sampling_flag and if_crossover:
        if if_supervised:
            parts.append(f"SU-{flag_for_supervised}")
        else:
            parts.append(f"UnSU-{flag_for_unsupervised}")

    return "_".join(parts)


def build_run_path(base_dir: str,
                   dataset: str,
                   model: str,
                   task_type: str,
                   epochs: int,
                   lr: float,
                   batch_size: int,
                   num_classes: int,
                   n_layers: int,
                   heads: int,
                   sampling_flag: bool,
                   sampling_by_batch: bool,
                   if_crossover: bool,
                   if_mutation: bool,
                   if_supervised: bool,
                   flag_for_supervised: str,
                   flag_for_unsupervised: str,
                   run_timestamp: str | None = None) -> str:
    strategy_parts = build_strategy_parts(sampling_flag=sampling_flag,
                                          sampling_by_batch=sampling_by_batch,
                                          if_crossover=if_crossover,
                                          if_mutation=if_mutation,
                                          if_supervised=if_supervised)
    leaf = build_leaf_name(task_type=task_type,
                           epochs=epochs,
                           lr=lr,
                           batch_size=batch_size,
                           num_classes=num_classes,
                           n_layers=n_layers,
                           heads=heads,
                           sampling_flag=sampling_flag,
                           if_crossover=if_crossover,
                           if_mutation=if_mutation,
                           if_supervised=if_supervised,
                           flag_for_supervised=flag_for_supervised,
                           flag_for_unsupervised=flag_for_unsupervised)
    run_timestamp = run_timestamp or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return os.path.join(base_dir,
                        dataset,
                        model,
                        *strategy_parts,
                        leaf,
                        f"run-{run_timestamp}")


def build_visualize_run_path(base_dir: str,
                             dataset: str,
                             model: str,
                             task_type: str,
                             run_timestamp: str | None = None) -> str:
    run_timestamp = run_timestamp or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return os.path.join(base_dir,
                        dataset,
                        model,
                        "visualize",
                        f"T-{task_type}",
                        f"run-{run_timestamp}")


def build_train_leaf_name(lr: float | None = None,
                          batch_size: int | None = None,
                          weight_decay: float | None = None,
                          lamda: float | None = None,
                          epochs: int | None = None,
                          seed: int | None = None) -> str:
    """
    Build a compact experiment leaf name for current SFEARNet train.py args.
    Only provided fields are encoded, so this stays compatible with scripts
    that do not expose all knobs.
    """
    parts: List[str] = []
    if lr is not None:
        parts.append(f"lr-{lr}")
    if batch_size is not None:
        parts.append(f"bs-{batch_size}")
    if weight_decay is not None:
        parts.append(f"wd-{weight_decay}")
    if lamda is not None:
        parts.append(f"lam-{lamda}")
    if epochs is not None:
        parts.append(f"ep-{epochs}")
    if seed is not None:
        parts.append(f"seed-{seed}")
    return "_".join(parts) if parts else "default"


def build_train_run_path(base_dir: str,
                         dataset: str,
                         model: str,
                         lr: float | None = None,
                         batch_size: int | None = None,
                         weight_decay: float | None = None,
                         lamda: float | None = None,
                         epochs: int | None = None,
                         seed: int | None = None,
                         run_timestamp: str | None = None) -> str:
    """
    Build run path as:
    logs/<dataset>/<model>/<hyperparam-leaf>/run_<timestamp>
    """
    leaf = build_train_leaf_name(lr=lr,
                                 batch_size=batch_size,
                                 weight_decay=weight_decay,
                                 lamda=lamda,
                                 epochs=epochs,
                                 seed=seed)
    run_timestamp = run_timestamp or time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    return os.path.join(base_dir,
                        str(dataset),
                        str(model),
                        leaf,
                        f"run_{run_timestamp}")


def prepare_train_log_dirs(args,
                           base_dir: str = "logs",
                           run_timestamp: str | None = None) -> tuple[str, str]:
    """
    Compatibility helper for current train.py.
    It reads only available attributes from args and returns:
    (run_dir, run_timestamp)
    """
    run_timestamp = run_timestamp or time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    dataset = getattr(args, "data", "unknown_data")
    model = getattr(args, "model", "unknown_model")

    batch_size = getattr(args, "train_batchsize", None)
    if batch_size is None:
        batch_size = getattr(args, "batch_size", None)

    run_dir = build_train_run_path(base_dir=base_dir,
                                   dataset=dataset,
                                   model=model,
                                   lr=getattr(args, "lr", None),
                                   batch_size=batch_size,
                                   weight_decay=getattr(args, "weight_decay", None),
                                   lamda=getattr(args, "lamda", None),
                                   epochs=getattr(args, "num_epochs", None),
                                   seed=getattr(args, "seed", None),
                                   run_timestamp=run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_timestamp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--sampling-flag", action="store_true", default=False)
    parser.add_argument("--sampling-by-batch", action="store_true", default=False)
    parser.add_argument("--if-crossover", action="store_true", default=False)
    parser.add_argument("--if-mutation", action="store_true", default=False)
    parser.add_argument("--if-supervised", action="store_true", default=False)
    parser.add_argument("--flag-for-supervised", default="RMSE")
    parser.add_argument("--flag-for-unsupervised", default="TV")
    parser.add_argument("--run-timestamp")
    parser.add_argument("--visualize", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.visualize:
        print(build_visualize_run_path(base_dir=args.base_dir,
                                       dataset=args.dataset,
                                       model=args.model,
                                       task_type=args.task_type,
                                       run_timestamp=args.run_timestamp))
    else:
        print(build_run_path(base_dir=args.base_dir,
                             dataset=args.dataset,
                             model=args.model,
                             task_type=args.task_type,
                             epochs=args.epochs,
                             lr=args.lr,
                             batch_size=args.batch_size,
                             num_classes=args.num_classes,
                             n_layers=args.n_layers,
                             heads=args.heads,
                             sampling_flag=args.sampling_flag,
                             sampling_by_batch=args.sampling_by_batch,
                             if_crossover=args.if_crossover,
                             if_mutation=args.if_mutation,
                             if_supervised=args.if_supervised,
                             flag_for_supervised=args.flag_for_supervised,
                             flag_for_unsupervised=args.flag_for_unsupervised,
                             run_timestamp=args.run_timestamp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


@functools.lru_cache()
def create_loggerGt(dirLog: Union[str, bytes, os.PathLike],
                    t: str,
                    dist_rank: int = 0,
                    name: str = "training"):
    """

    :param dirLog:
    :param t:
    :param dist_rank:
    :param name:
    :return:
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + \
                ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers

    file_handler = logging.FileHandler(os.path.join(dirLog, f'{name}_{t}_rank-{dist_rank}.log'),
                                       mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
