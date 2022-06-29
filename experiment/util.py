import subprocess
from itertools import product

import lab as B
import numpy as np
import pandas as pd
import scipy.special as sps
import torch
import wbml.parser as parser

__all__ = [
    "median_and_err",
    "load",
    "with_err",
    "parse_evaluation_logs",
    "build_results_df",
    "format_table",
]


def median_and_err(xs, alpha=0.05):
    """Compute the median with an error which construct a central
    `100 * (1 - alpha)`%-confidence region using the order statistics.

    Source:
        https://stats.stackexchange.com/a/160148

    Args:
        xs (vector): Data.
        alpha (scalar): Desired confidence level.

    Returns:
        scalar: Median.
        scalar: Error which constructs a central `100 * (1 - alpha)`%-confidence region.
    """
    xs = B.to_numpy(B.flatten(xs))
    xs = xs[~np.isnan(xs)]
    xs = np.sort(xs)
    n = len(xs)

    def compute_log_term(i):
        log_binom = sps.loggamma(n + 1) - sps.loggamma(i + 1)
        log_binom = log_binom - sps.loggamma(n - i + 1) - n * np.log(2)
        return log_binom

    lower, upper = n // 2, n // 2
    cum_sum = compute_log_term(n // 2)

    while cum_sum < np.log(1 - alpha):
        lower -= 1
        cum_sum = sps.logsumexp([cum_sum, compute_log_term(lower)])
        if cum_sum < np.log(1 - alpha):
            upper += 1
            cum_sum = sps.logsumexp([cum_sum, compute_log_term(upper)])

    median = np.median(xs)
    error = max(median - xs[lower], median - xs[upper])
    return np.median(xs), error


def with_err(vals, err=None, and_lower=False, and_upper=False):
    """Print the mean value of a list of values with error."""
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    if err is None:
        err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    res = f"{mean:10.5f} +- {err:10.5f}"
    if and_lower:
        res += f" ({mean - err:10.5f})"
    if and_upper:
        res += f" ({mean + err:10.5f})"
    return res


def load(last=False, device="cpu", **kw_args):
    """Load an existing model."""
    from train import main

    exp = main(**kw_args, load=True)
    wd = exp["wd"]
    f = "model-last.torch" if last else "model-best.torch"
    exp["model"].load_state_dict(torch.load(wd.file(f), map_location=device)["weights"])
    return exp


def parse_evaluation_logs(path, tasks):
    """Parse all evaluation logs in a directory.

    Args:
        path (str): Path to directory.
        tasks (list[tuple[str, str]]): A list of keys of tasks (`task` below in the
            key of the resulting dictionary) and the descriptions as those occur in the
            log.

    Returns:
        dict: A dictionary with keys that have the following structure:
           `(*dirs, header, task, {loglik,kl})`.
    """
    logs = (
        subprocess.check_output(
            f"find {path} | grep log_evaluate_out.txt",
            shell=True,
        )
        .strip()
        .splitlines(keepends=False)
    )
    results = {}
    for log in logs:
        run = log.decode().split("/")[:-1]
        for header in ["loglik", "elbo", "ar"]:
            p = parser.Parser(log)
            # Skip the configuration.
            p.find_line("number of parameters")
            try:
                p.find_line(header + ":")
            except RuntimeError:
                continue
            for task, desc in tasks:
                p.find_line(desc)
                p.next_line()
                res = p.parse("s| `Loglik (V):` f `+-` f")
                results[tuple(run) + (header, task, "loglik")] = res
                # Try to read the KL divergence.
                try:
                    p.next_line()
                    res = p.parse("s| `KL (full):` f `+-` f")
                    results[tuple(run) + (header, task, "kl")] = res
                except RuntimeError:
                    pass
    return results


def build_results_df(results, spec, spec_to_key, tasks, rows):
    """Build a data frame with results.

    Args:
        results (dict): Results dictionary.
        spec (list[tuple[str, list]]): Keys and associated possible values to loop over.
        spec_to_key (function): A function that takes in a value from each key in `spec`
            and gives back a prefix to the `tuple`-index in `results`.
        tasks (tuple[str]): Tasks
        rows (dict): Rows of the results table.

    Returns:
        :class:`pandas.DataFrame`: Data frame.
    """
    entries = []
    if spec:
        keys, all_values = zip(*spec)
    else:
        keys, all_values = (), ()

    for values in product(*all_values):
        for row in rows:
            for kind in ["loglik", "kl"]:
                entry = {k: v for k, v in zip(keys, values)}
                entry["name"] = row["name"]
                entry["kind"] = kind
                for task in tasks:
                    try:
                        val, err = results[
                            (*spec_to_key(*values), *row["key"], task, kind)
                        ]
                        entry[task] = val
                        entry[task + "-err"] = err
                    except KeyError:
                        continue
                entries.append(entry)

    return pd.DataFrame(entries)


def _n_best(df, col, col_err, *, ascending):
    df = df.sort_values(col, ascending=ascending)
    best_indices = set()
    val0, err0 = df[[col, col_err]].iloc[0]
    i = 0
    while True:
        best_indices.add(df.index[i])

        # Compare with the next.
        val, err = df[[col, col_err]].iloc[i + 1]
        diff = abs(val0 - val)
        diff_err = B.sqrt(err0**2 + err**2)

        if diff > diff_err:
            # Significantly better.
            return best_indices
        else:
            # Not significantly better. Try the next.
            i += 1


def _format_number(value, error, *, bold=False, possibly_negative=True):
    if B.isnan(value):
        return ""
    failed = B.abs(value) > 10 or B.abs(error) > 10
    minus = "\\scalebox{0.7}[1]{$-$}"
    if (value >= 0 or failed) and possibly_negative:
        sign_spacer = f"\\hphantom{{{minus}}}"
    else:
        sign_spacer = ""
    if bold:
        bold_start, bold_end = "\\mathbf{", "}"
    else:
        bold_start, bold_end = "", ""
    if failed:
        return (
            f"${sign_spacer}\\hphantom{{0.0}}$F"
            f"$\\,\\hphantom{{ \\scriptstyle \\pm  0.00 }}$"
        )
    else:
        minus = "" if value >= 0 else minus
        return (
            f"${sign_spacer}{bold_start}{minus}{abs(value):.2f}{bold_end}"
            f"\\,{{ \\scriptstyle \\pm  {error:.2f} }}$"
        )


def format_table(
    title1,
    title2,
    df,
    cols,
    *,
    possibly_negative,
    ascending=True,
    skip=(),
):
    """Format a table.

    Args:
        title1 (str): First line of the title.
        title2 (str): Second line of the title. Can be left empty.
        df (:class:`pandas.DataFrame`): Contents of the table.
        cols (list[dict]): Columns of the table.
        possibly_negative (bool): Can the numbers be negative?
        ascending (bool, optional): Is lower better? Defaults to `True`.
        skip (iterable): Skip rows where any element of `skip` is a substring.

    Returns:
        str: Table.
    """
    for col in cols:
        col["best"] = _n_best(df, col["value"], col["error"], ascending=ascending)

    res = f"\\begin{{tabular}}[t]{{l{'c' * len(cols)}}} \n"
    res += "\\toprule \n"

    # Print title.
    res += title1
    for col in cols:
        if title2:
            res += " & \\multirow{2}{*}{" + col["name"] + "}"
        else:
            res += " & " + col["name"]
    if title2:
        res += " \\\\ \n"
        res += title2 + " \\\\ \\midrule \n"
    else:
        res += " \\\\ \\midrule \n"

    # Print rows.
    for name, row in df.iterrows():
        # Check if the row needs to be skipped.
        if any(s.lower() in name.lower() for s in skip):
            continue
        res += name
        for col in cols:
            res += " & " + _format_number(
                row[col["value"]],
                row[col["error"]],
                bold=name in col["best"],
                possibly_negative=possibly_negative,
            )
        res += " \\\\ \n"

    res += "\\bottomrule \\\\ \n"
    res += "\\end{tabular} \n"

    return res
