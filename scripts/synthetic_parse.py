import subprocess

import numpy as np
import pandas as pd
import wbml.parser as parser

logs = (
    subprocess.check_output(
        "find aws_run_2022-05-06 | grep log_evaluate_out.txt",
        shell=True,
    )
    .strip()
    .splitlines(keepends=False)
)
results = {}
for log in logs:
    run = log.decode().split("/")[2:-1]
    for header in ["loglik", "elbo", "ar"]:
        p = parser.Parser(log)
        try:
            p.find_line(header + ":")
        except RuntimeError:
            continue
        for task, desc in [
            ("int", "interpolation in training range"),
            ("int-beyond", "interpolation beyond training range"),
            ("extr", "extrapolation beyond training range"),
        ]:
            p.find_line(desc)
            p.next_line()
            res = p.parse(
                parser.SkipUntil("|"),
                parser.Whitespace(),
                parser.Literal("Loglik (V):"),
                parser.Whitespace(),
                parser.Float(),
                parser.Whitespace(),
                parser.Literal("+-"),
                parser.Whitespace(),
                parser.Float(),
            )
            results[tuple(run) + (header, task, "loglik")] = res
            # Try to read the KL divergence.
            try:
                p.next_line()
                res = p.parse(
                    parser.SkipUntil("|"),
                    parser.Whitespace(),
                    parser.Literal("KL (full):"),
                    parser.Whitespace(),
                    parser.Float(),
                    parser.Whitespace(),
                    parser.Literal("+-"),
                    parser.Whitespace(),
                    parser.Float(),
                )
                results[tuple(run) + (header, task, "kl")] = res
            except RuntimeError:
                pass

# The keys have the followings structure:
#   `(data, x{1,2}_y{1,2}, model, loss, header, task, {loglik,kl})`.

p = parser.Parser("server_good/_experiments/synthetic_extra/log_out.txt")
for data in ["eq", "matern", "weakly-periodic", "sawtooth", "mixture"]:
    for dim_x in [1, 2]:
        for dim_y in [1, 2]:
            run_trivial = (data, f"x{dim_x}_y{dim_y}", "trivial", "loglik")
            run_diag = (data, f"x{dim_x}_y{dim_y}", "diag", "loglik")
            p.find_line(f"{data}-{dim_x}-{dim_y}:")
            for task, desc in [
                ("int", "interpolation in training range"),
                ("int-beyond", "interpolation beyond training range"),
                ("extr", "extrapolation beyond training range"),
            ]:
                p.find_line(desc)
                p.next_line()
                res = p.parse(
                    parser.Whitespace(),
                    parser.Literal("Logpdf (trivial):"),
                    parser.Whitespace(),
                    parser.Float(),
                )
                results[run_trivial + ("loglik", task, "loglik")] = (res, 0)
                # Try to read the KL divergences.
                try:
                    p.next_line()
                    res = p.parse(
                        parser.Whitespace(),
                        parser.Literal("Logpdf (diag):"),
                        parser.Whitespace(),
                        parser.Float(),
                        parser.Whitespace(),
                        parser.Literal("+-"),
                        parser.Whitespace(),
                        parser.Float(),
                    )
                    results[run_diag + ("loglik", task, "loglik")] = res
                    p.next_line()
                    res = p.parse(
                        parser.Whitespace(),
                        parser.Literal("KL (diag):"),
                        parser.Whitespace(),
                        parser.Float(),
                        parser.Whitespace(),
                        parser.Literal("+-"),
                        parser.Whitespace(),
                        parser.Float(),
                    )
                    results[run_diag + ("loglik", task, "kl")] = res
                    p.next_line()
                    res = p.parse(
                        parser.Whitespace(),
                        parser.Literal("KL (trivial):"),
                        parser.Whitespace(),
                        parser.Float(),
                        parser.Whitespace(),
                        parser.Literal("+-"),
                        parser.Whitespace(),
                        parser.Float(),
                    )
                    results[run_trivial + ("loglik", task, "kl")] = res
                except RuntimeError:
                    pass

rows = [
    {"name": "\\textit{Diagonal GP}", "key": ("diag", "loglik", "loglik")},
    {"name": "\\textit{Trivial}", "key": ("trivial", "loglik", "loglik")},
    {"name": "CNP", "key": ("cnp", "loglik", "loglik")},
    {"name": "\\textbf{CNP (AR)}", "key": ("cnp", "loglik", "ar")},
    {"name": "ACNP", "key": ("acnp", "loglik", "loglik")},
    {"name": "\\textbf{ACNP (AR)}", "key": ("acnp", "loglik", "ar")},
    {"name": "ConvCNP", "key": ("convcnp", "unet", "loglik", "loglik")},
    {"name": "\\textbf{ConvCNP (AR)}", "key": ("convcnp", "unet", "loglik", "ar")},
    {"name": "GNP", "key": ("gnp", "loglik", "loglik")},
    {"name": "AGNP", "key": ("agnp", "loglik", "loglik")},
    {"name": "ConvGNP", "key": ("convgnp", "unet", "loglik", "loglik")},
    {"name": "FullConvGNP", "key": ("fullconvgnp", "unet", "loglik", "loglik")},
    {"name": "NP (ELBO)", "key": ("np", "elbo", "elbo")},
    {"name": "NP (ELBO, ML)", "key": ("np", "elbo", "loglik")},
    {"name": "NP (ML)", "key": ("np", "loglik", "loglik")},
    {"name": "ANP (ELBO)", "key": ("anp", "elbo", "elbo")},
    {"name": "ANP (ELBO, ML)", "key": ("anp", "elbo", "loglik")},
    {"name": "ANP (ML)", "key": ("anp", "loglik", "loglik")},
    {"name": "ConvNP (ELBO)", "key": ("convnp", "unet", "elbo", "elbo")},
    {"name": "ConvNP (EL., ML)", "key": ("convnp", "unet", "elbo", "loglik")},
    {"name": "ConvNP (ML)", "key": ("convnp", "unet", "loglik", "loglik")},
]

# Build data frame.
entries = []
for data in ["eq", "matern", "weakly-periodic", "sawtooth", "mixture"]:
    for dim_x in [1, 2]:
        for dim_y in [1, 2]:
            for row in rows:
                # Build the entry for kind `loglik`.
                entry = {
                    "kind": "loglik",
                    "data": data,
                    "dim_x": dim_x,
                    "dim_y": dim_y,
                    "name": row["name"],
                }
                for task in ["int", "int-beyond", "extr"]:
                    try:
                        val, err = results[
                            (data, f"x{dim_x}_y{dim_y}", *row["key"], task, "loglik")
                        ]
                        entry[task] = val
                        entry[task + "-err"] = err
                    except KeyError:
                        continue
                entries.append(entry)
                # Build the entry for kind `kl`.
                entry = {
                    "kind": "kl",
                    "data": data,
                    "dim_x": dim_x,
                    "dim_y": dim_y,
                    "name": row["name"],
                }
                for task in ["int", "int-beyond", "extr"]:
                    try:
                        val, err = results[
                            (data, f"x{dim_x}_y{dim_y}", *row["key"], task, "kl")
                        ]
                        entry[task] = val
                        entry[task + "-err"] = err
                    except KeyError:
                        continue
                entries.append(entry)
df = pd.DataFrame(entries).set_index(["kind", "data", "dim_x", "dim_y", "name"])


def n_best(df, col, col_err, *, ascending):
    df = df.sort_values(col, ascending=ascending)
    best_indices = set()
    val0, err0 = df[[col, col_err]].iloc[0]
    i = 0
    while True:
        best_indices.add(df.index[i])

        # Compare with the next.
        val, err = df[[col, col_err]].iloc[i + 1]
        diff = abs(val0 - val)
        diff_err = np.sqrt(err0**2 + err**2)

        if diff > diff_err:
            # Significantly better.
            return best_indices
        else:
            # Not significantly better. Try the next.
            i += 1


def format_number(value, error, *, bold=False, possibly_negative=True):
    if np.isnan(value):
        return ""
    if np.abs(value) > 10 or np.abs(error) > 10:
        return "F"
    if value >= 0 and possibly_negative:
        sign_spacer = "\\hphantom{-}"
    else:
        sign_spacer = ""
    if bold:
        bold_start, bold_end = "\\mathbf{", "}"
    else:
        bold_start, bold_end = "", ""
    return f"${sign_spacer}{bold_start}{value:.2f}{bold_end} {{ \\pm \\small {error:.2f} }}$"


def format_table(title1, title2, df, cols, *, possibly_negative, ascending=True):
    for col in cols:
        col["best"] = n_best(df, col["value"], col["error"], ascending=ascending)

    res = f"\\begin{{tabular}}[t]{{l{'c' * len(cols)}}} \n"
    res += "\\toprule \n"
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
    for name, row in df.iterrows():
        res += name
        for col in cols:
            res += " & " + format_number(
                row[col["value"]],
                row[col["error"]],
                bold=name in col["best"],
                possibly_negative=possibly_negative,
            )
        res += " \\\\ \n"
    res += "\\bottomrule \\\\ \n"
    res += "\\end{tabular}"
    return res


def format_four_tables(title, df, *, possibly_negative, ascending=True):
    res = "\\centerline{ \n"
    res += "\\small\\scshape \n"
    res += "\\begin{tabular}{cc} \n"
    columns = [
        {"name": "Interp.", "value": "int", "error": "int-err"},
        {"name": "I.\\ Beyond", "value": "int-beyond", "error": "int-beyond-err"},
        {"name": "Extrap.", "value": "extr", "error": "extr-err"},
    ]
    res += (
        format_table(
            title,
            "$d_x=1$, $d_y=1$",
            df.xs(1, level="dim_x")
            .xs(1, level="dim_y")
            .sort_values("int", ascending=ascending),
            columns,
            possibly_negative=possibly_negative,
            ascending=ascending,
        )
        + "\n"
    )
    res += "&\n"
    res += (
        format_table(
            title,
            "$d_x=2$, $d_y=1$",
            df.xs(2, level="dim_x")
            .xs(1, level="dim_y")
            .sort_values("int", ascending=ascending),
            columns,
            possibly_negative=possibly_negative,
            ascending=ascending,
        )
        + " \\\\ \n"
    )
    res += (
        format_table(
            title,
            "$d_x=1$, $d_y=2$",
            df.xs(1, level="dim_x")
            .xs(2, level="dim_y")
            .sort_values("int", ascending=ascending),
            columns,
            possibly_negative=possibly_negative,
            ascending=ascending,
        )
        + "\n"
    )
    res += "& \n"
    res += (
        format_table(
            title,
            "$d_x=2$, $d_y=2$",
            df.xs(2, level="dim_x")
            .xs(2, level="dim_y")
            .sort_values("int", ascending=ascending),
            columns,
            possibly_negative=possibly_negative,
            ascending=ascending,
        )
        + "\n"
    )
    res += "\\end{tabular}\n"
    res += "}\n"
    return res


caption = """\\caption{{
    {kind_inline} of various models on the synthetic {title_inline} data set.
    Shows interpolation performance on the range $[-2, 2]$ in which the models were trained (interp.),
    interpolation performance on the range $[2, 6]$ which the models have never seen before (i. beyond),
    and extrapolation performance from $[-2, 2]$ into $[2, 6]$.
    Best numbers are boldfaced,
    and autoregressive models are also boldfaced for clarity.
    Numbers are omitted for models which failed or could not be run.
    Diagonal GP refers to the prediction of the ground-truth GP without correlations.
    Trivial refers to predicting the empirical mean and variance of the test data.
}}"""

for data, kind, title, title_inline in [
    ("eq", "kl", "EQ", "EQ"),
    ("matern", "kl", "Mat\\'ern--$\\tfrac52$", "Mat\\'ern--$\\tfrac52$"),
    ("weakly-periodic", "kl", "Weakly Periodic", "weakly periodic"),
    ("sawtooth", "loglik", "Sawtooth", "sawtooth"),
    ("mixture", "loglik", "Mixture", "mixture"),
]:
    if kind == "kl":
        ascending = True
        possibly_negative = False
        kind_inline = "Kullback--Leibler divergences"
    else:
        ascending = False
        possibly_negative = True
        kind_inline = "Log-likelihoods"
    print("\\begin{table}[h]")
    print(caption.format(kind_inline=kind_inline, title_inline=title_inline))
    print(f"\\label{{tab:synthetic-{data}}}")
    print(
        format_four_tables(
            title,
            df.xs(kind).xs(data),
            ascending=ascending,
            possibly_negative=possibly_negative,
        )
    )
    print("\\end{table}")


columns = [
    {"name": "Int. (1D)", "value": ("int", 1), "error": ("int-err", 1)},
    {
        "name": "Int. Bd (1D)",
        "value": ("int-beyond", 1),
        "error": ("int-beyond-err", 1),
    },
    {"name": "Extr. (1D)", "value": ("extr", 1), "error": ("extr-err", 1)},
    {"name": "Int. (2D)", "value": ("int", 2), "error": ("int-err", 2)},
    {
        "name": "Int. Bd (2D)",
        "value": ("int-beyond", 2),
        "error": ("int-beyond-err", 2),
    },
    {"name": "Extr. (2D)", "value": ("extr", 2), "error": ("extr-err", 2)},
]


def agg(df):
    res = {}
    for col in ["int", "int-beyond", "extr"]:
        res[col] = df[col].mean()
        res[col + "-err"] = (df[col + "-err"] ** 2).sum() ** 0.5
    return pd.Series(res)


# Compute averages over the Gaussian and non-Gaussian tasks.
df_gaussian = (
    df.iloc[
        np.array(df.reset_index("data").data.isin(["eq", "matern", "weakly-periodic"]))
    ]
    .xs("kl")
    .groupby(["name", "dim_x"])
    .apply(agg)
    .unstack("dim_x")
    .sort_values(("int", 1), ascending=True)
)
df_gaussian.columns = list(df_gaussian.columns)
df_nongaussian = (
    df.iloc[np.array(df.reset_index("data").data.isin(["sawtooth", "mixture"]))]
    .xs("loglik")
    .groupby(["name", "dim_x"])
    .apply(agg)
    .unstack("dim_x")
    .sort_values(("int", 1), ascending=False)
)
df_nongaussian.columns = list(df_nongaussian.columns)
df_nongaussian = df_nongaussian.iloc[
    ["diagonal" not in x.lower() for x in df_nongaussian.index]
]

print()
print("\\begin{table}[t]")
print("\\small")
print("\\scshape")
print(
    "\\caption{"
    "Average Kullback--Leibler divergences in the synthetic Gaussian "
    "experiments and average log-likelihoods in the synthetic non-Gaussian experiments."
    "}"
)
print("\\centerline{")
print(
    format_table(
        "Gaussian",
        "",
        df_gaussian,
        columns,
        possibly_negative=True,
        ascending=True,
    )
)
print("}")
print("\\centerline{")
print(
    format_table(
        "Non-Gaussian",
        "",
        df_nongaussian,
        columns,
        possibly_negative=True,
        ascending=False,
    )
)
print("}")
print("\\end{table}")
