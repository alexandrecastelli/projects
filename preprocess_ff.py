import polars as pl
import sys


def preprocess(parquet_file: str, output_file: str = "ff_summary.parquet") -> None:
    """Preprocess raw financial flow data.

    Parameters
    ----------
    parquet_file : str
        Path to the raw `ff.parquet` file.
    output_file : str
        Destination for the aggregated output.

    The function reads the Parquet file lazily using Polars to avoid loading
    the entire dataset in memory. It parses the ``valor`` column into a float
    and then aggregates the information by ``instituicao``,
    ``tipo_movimentacao`` and ``tipo_contraparte``.  The result is written to a
    Parquet file that can be quickly loaded by the dashboard.
    """

    df = (
        pl.scan_parquet(parquet_file)
        .with_columns(pl.col("valor").str.replace_all(",", "").cast(pl.Float64))
    )

    summary = (
        df.groupby(["instituicao", "tipo_movimentacao", "tipo_contraparte"])
        .agg(
            valor_total=pl.col("valor").sum(),
            quantidade_total=pl.col("quantidade").sum(),
            registros=pl.count(),
        )
    )

    summary.collect().write_parquet(output_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_ff.py <ff.parquet> [output.parquet]")
        sys.exit(1)

    parquet_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ff_summary.parquet"
    preprocess(parquet_path, out_path)
