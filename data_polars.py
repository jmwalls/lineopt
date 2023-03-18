"""XXX
"""
from pathlib import Path

import numpy as np
import polars as pl


def list_files(path: Path, pattern="*.csv") -> list[Path]:
    """Get list of files matching some pattern from a specified directory.

    @param path: path to directory
    @returns list of files matching pattern in path dir
    """
    assert path.is_dir()
    return sorted([p for p in path.glob(pattern) if p.stat().st_size > 1])


def concat(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Only exists so we can switch between pandas/polars..."""
    return pl.concat(dfs)


def trace_dataframe(path: Path) -> pl.DataFrame:
    """Build dataframe from a trace data csv.

    @param path: path to csv
    @returns polars DataFrame containing trace data
    """
    return pl.read_csv(
        path,
        dtypes={
            "trace_id": pl.UInt64,
            "frame_index": pl.UInt32,
            "edge_id": pl.Utf8,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        },
    )


def traces_dataframe(path: Path) -> pl.DataFrame:
    """Create dataframe given all trace csv files...
    """
    return concat([trace_dataframe(p) for p in list_files(path, 'trace_*.csv')])


def keypoint_dataframe(path: Path) -> pl.DataFrame:
    """Build dataframe from a keypoint data csv.

    @param path: path to csv
    @returns polars DataFrame containing keypoint data
    """
    return pl.read_csv(
        path,
        dtypes={
            "trace_id": pl.UInt64,
            "frame_index": pl.UInt32,
            "type": pl.UInt32,
            "position": pl.UInt32,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        },
    )


def keypoints_dataframe(path: Path) -> pl.DataFrame:
    """Create dataframe given all trace csv files...
    """
    return concat([keypoint_dataframe(p) for p in list_files(path, 'keypoint_*.csv')])


def keypoint_range(keypoint_df: pl.DataFrame, trace_df: pl.DataFrame) -> pl.DataFrame:
    """Compute range to keypoints from trace poses.

    @param keypoint_df:
    @param trace_df:
    returns polars DataFrame with range column appended
    """
    return (
        keypoint_df.join(
            trace_df.select([pl.exclude("edge_id")]),
            on=["trace_id", "frame_index"],
            how="inner",
        )
        .with_columns(
            (
                np.sqrt(
                    (pl.col("x") - pl.col("x_right")) ** 2
                    + (pl.col("y") - pl.col("y_right")) ** 2
                    + (pl.col("z") - pl.col("z_right")) ** 2
                )
            ).alias("r")
        )
        .select([pl.all().exclude(["x_right", "y_right", "z_right"])])
    )


def edge_keypoint_data(
    keypoint_df: pl.DataFrame, trace_df: pl.DataFrame, edge_id
) -> pl.DataFrame:
    """Compute range to keypoints from trace poses.

    @param keypoint_df:
    @param trace_df:
    returns polars DataFrame with range column appended
    """
    return (
        keypoint_df.join(
            trace_df.select([pl.exclude("edge_id")]),
            on=["trace_id", "frame_index"],
            how="inner",
        )
        .with_columns(
            (
                np.sqrt(
                    (pl.col("x") - pl.col("x_right")) ** 2
                    + (pl.col("y") - pl.col("y_right")) ** 2
                    + (pl.col("z") - pl.col("z_right")) ** 2
                )
            ).alias("r")
        )
        .select([pl.all().exclude(["x_right", "y_right", "z_right"])])
    )
