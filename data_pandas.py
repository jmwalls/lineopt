"""XXX
"""
from pathlib import Path

import numpy as np
import pandas as pd


def list_files(path: Path, pattern="*.csv") -> list[Path]:
    """Get list of files matching some pattern from a specified directory.

    @param path: path to directory
    @returns list of files matching pattern in path dir
    """
    assert path.is_dir()
    return sorted([p for p in path.glob(pattern) if p.stat().st_size > 1])


def concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Only exists so we can switch between pandas/polars..."""
    return pd.concat(dfs)


def trace_dataframe(path: Path) -> pd.DataFrame:
    """Build dataframe from a trace data csv.

    @param path: path to csv
    @returns pandas DataFrame containing trace data
    """
    return pd.read_csv(
        path,
        dtype={
            "trace_id": np.uint64,
            "frame_index": np.uint32,
            "edge_id": str,
            "x": np.float64,
            "y": np.float64,
            "z": np.float64,
        },
    )


def keypoint_dataframe(path: Path) -> pd.DataFrame:
    """Build dataframe from a keypoint data csv.

    @param path: path to csv
    @returns pandas DataFrame containing keypoint data
    """
    return pd.read_csv(
        path,
        dtype={
            "trace_id": np.uint64,
            "frame_index": np.uint32,
            "type": np.uint32,
            "position": np.uint32,
            "x": np.float64,
            "y": np.float64,
            "z": np.float64,
        },
    )


def keypoint_range(keypoint_df: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    """Compute range to keypoints from trace poses.

    @param keypoint_df:
    @param trace_df:
    returns pandas DataFrame with range column appended
    """
    return (
        pd.merge(
            keypoint_df,
            trace_df,
            on=["trace_id", "frame_index"],
            suffixes=("", "_trace"),
        )
        .assign(
            r=lambda df: np.sqrt(
                (df["x"] - df["x_trace"]) ** 2
                + (df["y"] - df["y_trace"]) ** 2
                + (df["z"] - df["z_trace"]) ** 2
            )
        )
        .drop(columns=["edge_id", "x_trace", "y_trace", "z_trace"])
    )
