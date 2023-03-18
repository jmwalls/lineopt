"""XXX
"""
import argparse
from pathlib import Path
from time import time

import geopandas as gpd
import polars as pl
from shapely.geometry import Point

import data_polars as data


def timer(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        res = func(*args, **kwargs)
        print(f"{func.__name__:20s} wall time: {time() - t1:.6f} [s]")
        return res

    return wrapper


def _xyz_to_point(df):
    return df.apply(lambda r: Point(r["x"], r["y"], r["z"]), axis=1)


@timer
def _load_traces(path: Path) -> pl.DataFrame:
    print("loading trace data...")
    return data.traces_dataframe(path)


@timer
def _load_keypoints(path: Path, df_traces: pl.DataFrame) -> pl.DataFrame:
    print("loading keypoint data...")
    df_keypoints = data.keypoints_dataframe(path)
    return data.keypoint_range(df_keypoints, df_traces)


@timer
def _df_to_parquet(df: gpd.GeoDataFrame, path: Path) -> None:
    print(f"saving {path} dataframe")
    df.to_parquet(path)


@timer
def _to_gdf(df: pl.DataFrame) -> gpd.GeoDataFrame:
    print("converting to gdf...")
    return gpd.GeoDataFrame(
        df
        .to_pandas()
        .assign(geometry=_xyz_to_point),  # this is slow...
        crs="EPSG:4978",
    ).to_crs("EPSG:4326")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir", type=Path, required=True, help="Path to directory of trace csvs"
    )
    args = parser.parse_args()

    df_traces = _load_traces(args.trace_dir)
    df_keypoints = _load_keypoints(args.trace_dir, df_traces)

    gdf_traces = _to_gdf(df_traces)
    _df_to_parquet(gdf_traces, Path("foo_traces.parquet"))

    gdf_keypoints = _to_gdf(df_keypoints)
    _df_to_parquet(gdf_keypoints, Path("foo_keypoints.parquet"))


if __name__ == "__main__":
    main()
