"""XXX
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon


def _segment_to_records(p: Path) -> list[dict]:
    with open(str(p), "r", encoding="utf-8") as f:
        segment = json.load(f)
    return [
        {
            "segment_id": segment["id"]["s"],
            "line_id": line["id"]["s"],
            "line_index": i,
            "geometry": LineString(
                [p["matrix"][0] for p in line["polyline"]["waypoints"]]
            ),
        }
        for i, line in enumerate(segment["lane_lines"])
    ]


def ino_to_lines_gdf(path: Path) -> gpd.GeoDataFrame:
    """XXX"""
    return gpd.GeoDataFrame(
        pd.DataFrame.from_records(
            [r for p in path.glob("*.segment") for r in _segment_to_records(p)]
        ),
        crs="EPSG:4978",
    ).to_crs("EPSG:4326")


def lines_to_groups_gdf(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create polygon regions over a set of lines within a segment.

    We're assuming that lines are ordered and defined in the same direction so
    that we can create a polygon given the first and last line coordinates. This
    is not a perfect approximation of the concave hull around the lane lines.

    Note: this is NOT correct everywhere... at least one area in US T003 shows
    lane lines not ordered continuously left to right.
    """
    return gpd.GeoDataFrame(
        df.sort_values(by="line_index")
        .groupby("segment_id")
        .apply(
            lambda df: Polygon(
                [
                    *df.iloc[0]["geometry"].coords,
                    *df.iloc[-1]["geometry"].reverse().coords,
                ]
            )
        )
        .reset_index()
        .rename(columns={0: "geometry"}),
        crs=df.crs,
    )


def groups_edges_overlap(
    df_groups: gpd.GeoDataFrame, df_edges: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Create a table that provides a list of edge ids corresponding to each
    segment id. We expect for most segment ids, there will only be a single edge
    id.
    """
    return (
        df_groups.sjoin(df_edges, how="inner")
        .drop(columns=["index_right"])
        .reset_index(drop=True)
        .groupby("segment_id")["id"]
        .apply(list)
        .reset_index(name="edge_ids")
        .assign(edge_count=lambda df: df["edge_ids"].apply(len))
    )


def create_groups_edges_overlap(
    path_ino: Path, path_edges: Path
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Create a bunch of dataframes...

    @param path_ino: path to Ino segments
    @param path_edges: path to road graph edge gpkg
    @returns
        df_lines
        df_groups
        df_edges
        df_groups_edges
    """
    df_lines = ino_to_lines_gdf(path_ino)
    df_groups = lines_to_groups_gdf(df_lines)
    df_edges = gpd.GeoDataFrame.from_file(path_edges)
    df_groups_edges = groups_edges_overlap(df_groups, df_edges)
    return df_lines, df_groups, df_edges, df_groups_edges
