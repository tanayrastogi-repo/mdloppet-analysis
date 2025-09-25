# components.py
from __future__ import annotations

import re
from datetime import timedelta
from typing import Optional, Iterable

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go


# ---------------------------- Utils ---------------------------- #


def parse_time_to_timedelta(s: str) -> timedelta:
    """Parse 'mm:ss' or 'h:mm:ss' to timedelta."""
    s = s.strip().replace("：", ":")
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = map(int, parts)
        return timedelta(minutes=m, seconds=sec)
    if len(parts) == 3:
        h, m, sec = map(int, parts)
        return timedelta(hours=h, minutes=m, seconds=sec)
    raise ValueError(f"Unrecognized time: {s}")


def split_class(value: str) -> tuple[str, str]:
    """Split class like 'M23-34' into (gender, agegrp)."""
    if not isinstance(value, str) or not value:
        return "", ""
    v = value.replace("\u2013", "-").replace(" ", "")
    m = re.match(r"^([A-Za-zÅÄÖåäö]{1,3})(\d{1,2}(?:-\d{1,2}|\+))?$", v)
    if m:
        g = m.group(1).upper()
        a = m.group(2) or ""
        return g, a
    # Fallbacks
    g = (re.match(r"([A-Za-zÅÄÖåäö]{1,3})", v) or [None, ""])[1].upper()
    a = (re.search(r"(\d{1,2}(?:-\d{1,2}|\+))", v) or [None, ""])[1]
    return g, a


def normalize_share(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-independent shares over full table sum (robust to zeros)."""
    out = df.copy()
    denom = float(out.sum(numeric_only=True).sum())
    return out / denom if denom > 0 else out


# ---------------------------- Data Models ---------------------------- #


class Race:
    """Race dataset: cleaning, summaries, and visuals."""

    def __init__(self, dataset_path: str = "data/results.parquet") -> None:
        self.data = self._read_any(dataset_path)
        self._clean()

    @staticmethod
    def _read_any(path: str) -> pd.DataFrame:
        """Read feather/parquet/csv by extension."""
        if path.lower().endswith((".feather", ".ft")):
            return pd.read_feather(path)
        if path.lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(path, engine="pyarrow")
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        # Default try parquet then feather then csv
        for fn in (pd.read_parquet, pd.read_feather, pd.read_csv):
            try:
                return fn(path)
            except Exception:
                continue
        raise ValueError(f"Unsupported or unreadable file: {path}")

    def _clean(self) -> None:
        """Standardize fields"""
        # Fix “1-15” → “13-15” if present
        self.data.loc[self.data["agegrp"] == "1-15", "agegrp"] = "13-15"
        # Replace all the empty agegrp with 75+. During scrapping, these were treated empty.
        self.data.replace({"agegrp": ""}, value="75+", inplace=True)
        # time conversion to suit pandas preferences
        self.data["time"] = self.data["time"].apply(
            lambda t: t.strftime("%H:%M:%S") if pd.notna(t) else "NaT"
        )
        self.data["time"] = pd.to_datetime(
            self.data["time"], format="%H:%M:%S", errors="coerce"
        )

    def age_gender_table(self) -> pd.DataFrame:
        """Count table indexed by agegrp with gender columns."""
        return (
            self.data.groupby(["agegrp", "gender"])["name"]
            .count()
            .reset_index()
            .pivot(index="agegrp", columns="gender", values="name")
            .fillna(0)
            .astype(int)
            .sort_index()
        )

    def mean_finish_time_by_age_gender(self) -> pd.DataFrame:
        """Mean finish time (minutes) pivot by agegrp × gender (excludes U, DNF)."""
        df = self.data[self.data["finished"]].copy()
        df = df[df["gender"].isin(["M", "F"])]
        # time in minutes (float) for easier plotting
        df["time_min"] = (
            df["time"].dt.hour * 60 + df["time"].dt.minute + df["time"].dt.second / 60
        )
        piv = (
            df.groupby(["agegrp", "gender"])["time_min"]
            .mean()
            .reset_index()
            .pivot(index="agegrp", columns="gender", values="time_min")
            .sort_index()
        )
        piv["delta(F−M)"] = piv.get("F") - piv.get("M")
        return piv

    def gender_ratio_or_none(self) -> Optional[float]:
        """Return M/F ratio if defined, else None."""
        vc = self.data["gender"].value_counts()
        m, f = vc.get("M", 0), vc.get("F", 0)
        return (m / f) if f else None


class SCB:
    """SCB population snapshot for Stockholm: helpers for age × gender."""

    def __init__(
        self, dataset_path: str = "data/MeanPop_by_year_Stockholm_age_sex.csv"
    ) -> None:
        self.data = pd.read_csv(dataset_path, sep=";")
        self._clean()

    def _clean(self) -> None:
        """Normalize types and column names."""
        self.data.replace({"age": "100+"}, 100, inplace=True)
        self.data["age"] = self.data["age"].astype(int)
        # Normalize column names → men/women if needed
        self.data.rename(columns={"Men": "men", "Women": "women"}, inplace=True)

    def age_gender_table(self, agegrps: Iterable[str]) -> pd.DataFrame:
        """Aggregate SCB counts to the provided age range bins (rows), cols=['F','M']."""

        def sum_range(txt: str) -> pd.Series:
            # Accept 'x-y' or 'x+'
            if "+" in txt:
                low = int(txt.replace("+", ""))
                mask = self.data["age"] >= low
            else:
                low, high = map(int, txt.split("-"))
                mask = (self.data["age"] >= low) & (self.data["age"] <= high)
            total = self.data.loc[mask, ["men", "women"]].sum()
            return pd.Series({"M": total["men"], "F": total["women"]})

        out = pd.DataFrame(index=list(agegrps), columns=["M", "F"])
        out = out.apply(lambda _: sum_range(_.name), axis=1)
        return out.astype(int).sort_index()

    def gender_ratio(self) -> float:
        """Return men/women ratio across all rows."""
        s = self.data[["men", "women"]].sum()
        if s["women"] == 0:
            return float("nan")
        return float(s["men"] / s["women"])


# ---------------------------- Plots ---------------------------- #


def build_age_gender_pyramid(
    scb_tbl: pd.DataFrame,
    scb_norm: pd.DataFrame,
    race_tbl: pd.DataFrame,
    race_norm: pd.DataFrame,
) -> go.Figure:
    """Horizontal pyramid: SCB vs Race normalized shares by age."""
    y = list(scb_norm.index)
    fig = go.Figure()

    # SCB
    fig.add_bar(y=y, x=scb_norm["M"], name="SCB — Men", orientation="h", opacity=0.45)
    fig.add_bar(
        y=y, x=-scb_norm["F"], name="SCB — Women", orientation="h", opacity=0.45
    )

    # Race
    fig.add_bar(
        y=y, x=race_norm.get("M", 0), name="Race — M", orientation="h", opacity=0.45
    )
    fig.add_bar(
        y=y, x=-race_norm.get("F", 0), name="Race — F", orientation="h", opacity=0.45
    )

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            title="Normalized share",
            tickvals=[-0.25, -0.1, 0, 0.1, 0.25],
            ticktext=["0.25", "0.10", "0", "0.10", "0.25"],
        ),
        yaxis=dict(title="Age group"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def build_finish_histogram(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """Histogram of finish times by gender (excludes DNF and 'U'); returns (figure, data used, medians)."""
    plot_df = df[(df["finished"]) & (df["gender"].isin(["M", "F"]))].copy()
    plot_df["time_min"] = (
        plot_df["time"].dt.hour * 60
        + plot_df["time"].dt.minute
        + plot_df["time"].dt.second / 60
    )

    hist_data = [
        plot_df[plot_df["gender"] == "F"]["time_min"],
        plot_df[plot_df["gender"] == "M"]["time_min"],
    ]
    group_labels = ["Female", "Male"]
    colors = ["#FF0000", "#008000"]

    fig = ff.create_distplot(
        hist_data,
        group_labels,
        bin_size=1,
        curve_type="normal",
        show_hist=True,
        show_rug=False,
        colors=colors,
    )
    # Medians
    for gender in ["M", "F"]:
        gender_data = plot_df[plot_df["gender"] == gender]
        median_time = gender_data["time_min"].median()

        # Add vertical line for median
        fig.add_vline(
            x=median_time,
            line_dash="dash",
            line_color="black",
            annotation_text=f"{median_time:.1f}",
            annotation_position="top",
            col=1 if gender == "M" else 2,  # Specify which column (facet)
        )
    fig.update_xaxes(title="Finish time (min)")
    fig.update_yaxes(title="Normal Distribution")
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig, plot_df
