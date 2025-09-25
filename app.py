# app.py
from __future__ import annotations

import streamlit as st
from components import (
    Race,
    SCB,
    normalize_share,
    build_age_gender_pyramid,
    build_finish_histogram,
)
import page_text
from pathlib import Path
from urllib.parse import quote

st.set_page_config(
    page_title="Midnattsloppet 10K — 2025", layout="wide", page_icon="🏃‍♂️"
)


def _css() -> None:
    """Lightweight page CSS."""
    st.markdown(
        """
        <style>
        .responsive-svg {
            max-width: 40%;
            width: 40%;
            height: auto;
            display: block;
            text-align: center;
            margin-left: auto;
            margin-right: auto;
        }
        .title {
            font-size: 50px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;

        }
    </style>
    """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_race(path: str) -> Race:
    """Cached loader for race dataset."""
    return Race(dataset_path=path)


@st.cache_data(show_spinner=False)
def _load_scb(path: str) -> SCB:
    """Cached loader for SCB dataset."""
    return SCB(dataset_path=path)


def main() -> None:
    """Streamlit entrypoint."""
    _css()

    # --- Title ---
    # Add the title image
    svg = Path("static/midnattsloppet-logo.svg").read_text(encoding="utf-8")
    data_uri = f"data:image/svg+xml;utf8,{quote(svg)}"
    st.html(f"""
    <img class="responsive-svg" src="{data_uri}" alt="Diagram" />
    """)
    # Title
    st.html("""
    <div class="title">
        My Race Analysis - Midnattsloppet Stockholm 10K — 2025
    </div>
    """)
    # My images
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.image("static/image1.jpeg", width="content")
    with col2:
        with st.container(border=True):
            st.image("static/image2.jpeg", width="content")

    # --- Sidebar inputs ---
    st.sidebar.header("Data")
    race_path = st.sidebar.text_input(
        "Race file (.parquet/.feather/.csv)", "data/results.parquet"
    )
    scb_path = st.sidebar.text_input(
        "SCB file (.csv)", "data/MeanPop_by_year_Stockholm_age_sex.csv"
    )

    # --- Intro text ---
    page_text.start()

    # --- Load data (cached) ---
    try:
        race = _load_race(race_path)
    except Exception as e:
        st.error(f"Could not load race data: {e}")
        return
    try:
        scb = _load_scb(scb_path)
    except Exception as e:
        st.error(f"Could not load SCB data: {e}")
        return

    # --- Data overview ---
    st.markdown("---")
    st.subheader("Data Overview")
    page_text.race_data()

    total = len(race.data)
    finished = int(race.data["finished"].sum())
    fastest = race.data.loc[race.data["finished"], "time"].min()
    slowest = race.data.loc[race.data["finished"], "time"].max()
    average = race.data.loc[race.data["finished"], "time"].mean()
    m_f_ratio_race = race.gender_ratio_or_none()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participants", f"{total:,}")
    c2.metric("Finished", f"{finished:,}")
    c3.metric("Fastest", f"{fastest.strftime('%H:%M:%S') if fastest else '—'}")
    c4.metric("Average", f"{average.strftime('%H:%M:%S') if average else '—'}")
    c5.metric("Slowest", f"{slowest.strftime('%H:%M:%S') if slowest else '—'}")
    # View Dataframe
    st.dataframe(race.data.head(10), width="stretch")

    # --- SCB snippet ---
    page_text.scb_data()
    # View Dataframe
    st.dataframe(scb.data.head(10), width="stretch")

    # --- Demographics ---
    st.markdown("---")
    st.subheader("Participants — Age × Gender")
    page_text.age_gender_pyramid()

    race_tbl = race.age_gender_table()
    scb_tbl = scb.age_gender_table(agegrps=race_tbl.index)
    fig_pyr = build_age_gender_pyramid(
        scb_tbl, normalize_share(scb_tbl), race_tbl, normalize_share(race_tbl)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Race gender ratio (M/F)**")
        st.metric("Ratio", f"{m_f_ratio_race:.3f}" if m_f_ratio_race else "—")
    with c2:
        st.markdown("**SCB gender ratio (men/women)**")
        st.metric("Ratio", f"{scb.gender_ratio():.3f}")

    st.plotly_chart(fig_pyr, use_container_width=True)
    with st.expander("Tables"):
        cc1, cc2 = st.columns(2)
        cc1.markdown("**SCB (counts)**")
        cc1.dataframe(scb_tbl)
        cc2.markdown("**Race (counts)**")
        cc2.dataframe(race_tbl)

    # --- Finish times ---
    st.markdown("---")
    st.subheader("Finish Time Distribution")
    page_text.finish_time_1()

    tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
    with tab1:
        fig_hist, plot_df = build_finish_histogram(race.data)
        st.plotly_chart(fig_hist, use_container_width=True)
        with st.expander("Mean finish time (min) by Age × Gender"):
            pivot = (
                plot_df.groupby(["agegrp", "gender"])["time"]
                .mean()
                .dt.second.div(60)
                .reset_index()
                .pivot(index="agegrp", columns="gender", values="time")
            )
            st.dataframe(pivot)
    with tab2:
        page_text.finish_time_2()
        st.table(race.mean_finish_time_by_age_gender())

    # --- Runner vs Everyone ---
    st.markdown("---")
    st.subheader("Individual vs Everyone")


if __name__ == "__main__":
    main()
