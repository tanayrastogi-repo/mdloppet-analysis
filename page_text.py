# page_text.py
import streamlit as st


def start() -> None:
    """Intro panel."""
    st.markdown(
        """
**Another Saturday night, with thousands of runners, drums, and neon—Midnattsloppet Stockholm 10K (2025) done and dusted!**

Second year in a row, and this time I cracked the sub‑60 barrier with an official 59:38.
It’s definitely an improvement from last year’s run, and seeing that finish clock start with a 5 felt ridiculously good. However, after shaking off the race buzz, there were some curious questions popped into my head. This blog is mostly to scratch that curiosity using "DATA". Mostly, I am trying to answer:
- Who are the runners in this race?
- How does my performance compare to others in my age group?
"""
    )


def race_data() -> None:
    """Short description for the race dataset section."""
    st.markdown(
        """
**Race data** for this analysis is coming from directly from their result website - [Midnattsloppet 2025 Result](https://results.midnattsloppet.com/stockholm/?q). I scraped all the data from the results table into a parquet file, that is then used for analysis and visualization. Gender and age groups are derived from the “Class” field; finishers are those with an allocated place.
        """
    )


def scb_data() -> None:
    """Short description for the SCB dataset."""
    st.markdown(
        """
**SCB data** provides Stockholm’s population by age and gender in a CSV format. Its a open-source data platform to collect aggregates about Swedish population - [Mean population (by year of birth) by region, age and sex. Year 2006 - 2024](https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__BE__BE0101__BE0101D/MedelfolkFodelsear/).
        """
    )


def age_gender_pyramid() -> None:
    """Note for the pyramid chart section."""
    st.markdown(
        """
We compare normalized **Age × Gender** profiles: population vs. race. 

    Insights from Chart:

    - Being a very fitness focused city, unsprisingly, the participation on these races follows the population distribution quite closely.
    - The participation is huge in the age group 23-34, even more than the share of actual population in that age group.
    - There is a good aount of participation from 60+, however is less than the population share.
    - Male domination is visible in all age groups, except at in group 20-22 and 23-34.

Caveat:
- Removed category "U" in race because lack of information in SCB data.
- The race participants are likely not representative of the entire Stockholm population -- there are many who are coming out of town to run this.
"""
    )


def finish_time_1() -> None:
    st.markdown(
        """
The race finish times categoried by Gender are shown.

    Insights from Chart:

    - The men (at an average) are 8.3 min faster than women in this race.
    - The tail for both of the distributions are quite long, indicating a few very slow finishers.

Caveat:
- The graph above is only for the participants who finished the race.

Hover or select category to inspect ranges in the plot below.
        """
    )


def finish_time_2() -> None:
    st.markdown(
        """
Averages by **Age × Gender** give a compact view; medians (on the histogram) are robust for skew.
        """
    )


def me_vs_everyone() -> None:
    st.markdown(
        """
Averages by **Age × Gender** give a compact view; medians (on the histogram) are robust for skew.
        """
    )
