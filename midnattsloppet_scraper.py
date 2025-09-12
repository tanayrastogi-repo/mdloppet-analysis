#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup


# -------------------------------
# Configuration & Data Structures
# -------------------------------


@dataclass
class ScrapeConfig:
    """Configuration for the scraping run.

    Attributes
    ----------
    base_url : str
        Base URL for the results page without the `?page=` query string appended.
    start_page : int
        First page to scrape (inclusive).
    end_page : int
        Last page to scrape (inclusive).
    backend : str
        Scrape backend to use: "playwright" (default) or "selenium".
    headless : bool
        Whether to run the browser in headless mode.
    user_agent : Optional[str]
        Custom User-Agent string to reduce trivial bot detection or for politeness.
    timeout_sec : int
        Page load timeout in seconds.
    wait_selector : str
        CSS selector that indicates the results table is present in the DOM.
    optional_selector : Optional[str]
        An optional selector to wait for (e.g., pagination). Absence won't fail the page.
    retry_count : int
        Max number of retries per page.
    backoff_base_sec : float
        Base seconds for exponential backoff between retries.
    polite_sleep_min : float
        Minimum randomized sleep in seconds between pages (0 for none).
    polite_sleep_max : float
        Maximum randomized sleep in seconds between pages (0 for none).
    checkpoint_interval : int
        Save a checkpoint every N pages. 0 disables checkpointing.
    checkpoint_path : Optional[str]
        Where to write checkpoint files. Defaults to "<output>.checkpoint.parquet" or ".csv".
    verbose : bool
        Emit verbose logs to stdout.
    """

    base_url: str
    start_page: int
    end_page: int
    backend: str = "playwright"
    headless: bool = True
    user_agent: Optional[str] = None
    timeout_sec: int = 30
    wait_selector: str = ".results-table .row"
    optional_selector: Optional[str] = ".pagination"
    retry_count: int = 3
    backoff_base_sec: float = 1.5
    polite_sleep_min: float = 0.0
    polite_sleep_max: float = 0.0
    checkpoint_interval: int = 50
    checkpoint_path: Optional[str] = None
    verbose: bool = False
    geckodriver_path: Optional[str] = "bin/geckodriver"  # for selenium backend only


# --------------
# Util Functions
# --------------


def log(msg: str, verbose: bool = True) -> None:
    """Print a message if verbosity is enabled.

    Parameters
    ----------
    msg : str
        The message to print.
    verbose : bool
        Whether to print the message.
    """
    if verbose:
        print(msg, flush=True)


def sleep_politely(cfg: ScrapeConfig) -> None:
    """Sleep a randomized short interval between pages for polite scraping.

    Parameters
    ----------
    cfg : ScrapeConfig
        The current scraper configuration which controls the min/max sleep.
    """
    if cfg.polite_sleep_max <= 0:
        return
    span = max(0.0, cfg.polite_sleep_max - cfg.polite_sleep_min)
    duration = cfg.polite_sleep_min + random.random() * span
    time.sleep(duration)


def is_header_like(row_soup: BeautifulSoup) -> bool:
    """Heuristically determine whether a selected `.row` is a header row.

    This function checks for the presence/absence of data-bearing nodes and common
    header traits. Adjust if the site's HTML changes.

    Parameters
    ----------
    row_soup : BeautifulSoup
        A soup node corresponding to one "row" in the results table.

    Returns
    -------
    bool
        True if the row appears to be a header or a non-data row, False otherwise.
    """
    # Example heuristic: data rows have an anchor inside `.name.row-content a`
    if row_soup.select_one(".name.row-content a"):
        return False
    # If the row clearly lacks `.row-content` children, it's probably not a data row.
    if not row_soup.select(".row-content"):
        return True
    # If the row has table headers (`.table-header`) but no typical content, call it header-like.
    if row_soup.select(".table-header") and not row_soup.select(".name.row-content"):
        return True
    return False


def extract_text(node: Optional[BeautifulSoup]) -> Optional[str]:
    """Extract stripped text from a BeautifulSoup node, returning None if node is missing.

    Parameters
    ----------
    node : Optional[BeautifulSoup]
        The node to extract text from.

    Returns
    -------
    Optional[str]
        The node's stripped text, or None if the node is None.
    """
    if node is None:
        return None
    return node.get_text(strip=True)


def parse_rows(html: str) -> List[dict]:
    """Parse a results page's HTML into a list of record dictionaries with strict types.

    The parser is defensive and tries to:
    - Skip header-like rows
    - Separate `bib` from `name` using DOM if possible; else fallback to regex
    - Map "Class" / "Klass" to the same `class` field and split it into `gender` and `agegrp`
      (e.g., "M23-34" → gender="M", agegrp="23-34"; supports en dash "–" and "65+")
    - Extract `place` (int if present), `team`, primary `time` as datetime.time, and `finished` (bool)

    Returns
    -------
    List[dict]
        Keys and enforced Python types:
          - "place": int if present else None
          - "bib": int if present else None
          - "name": str ('' if missing)
          - "class": str ('' if missing)
          - "gender": str ('' if missing)
          - "agegrp": str ('' if missing)
          - "team": str ('' if missing)
          - "time": datetime.time if parseable else None
          - "finished": bool (True iff place is an int)
    """
    from datetime import time as dtime

    def _to_int(s: Optional[str]) -> Optional[int]:
        """Best-effort cast to int from a string containing digits; returns None on failure."""
        if not s:
            return None
        txt = s.strip()
        m = re.match(r"^\D*(\d+)\D*$", txt)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _parse_time_to_timeobj(s: Optional[str]) -> Optional[dtime]:
        """Parse 'mm:ss' or 'h:mm:ss' into datetime.time; returns None if not parseable.

        Examples handled:
            '30:51'   -> 00:30:51
            '0:00'    -> 00:00:00
            '7:29'    -> 00:07:29
            '1:34:25' -> 01:34:25
        """
        if not s:
            return None

        # Normalize spaces, non-ASCII colon, and trim
        txt = s.strip().replace(" ", "").replace("：", ":")
        parts = txt.split(":")
        if len(parts) == 2:
            # mm:ss
            mm, ss = parts
            if not (mm.isdigit() and ss.isdigit()):
                return None
            m_val = int(mm)
            s_val = int(ss)
            if not (0 <= m_val <= 59 and 0 <= s_val <= 59):
                return None
            return dtime(hour=0, minute=m_val, second=s_val)
        elif len(parts) == 3:
            # h:mm:ss
            hh, mm, ss = parts
            if not (hh.isdigit() and mm.isdigit() and ss.isdigit()):
                return None
            h_val = int(hh)
            m_val = int(mm)
            s_val = int(ss)
            # datetime.time can't exceed 23 hours; race times reasonably fall under that.
            if not (0 <= h_val <= 23 and 0 <= m_val <= 59 and 0 <= s_val <= 59):
                return None
            return dtime(hour=h_val, minute=m_val, second=s_val)
        else:
            return None

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select(".results-table .row")
    out: List[dict] = []

    for row in rows:
        if is_header_like(row):
            continue

        # Name (and possibly bib) — prefer separate DOM nodes if available.
        name_anchor = row.select_one(".name.row-content a")
        name_raw = extract_text(name_anchor)

        bib: Optional[int] = None
        name_txt: str = name_raw or ""

        # If no dedicated bib node exists, try regex fallback on the name text.
        if name_raw:
            # If the first token is numeric, treat it as bib.
            m_name = re.match(r"^\s*(\d+)\s+(.*)$", name_raw)
            if m_name:
                bib = _to_int(m_name.group(1))
                name_txt = (m_name.group(2) or "").strip()

        # Place (often first pair's row-content)
        place_el = row.select_one(".pair .row-content")
        place_txt = extract_text(place_el)
        place_int = _to_int(place_txt)

        # Team
        team_el = row.select_one(".team.row-content")
        team_txt = extract_text(team_el) or ""

        # Class / Klass: search all pairs for a header label match
        classcat: Optional[str] = None
        for pair in row.select(".pair"):
            header = pair.select_one(".table-header")
            label = extract_text(header)
            if label:
                label_l = label.lower()
                if label_l in {"class", "klass"}:
                    classcat = extract_text(pair.select_one(".row-content"))
                    break

        # Derive gender and agegrp from classcat when possible
        gender_txt: str = ""
        agegrp_txt: str = ""
        class_txt: str = (classcat or "").strip()
        if class_txt:
            # Normalize: collapse spaces and convert en dash to hyphen
            cc_norm = class_txt.replace("\u2013", "-").replace(" ", "")
            # Pattern: <letters><age?>   e.g., M23-34, K45-49, F65+
            m_class = re.match(
                r"^([A-Za-zÅÄÖåäö]{1,3})([0-9]{1,2}(?:-[0-9]{1,2}|\+))?$", cc_norm
            )
            if m_class:
                gender_txt = m_class.group(1).upper()
                if m_class.group(2):
                    agegrp_txt = m_class.group(2)
            else:
                # Fallbacks: extract age and gender heuristically
                m_age = re.search(r"([0-9]{1,2}(?:-[0-9]{1,2}|\+))", cc_norm)
                if m_age:
                    agegrp_txt = m_age.group(1)
                m_gen = re.match(r"([A-Za-zÅÄÖåäö]{1,3})", cc_norm)
                if m_gen:
                    gender_txt = m_gen.group(1).upper()

        # Time: parse to datetime.time (supports mm:ss and h:mm:ss)
        time_el = row.select_one(".pair.right .row-content")
        time_txt = extract_text(time_el)
        time_obj = _parse_time_to_timeobj(time_txt)

        # Finished flag: True if a place value is present (parsed to int), else False
        finished = place_int is not None

        # Only keep rows that have a name or bib – discard blanks.
        if name_txt or (bib is not None):
            out.append(
                {
                    "place": place_int,  # int | None
                    "bib": bib,  # int | None
                    "name": name_txt,  # str
                    "class": class_txt,  # str
                    "gender": gender_txt,  # str
                    "agegrp": agegrp_txt,  # str
                    "team": team_txt,  # str
                    "time": time_obj,  # datetime.time | None
                    "finished": finished,  # bool
                }
            )

    return out


# -----------------------
# Backend: Playwright API
# -----------------------


def get_html_playwright(url: str, cfg: ScrapeConfig) -> str:
    """Fetch a page's HTML using Playwright (synchronous API).

    Parameters
    ----------
    url : str
        The fully-qualified URL of the page to load.
    cfg : ScrapeConfig
        The scraper configuration controlling timeouts, user-agent, and headless mode.

    Returns
    -------
    str
        The page HTML string, or an empty string on failure.

    Notes
    -----
    - Uses `wait_until="networkidle"` and then explicitly waits for `cfg.wait_selector` to be present.
    - If `cfg.optional_selector` is provided, tries to wait for it but doesn't fail if missing.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
    except Exception as e:
        raise RuntimeError(
            "Playwright is not installed. Install with `pip install playwright` and run `playwright install`."
        ) from e

    html = ""
    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=cfg.headless)
            context_kwargs = {}
            if cfg.user_agent:
                context_kwargs["user_agent"] = cfg.user_agent
            context = browser.new_context(**context_kwargs)
            page = context.new_page()

            page.set_default_navigation_timeout(cfg.timeout_sec * 1000)
            page.set_default_timeout(cfg.timeout_sec * 1000)

            page.goto(url, wait_until="networkidle")
            page.wait_for_selector(cfg.wait_selector, timeout=cfg.timeout_sec * 1000)
            if cfg.optional_selector:
                try:
                    page.wait_for_selector(cfg.optional_selector, timeout=3000)
                except PWTimeoutError:
                    # Optional selector missing is fine
                    pass

            html = page.content()
            context.close()
            browser.close()
    except Exception as e:
        # Return empty string on failure; caller will handle retries.
        html = ""
    return html


# --------------------
# Backend: Selenium API
# --------------------


def make_selenium_driver(cfg: ScrapeConfig):
    """Create a single reusable Firefox Selenium WebDriver instance.

    Parameters
    ----------
    cfg : ScrapeConfig
        Current scraper configuration. Uses `cfg.headless`, `cfg.user_agent`, and `cfg.geckodriver_path`.

    Returns
    -------
    selenium.webdriver.Firefox
        A configured Firefox WebDriver.

    Raises
    ------
    RuntimeError
        If Selenium or its dependencies are not installed.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.firefox.service import Service
        from selenium.webdriver.firefox.options import Options
    except Exception as e:
        raise RuntimeError(
            "Selenium is not installed. Install with `pip install selenium`."
        ) from e

    options = Options()
    options.headless = cfg.headless
    if cfg.user_agent:
        options.set_preference("general.useragent.override", cfg.user_agent)

    service = Service(cfg.geckodriver_path) if cfg.geckodriver_path else Service()
    driver = webdriver.Firefox(service=service, options=options)
    return driver


def get_html_selenium(driver, url: str, cfg: ScrapeConfig) -> str:
    """Fetch a page's HTML using a reusable Selenium WebDriver.

    Parameters
    ----------
    driver : selenium.webdriver.Firefox
        A reusable Firefox WebDriver returned by `make_selenium_driver`.
    url : str
        The fully-qualified URL of the page to load.
    cfg : ScrapeConfig
        The scraper configuration controlling timeouts and selectors.

    Returns
    -------
    str
        The page HTML string, or an empty string on failure.
    """
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except Exception:
        return ""

    html = ""
    try:
        driver.set_page_load_timeout(cfg.timeout_sec)
        driver.get(url)
        wait = WebDriverWait(driver, cfg.timeout_sec)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, cfg.wait_selector)))
        if cfg.optional_selector:
            try:
                wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, cfg.optional_selector)
                    )
                )
            except Exception:
                # optional element not found is fine
                pass
        html = driver.page_source
    except Exception:
        html = ""
    return html


# -----------------------
# Core Scraping Functions
# -----------------------


def fetch_page_html(url: str, cfg: ScrapeConfig, selenium_driver=None) -> str:
    """Fetch a page's HTML according to the selected backend.

    Parameters
    ----------
    url : str
        The target page URL.
    cfg : ScrapeConfig
        Scraper configuration specifying the backend and waits.
    selenium_driver : Optional[selenium.webdriver.Firefox]
        Reusable Selenium driver if backend is "selenium". Ignored for Playwright.

    Returns
    -------
    str
        HTML string of the page, or empty string on error.
    """
    if cfg.backend.lower() == "selenium":
        if selenium_driver is None:
            raise ValueError("selenium_driver is required when backend='selenium'.")
        return get_html_selenium(selenium_driver, url, cfg)
    else:
        return get_html_playwright(url, cfg)


def scrape_range(cfg: ScrapeConfig) -> pd.DataFrame:
    """Scrape a range of pages and return a consolidated DataFrame.

    Parameters
    ----------
    cfg : ScrapeConfig
        The scraper configuration containing page range, backend choice, and other options.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns: ["place", "bib", "name", "class", "team", "time"].
    """
    all_frames: List[pd.DataFrame] = []
    checkpoint_template = cfg.checkpoint_path

    # Prepare Selenium driver if needed
    selenium_driver = None
    if cfg.backend.lower() == "selenium":
        selenium_driver = make_selenium_driver(cfg)

    try:
        total_pages = cfg.end_page - cfg.start_page + 1
        for idx, page in enumerate(range(cfg.start_page, cfg.end_page + 1), start=1):
            url = f"{cfg.base_url}?page={page}"
            log(f"[{idx}/{total_pages}] Fetching: {url}", cfg.verbose)

            # Retry loop
            html = ""
            for attempt in range(1, cfg.retry_count + 1):
                html = fetch_page_html(url, cfg, selenium_driver=selenium_driver)
                if html:
                    break
                backoff = cfg.backoff_base_sec**attempt
                log(
                    f"  Attempt {attempt} failed. Backing off {backoff:.2f}s",
                    cfg.verbose,
                )
                time.sleep(backoff)

            if not html:
                log(
                    f"  Failed to fetch page {page} after {cfg.retry_count} attempts. Skipping.",
                    True,
                )
                continue

            rows = parse_rows(html)
            log(f"  Parsed {len(rows)} rows from page {page}", cfg.verbose)

            if rows:
                all_frames.append(pd.DataFrame(rows))

            # Checkpointing
            if (
                cfg.checkpoint_interval
                and (idx % cfg.checkpoint_interval == 0)
                and all_frames
            ):
                checkpoint_path = checkpoint_template
                if not checkpoint_path:
                    # Infer checkpoint path from final output if possible — caller may change it later.
                    checkpoint_path = "checkpoint.parquet"
                tmp_df = pd.concat(all_frames, ignore_index=True)
                save_results(
                    tmp_df,
                    checkpoint_path,
                    fmt=_infer_format_from_path(checkpoint_path),
                    verbose=cfg.verbose,
                )
                log(f"  Wrote checkpoint to {checkpoint_path}", True)

            # Polite delay
            sleep_politely(cfg)

    finally:
        if selenium_driver is not None:
            try:
                selenium_driver.quit()
            except Exception:
                pass

    if not all_frames:
        return pd.DataFrame(columns=["place", "bib", "name", "class", "team", "time"])
    return pd.concat(all_frames, ignore_index=True)


# ----------------
# Output Utilities
# ----------------


def _infer_format_from_path(path: str) -> str:
    """Infer output format from a file path's extension.

    Parameters
    ----------
    path : str
        Target path.

    Returns
    -------
    str
        One of {"feather", "parquet", "csv"} with a preference hierarchy.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".feather", ".ft"}:
        return "feather"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    if ext == ".csv":
        return "csv"
    # Default to parquet
    return "parquet"


def save_results(
    df: pd.DataFrame, output_path: str, fmt: str, verbose: bool = False
) -> None:
    """Save results to disk with fallbacks if preferred format isn't available.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    output_path : str
        Path to write the file. Extension may be adjusted if fallback occurs.
    fmt : str
        Requested format: "feather", "parquet", or "csv".
    verbose : bool
        Emit logs about chosen format and any fallbacks.
    """
    fmt = fmt.lower()
    wrote = False

    if fmt == "feather":
        try:
            df.to_feather(output_path)
            log(f"Saved Feather: {output_path}", verbose)
            wrote = True
        except Exception as e:
            log(f"Feather failed ({e}). Falling back to Parquet...", True)
            fmt = "parquet"

    if not wrote and fmt == "parquet":
        try:
            df.to_parquet(output_path, index=False)
            log(f"Saved Parquet: {output_path}", verbose)
            wrote = True
        except Exception as e:
            log(f"Parquet failed ({e}). Falling back to CSV...", True)
            fmt = "csv"

    if not wrote and fmt == "csv":
        try:
            df.to_csv(output_path, index=False)
            log(f"Saved CSV: {output_path}", verbose)
            wrote = True
        except Exception as e:
            raise RuntimeError(f"Failed to save results as CSV: {e}") from e


# --------------
# CLI Entrypoint
# --------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        A configured ArgumentParser instance for this script.
    """
    p = argparse.ArgumentParser(
        description="Scrape Midnattsloppet results with Playwright or Selenium."
    )
    p.add_argument(
        "--base-url",
        default="https://results.midnattsloppet.com/stockholm/results",
        help="Base results URL without the '?page=' query (default: %(default)s)",
    )
    p.add_argument(
        "--start-page", type=int, default=1, help="First page to scrape (inclusive)."
    )
    p.add_argument(
        "--end-page", type=int, required=True, help="Last page to scrape (inclusive)."
    )
    p.add_argument(
        "--backend",
        choices=["playwright", "selenium"],
        default="playwright",
        help="Scrape backend to use (default: %(default)s).",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser headless (default: True).",
    )
    p.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run browser with UI.",
    )
    p.add_argument(
        "--user-agent", default=None, help="Optional custom User-Agent string."
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=30,
        help="Page load timeout in seconds (default: %(default)s).",
    )
    p.add_argument(
        "--retry-count",
        type=int,
        default=3,
        help="Retry count per page (default: %(default)s).",
    )
    p.add_argument(
        "--backoff-base-sec",
        type=float,
        default=1.5,
        help="Exponential backoff base seconds (default: %(default)s).",
    )
    p.add_argument(
        "--polite-sleep-min",
        type=float,
        default=0.0,
        help="Minimum randomized sleep between pages (default: %(default)s).",
    )
    p.add_argument(
        "--polite-sleep-max",
        type=float,
        default=0.0,
        help="Maximum randomized sleep between pages (default: %(default)s).",
    )
    p.add_argument(
        "--wait-selector",
        default=".results-table .row",
        help="Required CSS selector to wait for (default: %(default)s).",
    )
    p.add_argument(
        "--optional-selector",
        default=".pagination",
        help="Optional CSS selector to wait for if present (default: %(default)s).",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Write checkpoint every N pages (0 disables).",
    )
    p.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path for checkpoint file. Defaults to '<output>.checkpoint.<fmt>'.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write final results (e.g., results.parquet).",
    )
    p.add_argument(
        "--format",
        choices=["feather", "parquet", "csv"],
        default=None,
        help="Output format; inferred from --output if omitted.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    p.add_argument(
        "--geckodriver-path",
        default=None,
        help="Path to geckodriver (Selenium backend only).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint. Parses args, runs the scrape, and writes outputs.

    Parameters
    ----------
    argv : Optional[List[str]]
        CLI arguments (defaults to sys.argv if None).
    """
    args = build_arg_parser().parse_args(argv)

    output_fmt = args.format or _infer_format_from_path(args.output)
    checkpoint_path = args.checkpoint_path
    if args.checkpoint_interval and not checkpoint_path:
        # Build a default checkpoint path based on output
        base, ext = os.path.splitext(args.output)
        checkpoint_path = f"{base}.checkpoint{ext or '.parquet'}"

    cfg = ScrapeConfig(
        base_url=args.base_url,
        start_page=args.start_page,
        end_page=args.end_page,
        backend=args.backend,
        headless=args.headless,
        user_agent=args.user_agent,
        timeout_sec=args.timeout_sec,
        wait_selector=args.wait_selector,
        optional_selector=args.optional_selector,
        retry_count=args.retry_count,
        backoff_base_sec=args.backoff_base_sec,
        polite_sleep_min=args.polite_sleep_min,
        polite_sleep_max=args.polite_sleep_max,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=checkpoint_path,
        verbose=args.verbose,
        geckodriver_path=args.geckodriver_path,
    )

    df = scrape_range(cfg)
    if df.empty:
        log("No rows were scraped. Exiting without writing output.", True)
        return

    save_results(df, args.output, fmt=output_fmt, verbose=True)
    log(f"Done. Wrote {len(df)} rows to {args.output}", True)


if __name__ == "__main__":
    main()
