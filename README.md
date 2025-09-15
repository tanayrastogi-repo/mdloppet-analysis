# Midnattsloppet 2025 Results Scraper

A robust, production-ready scraper for Midnattsloppet results that supports two backends:
1) Playwright (default; fast and resilient for JS-heavy sites)
2) Selenium (alternative; uses a single, reusable Firefox driver)

Key features
------------
- Reuses a single browser context/driver for *all* pages (no per-page spawn).
- Explicit, resilient waiting strategies with retries and backoff.
- Clean HTML parsing with BeautifulSoup and header-row detection.
- Locale-aware parsing for "Class"/"Klass" labels.
- Optional randomized polite sleeps between pages.
- Checkpointing to avoid data loss on long runs.
- Flexible outputs: Feather (preferred), Parquet, or CSV with automatic fallback.

Usage (examples)
----------------
    # Basic scrape with Playwright for pages 1..100 and save to Parquet
    python midnattsloppet_scraper.py --start-page 1 --end-page 100 --format parquet --output results.parquet

    # Use Selenium backend instead of Playwright (requires geckodriver in PATH)
    python midnattsloppet_scraper.py --backend selenium --start-page 1 --end-page 100 --output results.parquet

    # Change base URL if needed and enable verbose logs
    python midnattsloppet_scraper.py --base-url "https://results.midnattsloppet.com/stockholm/results" --verbose


Important Flags
----------------

- `--base-url` (default: `https://results.midnattsloppet.com/stockholm/results`)  
  The script appends `?page=<n>` for each page.

- `--start-page`, `--end-page`  
  Define the page range (inclusive).

- `--backend` (`playwright` | `selenium`)  
  Choose the engine. Playwright is faster and more resilient for JS-heavy sites.

- `--headless` / `--no-headless`  
  Run without or with browser UI.

- `--user-agent`  
  Set a custom UA string for politeness.

- `--timeout-sec`, `--retry-count`, `--backoff-base-sec`  
  Control page wait time, number of retries, and exponential backoff.

- `--wait-selector`, `--optional-selector`  
  CSS selectors to wait for. Required selector must appear before parsing. Optional selector (e.g., pagination) is attempted but not required.

- `--polite-sleep-min`, `--polite-sleep-max`  
  Randomized delay between pages (e.g., `--polite-sleep-min 0.5 --polite-sleep-max 2.0`).

- `--checkpoint-interval`, `--checkpoint-path`  
  Write a checkpoint file every N pages to avoid losing progress. If `--checkpoint-path` is omitted, it derives from `--output`.

- `--geckodriver-path` (Selenium only)  
  Path to geckodriver if not on PATH.





Dependencies
------------
- BeautifulSoup4  (bs4)
- pandas
- Playwright (for default backend): `pip install playwright` then `playwright install`
- Selenium (for alternative backend): `pip install selenium` and have `geckodriver` installed
- Optional for Feather/Parquet: `pyarrow` or `fastparquet`

Notes
-----
- The scraper assumes the public structure of results pages. If the site layout changes,
  adjust the CSS selectors in `parse_rows` accordingly.
- Default timeouts and waits are conservative and can be tuned via CLI flags.


AI Declaration
-----
The original code was human-code to scrape the data, however the updates using Playwright and some re-factoring is done using ChatGPT.
