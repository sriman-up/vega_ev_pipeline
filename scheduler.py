# scheduler.py
"""
Monthly scheduler — wraps pipeline.monthly_update() using APScheduler.
Run this as a long-lived process (e.g. in a systemd service or Docker container).

Alternatively, add a cron entry:
  0 8 10 * *  /usr/bin/python /path/to/ev_pipeline/pipeline.py --mode monthly

Usage:
    python scheduler.py
"""

import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import MONTHLY_SCRAPE_DAY, LOG_FILE
from pipeline import monthly_update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger("scheduler")

scheduler = BlockingScheduler(timezone="Asia/Kolkata")

# Run on MONTHLY_SCRAPE_DAY of every month at 08:00 IST
scheduler.add_job(
    monthly_update,
    CronTrigger(day=MONTHLY_SCRAPE_DAY, hour=8, minute=0),
    id="monthly_bill_scrape",
    name="Monthly TGSPDCL Bill Scrape + Feature Update",
    misfire_grace_time=3600,
    coalesce=True,
)

log.info(
    "Scheduler started. Monthly job fires on day %d of each month at 08:00 IST.",
    MONTHLY_SCRAPE_DAY,
)

if __name__ == "__main__":
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")