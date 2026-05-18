from datetime import datetime
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


def now_ny() -> datetime:
    return datetime.now(NY_TZ)


def ny_date_str() -> str:
    return now_ny().strftime("%Y-%m-%d")
