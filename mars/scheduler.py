"""
MARS Scheduled Monitoring — automated periodic literature tracking.

Keeps a persistent JSON config of research topics and runs incremental
searches at scheduled intervals to find new papers.

Usage::

    from mars.scheduler import Scheduler

    sched = Scheduler()
    sched.add("federated learning privacy", interval_hours=24, at_time="09:00")
    sched.remove("federated learning privacy")
    sched.list_schedules()
    sched.run_due()     # run all topics whose interval has elapsed
    sched.serve()       # background loop (blocking)
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mars.config import settings

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / ".mars" / "schedules.json"


@dataclass
class ScheduleEntry:
    """A single scheduled research topic."""
    topic: str
    interval_hours: int = 24
    at_time: str = "09:00"       # "HH:MM" local time
    last_run: str = ""            # ISO timestamp
    max_papers: int = 50
    enabled: bool = True

    def is_due(self, now: datetime.datetime | None = None) -> bool:
        """Check if enough time has passed since last_run."""
        if not self.enabled or not self.last_run:
            return True
        now = now or datetime.datetime.now()
        try:
            last = datetime.datetime.fromisoformat(self.last_run)
        except (ValueError, TypeError):
            return True
        delta = datetime.timedelta(hours=self.interval_hours)
        return (now - last) >= delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "interval_hours": self.interval_hours,
            "at_time": self.at_time,
            "last_run": self.last_run,
            "max_papers": self.max_papers,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScheduleEntry":
        return cls(
            topic=d.get("topic", ""),
            interval_hours=d.get("interval_hours", 24),
            at_time=d.get("at_time", "09:00"),
            last_run=d.get("last_run", ""),
            max_papers=d.get("max_papers", 50),
            enabled=d.get("enabled", True),
        )


class Scheduler:
    """Manages scheduled research topic monitoring.

    Persists to ``~/.mars/schedules.json``.
    """

    def __init__(self, config_path: str | Path | None = None):
        self._path = Path(config_path or _CONFIG_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, ScheduleEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        topic: str,
        interval_hours: int = 24,
        at_time: str = "09:00",
        max_papers: int = 50,
    ) -> ScheduleEntry:
        """Add or update a scheduled topic."""
        entry = ScheduleEntry(
            topic=topic,
            interval_hours=interval_hours,
            at_time=at_time,
            max_papers=max_papers,
        )
        self._entries[topic] = entry
        self._save()
        logger.info("Schedule added: '%s' every %dh at %s", topic, interval_hours, at_time)
        return entry

    def remove(self, topic: str) -> bool:
        """Remove a scheduled topic. Returns True if it existed."""
        if topic in self._entries:
            del self._entries[topic]
            self._save()
            logger.info("Schedule removed: '%s'", topic)
            return True
        return False

    def get(self, topic: str) -> Optional[ScheduleEntry]:
        return self._entries.get(topic)

    def list_all(self) -> List[ScheduleEntry]:
        return list(self._entries.values())

    def set_enabled(self, topic: str, enabled: bool) -> bool:
        entry = self._entries.get(topic)
        if entry:
            entry.enabled = enabled
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_due(self) -> List[Dict[str, Any]]:
        """Run all schedules whose interval has elapsed. Returns per-topic logs."""
        now = datetime.datetime.now()
        due = [e for e in self._entries.values() if e.is_due(now)]
        results: List[Dict[str, Any]] = []

        for entry in due:
            logger.info("Running scheduled topic: '%s'", entry.topic)
            try:
                output = self._run_one(entry)
                entry.last_run = now.isoformat()
                results.append({"topic": entry.topic, "status": "success", "output": output})
            except Exception as exc:
                logger.error("Scheduled run failed for '%s': %s", entry.topic, exc)
                results.append({"topic": entry.topic, "status": "error", "error": str(exc)})

        if due:
            self._save()
        return results

    def serve(self, check_interval: int = 60) -> None:
        """Run the scheduler in a background loop (blocking).

        Args:
            check_interval: Seconds between checks for due schedules.
        """
        logger.info("MARS scheduler started (check every %ds). Ctrl+C to stop.", check_interval)
        try:
            while True:
                due_count = sum(1 for e in self._entries.values() if e.is_due() and e.enabled)
                if due_count > 0:
                    logger.info("%d schedule(s) due — running now.", due_count)
                    self.run_due()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped.")

    def serve_background(self, check_interval: int = 60) -> threading.Thread:
        """Start the scheduler in a background daemon thread."""
        t = threading.Thread(target=self.serve, args=(check_interval,), daemon=True)
        t.start()
        logger.info("Scheduler background thread started.")
        return t

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_one(self, entry: ScheduleEntry) -> Dict[str, Any]:
        """Execute a single scheduled topic search."""
        from mars.crews.search_crew import run_search

        result = run_search(entry.topic, max_results=entry.max_papers)
        return {
            "result_length": len(result),
            "output_dir": str(settings.OUTPUT_DIR),
        }

    def _load(self) -> None:
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._entries = {
                    topic: ScheduleEntry.from_dict(d)
                    for topic, d in data.items()
                }
                logger.info("Loaded %d schedule(s) from %s", len(self._entries), self._path)
            except (json.JSONDecodeError, KeyError):
                self._entries = {}

    def _save(self) -> None:
        data = {topic: e.to_dict() for topic, e in self._entries.items()}
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
