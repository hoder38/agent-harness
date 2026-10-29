"""arq entrypoints:  arq worker.Settings | worker.BackgroundSettings | worker.CronSettings"""
from arq import cron
from arq.connections import RedisSettings

from app.config import load_yaml
from app.jobs import alerts, memory_jobs
from app.jobs.worker import run_agent, shutdown, startup
from app.settings import get_settings

_s = get_settings()
_queues = load_yaml(_s.policy_path).get("queues", {})
_redis = RedisSettings.from_dsn(_s.redis_url)


async def _cron_startup(ctx):
    from app import db

    await db.open_db()


async def _cron_shutdown(ctx):
    from app import db

    await db.close_db()


class Settings:
    """Interactive queue (highest priority, dedicated worker)."""
    redis_settings = _redis
    queue_name = _queues.get("interactive", {}).get("name", "arq:interactive")
    functions = [run_agent]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 4
    job_timeout = int(_queues.get("interactive", {}).get("timeout_s", 300))
    max_tries = 3
    allow_abort_jobs = True
    health_check_interval = 30


class BackgroundSettings:
    """Background / coding / media queue. (arq ignores inherited attributes, so list them all.)"""
    redis_settings = _redis
    functions = [run_agent, memory_jobs.extract_memory, memory_jobs.summarize_episode, memory_jobs.index_docs_job]
    on_startup = startup
    on_shutdown = shutdown
    max_tries = 3
    allow_abort_jobs = True
    health_check_interval = 30
    queue_name = _queues.get("background", {}).get("name", "arq:background")
    max_jobs = 2
    job_timeout = int(_queues.get("background", {}).get("timeout_s", 1800))


class CronSettings:
    """Scheduler: health/budget/governor watcher every minute; deferred-job resume (§11) every 20 s;
    docs re-index, episode summaries and memory TTL sweep (§10)."""
    redis_settings = _redis
    queue_name = "arq:cron"
    functions = []
    cron_jobs = [
        cron(alerts.alert_watcher, second={0}, run_at_startup=True, unique=True, timeout=50),
        cron(alerts.resume_deferred, second={10, 30, 50}, unique=True, timeout=15),
        cron(memory_jobs.index_docs_cron, second={20}, unique=True, timeout=1800),
        cron(memory_jobs.episodes_cron, minute=set(range(0, 60, 5)), second={40}, unique=True, timeout=30),
        cron(memory_jobs.sweep_memory_ttl, minute={7}, second={0}, unique=True, timeout=300),
    ]
    on_startup = _cron_startup
    on_shutdown = _cron_shutdown
    health_check_interval = 30
