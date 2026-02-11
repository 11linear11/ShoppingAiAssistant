import importlib
import logging



def _reload_pipeline_logger(monkeypatch, debug_log: bool):
    monkeypatch.setenv("DEBUG_LOG", "true" if debug_log else "false")
    monkeypatch.setenv("DEBUG_MODE", "false")
    monkeypatch.setenv("PIPELINE_LOG_TO_FILE", "false")

    import src.pipeline_logger as pipeline_logger

    return importlib.reload(pipeline_logger)


def test_non_debug_mode_keeps_only_user_requests_and_errors(monkeypatch):
    pl = _reload_pipeline_logger(monkeypatch, debug_log=False)

    events = []

    def fake_log(level, message, extra=None, exc_info=None):
        events.append((level, message, extra))

    monkeypatch.setattr(pl.pipeline_logger, "log", fake_log)

    pl.log_pipeline("SEARCH", "normal info", level=logging.INFO)
    pl.log_pipeline("AGENT", "USER_REQUEST", level=logging.INFO)
    pl.log_pipeline("SEARCH", "something failed", level=logging.ERROR)

    assert len(events) == 2
    assert events[0][1].startswith("USER_REQUEST")
    assert events[1][0] == logging.ERROR


def test_debug_mode_logs_normal_events(monkeypatch):
    pl = _reload_pipeline_logger(monkeypatch, debug_log=True)

    events = []

    def fake_log(level, message, extra=None, exc_info=None):
        events.append((level, message, extra))

    monkeypatch.setattr(pl.pipeline_logger, "log", fake_log)

    pl.log_pipeline("SEARCH", "normal info", level=logging.INFO)
    assert len(events) == 1
    assert events[0][2]["stage"] == "SEARCH"


def test_truncate_data_redacts_sensitive_keys_and_long_values(monkeypatch):
    pl = _reload_pipeline_logger(monkeypatch, debug_log=True)

    data = {
        "token": "secret-token",
        "query": "x" * 200,
        "nested": {"password": "p", "ok": "yes"},
    }
    out = pl._truncate_data(data, max_len=20)

    assert out["token"] == "***REDACTED***"
    assert out["query"].endswith("...")
    assert out["nested"]["password"] == "***REDACTED***"


def test_trace_stage_appends_stage_result(monkeypatch):
    pl = _reload_pipeline_logger(monkeypatch, debug_log=True)

    with pl.trace_query("سلام", "sess-1") as trace:
        with pl.trace_stage("INTERPRET", "classify"):
            pass

    assert len(trace.stages) == 1
    assert trace.stages[0]["stage"] == "INTERPRET"
    assert trace.stages[0]["success"] is True
