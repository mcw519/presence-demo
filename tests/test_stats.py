from presence_demo.stats import ConnStats, _pct, system_stats


class Clk:
    t = 100.0

    def __call__(self):
        return self.t


def test_percentile():
    assert _pct([], 0.95) == 0.0
    assert _pct(list(range(1, 101)), 0.95) == 95


def test_conn_stats_fps_latency_bandwidth():
    c = Clk()
    s = ConnStats(clock=c)
    for i in range(10):
        c.t += 0.5
        s.on_received(1000)
        s.on_processed(
            {"decode": 1, "detect": 10 + i, "embed": 5, "tracker": 0.1, "total": 20, "w": 640, "h": 480}
        )
    s.dropped = 2
    snap = s.snapshot()
    assert snap["fps"] == 2.0 and snap["bandwidth_Bps"] == 2000
    assert snap["latency_ms"]["detect"]["avg"] == 14.5 and snap["latency_ms"]["detect"]["p95"] == 19
    assert snap["frames"] == {"received": 10, "processed": 10, "dropped": 2}
    assert snap["frame_size"] == [640, 480] and snap["rtt_ms"] is None


def test_system_stats_shape():
    st = system_stats()
    assert st["proc_rss"] > 0 and st["mem_total"] > 0 and isinstance(st["per_core"], list)
