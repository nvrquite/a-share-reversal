"""锁定样本外保护的测试。"""
import copy

import pytest

from src.data import loader

CFG = loader.load_config()
SNAPSHOT = loader.ROOT / CFG["paths"]["daily_snapshot"]
needs_data = pytest.mark.skipif(not SNAPSHOT.exists(), reason="本机没有日K快照")


def test_config_is_locked():
    """仓库里的配置必须是未解锁状态。"""
    assert CFG["lock"]["final_evaluation_unlocked"] is False


@needs_data
def test_research_load_stops_before_lock():
    """研究读取的最后一天必须早于锁定期第一天。"""
    df = loader.load_daily(columns=["date", "code"])
    assert df["date"].max() < loader.lock_start(CFG)
    assert str(df["date"].min()) == CFG["splits"]["warmup"]["start"]


@needs_data
def test_research_load_without_date_column_is_still_cut():
    """不要 date 列时也要截断，并且只返回请求的列。"""
    full = loader.load_daily(columns=["date", "code"])
    only_code = loader.load_daily(columns=["code"])
    assert list(only_code.columns) == ["code"]
    assert len(only_code) == len(full)


@needs_data
def test_structural_load_is_logged(tmp_path):
    """读锁定期必须写用途，并且留下一行记录。"""
    cfg = copy.deepcopy(CFG)
    log = tmp_path / "access.csv"
    cfg["lock"]["access_log"] = str(log)
    df = loader.load_daily_structural("单元测试：检查截止日", columns=["date"], cfg=cfg)
    assert str(df["date"].max()) == CFG["splits"]["locked"]["end"]
    assert "单元测试：检查截止日" in log.read_text(encoding="utf-8")
    with pytest.raises(AssertionError):
        loader.load_daily_structural("  ", columns=["date"], cfg=cfg)


@pytest.mark.skipif(not (loader.ROOT / CFG["paths"]["wide_dir"] / "close.parquet").exists(), reason="还没有生成宽表")
def test_wide_and_clean_loads_stop_before_lock():
    """宽表和清洗后长表的研究读取，也必须在锁定期之前结束。"""
    import pandas as pd
    start = pd.Timestamp(loader.lock_start(CFG))
    assert loader.load_wide("close").index.max() < start
    assert loader.load_clean(columns=["date"])["date"].max() < start
