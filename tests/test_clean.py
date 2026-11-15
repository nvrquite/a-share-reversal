"""清洗步骤的测试：用几行手造数据检查每一步。"""
import numpy as np
import pandas as pd

from src.data import clean, limits


def frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"code": ["sz.000001"] * n, "date": pd.bdate_range("2021-03-01", periods=n)}
    return pd.DataFrame({**base, **cols})


def test_merge_code_change_keeps_old_history_and_fixes_factor():
    days = pd.bdate_range("2025-02-13", periods=4)   # 13, 14 | 17, 18
    old = pd.DataFrame({"code": "sz.300114", "date": days[:2], "close": [60.0, 66.0], "close_raw": [10.0, 11.0]})
    new = pd.DataFrame({"code": "sz.302132", "date": days, "close": [10.0, 11.0, 12.0, 78.0],
                        "close_raw": [10.0, 11.0, 12.0, 13.0]})
    for d in (old, new):
        d["open"] = d["high"] = d["low"] = d["close"]
    change = [{"old": "sz.300114", "new": "sz.302132", "last_old_date": "2025-02-14"}]
    out = clean.merge_code_changes(pd.concat([old, new]), change)
    assert out["code"].unique().tolist() == ["sz.302132"]
    # 切换前用旧代码的后复权价；切换后第一天新代码的因子是 1（错的），要补上旧因子 6
    assert out["close"].tolist() == [60.0, 66.0, 72.0, 78.0]


def test_suspension_prices_become_nan():
    df = frame(open=[10.0, 10.0], high=[10.0, 10.0], low=[10.0, 10.0], close=[10.0, 10.0],
               close_raw=[5.0, 5.0], volume=[100.0, np.nan], amount=[500.0, np.nan],
               turn=[1.0, np.nan], tradestatus=[1, 0])
    out = clean.mark_suspension(df)
    assert out["suspended"].tolist() == [False, True]
    assert np.isnan(out["close"].iloc[1]) and out["volume"].iloc[1] == 0
    assert out["close_raw_filled"].iloc[1] == 5.0 and out["adj_factor"].iloc[1] == 2.0


def test_preclose_on_ex_dividend_day_is_reference_price():
    # 第 2 天 10 送 10：不复权价从 20 变成 10.5，后复权因子从 1 变 2
    df = frame(open=[20.0, 20.0], high=[20.0, 21.0], low=[20.0, 20.0], close=[20.0, 21.0],
               close_raw=[20.0, 10.5], volume=[1.0, 1.0], amount=[20.0, 10.5], adj_factor=[1.0, 2.0])
    out = clean.add_raw_prices(df)
    assert out["preclose_raw"].iloc[1] == 10.0      # 除权参考价，不是昨收 20
    assert out["high_raw"].iloc[1] == 10.5


def test_preclose_after_suspension_uses_last_traded_close():
    df = frame(open=[10.0, np.nan, 11.0], high=[10.0, np.nan, 11.0], low=[10.0, np.nan, 11.0],
               close=[10.0, np.nan, 11.0], close_raw=[10.0, np.nan, 11.0],
               volume=[1.0, 0.0, 1.0], amount=[10.0, 0.0, 11.0], adj_factor=[1.0, 1.0, 1.0])
    assert clean.add_raw_prices(df)["preclose_raw"].iloc[2] == 10.0


def test_float_shares_uses_only_past_and_fills_suspension():
    # 流通股本 100, 100, 停牌, 200, 200, 200：中位数窗口 3
    df = frame(volume=[1.0, 1.0, 0.0, 2.0, 2.0, 2.0], turn=[1.0, 1.0, np.nan, 1.0, 1.0, 1.0],
               close_raw_filled=[1.0] * 6)
    out = clean.add_float_mcap(df, window=3)
    assert out["float_shares"].tolist() == [100.0, 100.0, 100.0, 100.0, 200.0, 200.0]


def test_delisting_period_flags():
    days = pd.bdate_range("2022-05-02", periods=6)
    df = pd.DataFrame({"code": "sh.600001", "date": days, "suspended": [False, True, True, False, False, False],
                       "out_date": pd.Timestamp("2022-06-01")})
    rules = limits.load_rules()
    out = clean.add_delisting_flags(df, rules)
    assert out["in_delisting"].tolist() == [False, False, False, True, True, True]
    assert out["delist_first_day"].tolist() == [False, False, False, True, False, False]
    still_listed = df.assign(out_date=pd.NaT)
    assert not clean.add_delisting_flags(still_listed, rules)["in_delisting"].any()


def test_delisting_first_day_has_no_limit():
    df = pd.DataFrame({"code": ["sh.600001"] * 2, "date": pd.to_datetime(["2022-05-05", "2022-05-06"]),
                       "preclose_raw": [2.0, 1.0], "isST": [0, 0], "ipo_date": pd.Timestamp("2010-01-04"),
                       "day_no": [2000, 2001], "delist_first_day": [True, False]})
    out = limits.add_limits(df)
    assert np.isnan(out["up_limit"].iloc[0]) and out["up_limit"].iloc[1] == 1.10


def test_listed_no_limit_event_from_config():
    df = pd.DataFrame({"code": ["sz.000670"], "date": pd.to_datetime(["2022-08-22"]), "preclose_raw": [2.25],
                       "isST": [0], "ipo_date": pd.Timestamp("1996-12-17"), "day_no": [2000]})
    assert np.isnan(limits.add_limits(df)["up_limit"].iloc[0])


def test_repair_adjustment_removes_spurious_factor_jump():
    # 第 3 天后复权因子无故从 2 变 1.5：后复权收益是 -24.6%，真实涨跌幅是 +0.5%
    df = frame(open=[20.0, 20.2, 15.075], high=[20.0, 20.2, 15.075], low=[20.0, 20.2, 15.075],
               close=[20.0, 20.2, 15.075], close_raw=[10.0, 10.1, 10.05])
    other = df.assign(code="sz.000002")
    ref = pd.DataFrame({"date": df["date"], "code": "sz.000001", "close": [10.0, 10.1, 10.05],
                        "preclose": [9.9, 10.0, 10.1], "tradestatus": [1, 1, 1]})
    out = clean.repair_adjustment(pd.concat([df, other], ignore_index=True), ref)
    fixed = out[out["code"] == "sz.000001"]["close"]
    assert np.allclose(fixed.pct_change().dropna(), [0.01, 10.05 / 10.1 - 1])
    assert fixed.iloc[0] == 20.0                                     # 起点不变
    assert out[out["code"] == "sz.000002"]["close"].tolist() == [20.0, 20.2, 15.075]   # 不在名单里的不动


def test_repair_adjustment_keeps_real_ex_dividend():
    # 第 2 天真实除权（10 送 10）：交易所前收盘 10，收盘 10.5，真实收益 +5%
    df = frame(open=[20.0, 21.0], high=[20.0, 21.0], low=[20.0, 21.0], close=[20.0, 21.0], close_raw=[20.0, 10.5])
    ref = pd.DataFrame({"date": df["date"], "code": "sz.000001", "close": [20.0, 10.5],
                        "preclose": [19.0, 10.0], "tradestatus": [1, 1]})
    out = clean.repair_adjustment(df, ref)
    assert np.allclose(out["close"], [20.0, 21.0])
