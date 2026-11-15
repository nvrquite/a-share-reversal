"""涨跌停价规则的测试（先于实现写成）。"""
import numpy as np
import pandas as pd
import pytest

from src.data import limits

OLD = "2015-01-05"   # 一个很早的上市日，代表老股票


def one(code, date, preclose, is_st=0, ipo=OLD, day_no=999):
    """算一只股票一天的（涨停价, 跌停价）。"""
    df = pd.DataFrame({"code": [code], "date": [pd.Timestamp(date)], "preclose_raw": [preclose],
                       "isST": [is_st], "ipo_date": [pd.Timestamp(ipo)], "day_no": [day_no]})
    out = limits.add_limits(df)
    return out["up_limit"].iloc[0], out["down_limit"].iloc[0]


def test_board_of():
    codes = pd.Series(["sh.600000", "sh.688001", "sz.000001", "sz.002001", "sz.300001", "sz.301001", "sz.302132"])
    assert limits.board_of(codes).tolist() == ["main", "star", "main", "main", "chinext", "chinext", "chinext"]


def test_main_board_ten_percent():
    assert one("sh.600000", "2021-03-01", 10.00) == (11.00, 9.00)


def test_round_half_up_not_bankers():
    # 10.05*1.1=11.055 → 11.06；10.05*0.9=9.045 → 9.05（四舍五入，不是银行家舍入）
    assert one("sz.000001", "2021-03-01", 10.05) == (11.06, 9.05)
    # 浮点误差的典型例子：4.35*1.1 在浮点里是 4.785000000000001 或 4.7849999
    assert one("sz.000001", "2021-03-01", 4.35) == (4.79, 3.92)


def test_main_st_before_and_after_2026_07_06():
    assert one("sh.600001", "2026-07-03", 2.33, is_st=1) == (2.45, 2.21)
    assert one("sh.600001", "2026-07-06", 2.33, is_st=1) == (2.56, 2.10)


def test_s_share_is_five_percent_even_when_not_st():
    assert one("sh.600182", "2024-03-01", 10.00, is_st=0) == (10.50, 9.50)


def test_chinext_switch_on_2020_08_24():
    assert one("sz.300001", "2020-08-21", 20.00) == (22.00, 18.00)
    assert one("sz.300001", "2020-08-24", 20.00) == (24.00, 16.00)


def test_chinext_st_switch():
    assert one("sz.300001", "2020-08-21", 2.00, is_st=1) == (2.10, 1.90)
    assert one("sz.300001", "2020-08-24", 2.00, is_st=1) == (2.40, 1.60)


def test_star_always_twenty_percent_even_st():
    assert one("sh.688001", "2020-03-02", 50.00) == (60.00, 40.00)
    assert one("sh.688001", "2024-03-01", 5.00, is_st=1) == (6.00, 4.00)


@pytest.mark.parametrize("code,ipo,date,day_no,limited", [
    ("sh.688001", "2021-06-01", "2021-06-07", 5, False),   # 科创板前 5 日无限制
    ("sh.688001", "2021-06-01", "2021-06-08", 6, True),
    ("sz.301001", "2020-08-24", "2020-08-28", 5, False),   # 注册制创业板新股
    ("sz.300999", "2020-07-01", "2020-07-01", 1, False),   # 核准制创业板新股：首日基数是发行价，不算
    ("sz.300999", "2020-07-01", "2020-07-02", 2, True),
    ("sh.601001", "2023-04-10", "2023-04-14", 5, False),   # 全面注册制后的主板新股
    ("sh.601001", "2023-04-10", "2023-04-17", 6, True),
    ("sh.601002", "2022-05-10", "2022-05-10", 1, False),   # 核准制主板新股首日
    ("sh.601002", "2022-05-10", "2022-05-11", 2, True),
])
def test_new_stock_days(code, ipo, date, day_no, limited):
    up, down = one(code, date, 10.00, ipo=ipo, day_no=day_no)
    assert (not np.isnan(up)) == limited
    assert (not np.isnan(down)) == limited


def test_missing_preclose_gives_nan():
    up, down = one("sh.600000", "2021-03-01", np.nan)
    assert np.isnan(up) and np.isnan(down)


def test_violation_flags():
    df = pd.DataFrame({"high_raw": [11.00, 11.02, 10.5], "low_raw": [9.00, 9.5, 8.98],
                       "up_limit": [11.00, 11.00, np.nan], "down_limit": [9.00, 9.00, np.nan]})
    assert limits.violations(df).tolist() == [False, True, False]
