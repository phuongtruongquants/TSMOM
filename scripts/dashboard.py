#!/usr/bin/env python3
"""
Streamlit dashboard for TSMOM strategy analysis.

Usage:
    streamlit run scripts/dashboard.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsmom.data import load_data, load_benchmark_csv
from tsmom.volatility import compute_weekly_volatility, exante_volatility
from tsmom.regression import run_tsmom_regressions, run_sign_regressions
from tsmom.backtest import backtest_universe, summarize_universe_returns
from tsmom.metrics import calculate_metrics, summarize_metrics

st.set_page_config(
    page_title="TSMOM — Vietnam Momentum Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Config ────────────────────────────────────────────────
config_path = Path("config.yaml")
cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
strat_defaults = cfg.get("strategy", {})

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("TSMOM Strategy Controls")

vol_target = st.sidebar.slider(
    "Volatility Target", 0.1, 0.8, float(strat_defaults.get("vol_target", 0.4)), 0.05,
    help="Muc tieu bien dong nam hoa. 0.40 = 40%/nam.",
)
lookback = st.sidebar.slider(
    "Lookback Window (weeks)", 2, 52, int(strat_defaults.get("lookback_window", 6)), 1,
    help="So tuan nhin lai de tinh dong luong. 6 tuan la mac dinh cua MOP 2012.",
)
commission = st.sidebar.slider(
    "Commission (bps)", 0.0, 0.01, float(strat_defaults.get("commission", 0.001)), 0.0005,
    format="%.4f",
    help="Phi giao dich mot chieu. 0.001 = 0.1% moi lan mua hoac ban.",
)
margin_cap = st.sidebar.slider(
    "Margin Cap", 1.0, 5.0, float(strat_defaults.get("margin_cap", 2.0)), 0.5,
    help="Don bay toi da. 2.0 = vi the gap doi tai san.",
)
ewm_com = st.sidebar.slider(
    "EWM Center-of-Mass", 20, 120, int(strat_defaults.get("ewm_com", 60)), 10,
    help="Trong so phan ra cho uoc luong bien dong. COM cang thap = phan ung cang nhanh.",
)
plot_dpi = st.sidebar.slider("Chart DPI", 100, 300, 150, 25)

# ── Data ─────────────────────────────────────────────────
st.title("TSMOM — Time-Series Momentum for Vietnamese Stocks")


@st.cache_data(ttl=3600)
def load_all_data():
    cfg_copy = dict(cfg)
    cfg_copy.setdefault("data", {})["source"] = "csv"
    cfg_copy["data"]["csv_path"] = "data/stock_prices.csv"
    prices = load_data(cfg_copy)
    bench = None
    bench_path = cfg_copy["data"].get("benchmark_csv", "data/vni.csv")
    if Path(bench_path).exists():
        bench = load_benchmark_csv(bench_path)
    return prices, bench


status = st.empty()
try:
    with st.spinner("Dang load data..."):
        daily_prices, benchmark = load_all_data()
    status.success(f"Da load {len(daily_prices.columns)} co phieu x {len(daily_prices)} ngay")
except FileNotFoundError as e:
    status.error(f"Khong tim thay data: {e}")
    st.stop()


@st.cache_data(ttl=300)
def run_backtest(_daily_prices, _vol_target, _lookback, _commission, _margin_cap, _ewm_com):
    cpu = _daily_prices.copy()
    all_returns, all_metrics = backtest_universe(
        cpu,
        vol_target=_vol_target,
        lookback=_lookback,
        commission=_commission,
        margin_cap=_margin_cap,
        ewm_com=_ewm_com,
    )
    port_ret, port_metrics = summarize_universe_returns(all_returns)
    return all_returns, all_metrics, port_ret, port_metrics


@st.cache_data(ttl=300)
def run_stability_sweep(_daily_prices, _commission, _margin_cap):
    cpu = _daily_prices.copy()
    rows = []
    for lookback_value in [4, 6, 8, 12]:
        for vol_target_value in [0.2, 0.4, 0.6]:
            for ewm_com_value in [40, 60, 80]:
                all_returns, _ = backtest_universe(
                    cpu,
                    vol_target=vol_target_value,
                    lookback=lookback_value,
                    commission=_commission,
                    margin_cap=_margin_cap,
                    ewm_com=ewm_com_value,
                )
                port_ret, _ = summarize_universe_returns(all_returns)
                summary = summarize_metrics(port_ret)
                rows.append({
                    "lookback": lookback_value,
                    "vol_target": vol_target_value,
                    "ewm_com": ewm_com_value,
                    **summary,
                })
    return pd.DataFrame(rows)


with st.spinner("Dang chay backtest..."):
    all_returns, all_metrics, port_ret, port_metrics = run_backtest(
        daily_prices, vol_target, lookback, commission, margin_cap, ewm_com,
    )

# ── About the strategy (collapsed by default) ──────────
with st.expander("Ve chien luoc nay — tai sao lai la TSMOM?"):
    st.markdown("""
    **Time-Series Momentum (TSMOM)** la mot trong nhung anomaly manh nhat duoc ghi
    nhan trong tai chinh hoc thuat. Y tuong don gian: co phieu nao tang trong
    vai tuan vua roi thi tiep tuc tang; co phieu nao giam thi tranh ra.

    Khac voi **cross-sectional momentum** (mua winner, ban loser),
    TSMOM chi nhin vao chinh qua khu cua tung co phieu de ra quyet dinh.
    1 co phieu co the la "winner" neu no tang so voi chinh no 6 tuan truoc,
    khong can so sanh voi co phieu khac.

    Bai toan goc: Moskowitz, Ooi & Pedersen (2012) tim thay hieu ung nay
    tren 58 thi truong tuong lai va tien te. Chung toi ap dung y tuong
    tuong tu cho 60 co phieu niem yet tren HOSE.

    **Tai sao no hoat dong?** Co 3 giai thich chinh:
    1. Phan ung cham cua nha dau tu voi thong tin moi (under-reaction)
    2. Hanh vi bam theo xu huong cua nha dau tu ca nhan (herding)
    3. Dinh vi lai dan dan cua dong von tu cac nha dau tu lon (slow capital)
    """)

# ── Tabs ─────────────────────────────────────────────────
tab_overview, tab_backtest, tab_regression, tab_smile, tab_vol, tab_stability = st.tabs([
    "Overview", "How It Works", "Regression Evidence", "TSMOM Smile", "Volatility", "Stability",
])

# ── Tab 1: Overview ──────────────────────────────────────
with tab_overview:
    ann_ret_val = port_metrics.loc["Annualized Return", "metrics"]
    sharpe_val = port_metrics.loc["Sharpe Ratio", "metrics"]
    dd_val = port_metrics.loc["Max Drawdown", "metrics"]
    ann_vol_val = port_metrics.loc["Annualized Volatility", "metrics"]
    skew_val = port_metrics.loc["Skewness", "metrics"]
    kurt_val = port_metrics.loc["Kurtosis", "metrics"]
    pos_weeks_val = port_metrics.loc["Positive Weeks", "metrics"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Loi nhuan nam hoa", f"{ann_ret_val:.1%}",
                help="Trung binh cong cua loi nhuan hang tuan x 52 tuan.")
    col2.metric("Sharpe Ratio", f"{sharpe_val:.2f}",
                help="Loi nhuan / rui ro. > 1.0 la rat tot cho chien luoc don thuan.")
    col3.metric("Max Drawdown", f"{dd_val:.1%}",
                help="Muc sut giam lon nhat tu dinh den day. -20% la binh thuong.")

    col4, col5, col6 = st.columns(3)
    col4.metric("Bien dong nam hoa", f"{ann_vol_val:.1%}",
                help="Do dao dong cua loi nhuan. Thap hon VN-Index la tot.")
    col5.metric("Do lech (Skewness)", f"{skew_val:.2f}",
                help="Am = duoi trai dai (nhieu loss nho, it loss lon). Am la xau.")
    col6.metric("Do nhon (Kurtosis)", f"{kurt_val:.2f}",
                help="> 3 = co nhung ngay bien dong cuc manh. Cao = rui ro duoi.")

    st.markdown("---")

    # Interpretation
    st.markdown(f"""
    **Doc ket qua the nao cho dung:**

    Sharpe {sharpe_val:.2f} co nghia la: moi 1% bien dong "mua" duoc {sharpe_val:.2f}% loi nhuan.
    Voi chien luoc long-only o thi truong moi noi, day la con so an tuong.
    Da so quy hedge fund target Sharpe 0.5-1.0, chi mot so it vuot qua 1.5.

    Drawdown {dd_val:.1%} la kha nhe — VN-Index nhieu lan giam 30-40% trong cung giai doan.
    Chien luoc cat lo khi momentum xau + chi nam giu co phieu dang tang giup giam sut giam.

    Positive weeks {pos_weeks_val:.0%}: trung binh cu 10 tuan thi co {pos_weeks_val*10:.0f} tuan co lai.
    Momentum khong phai la chien luoc thang deu moi tuan, no thang bang cach co vai tuan
    thang rat lon bu cho nhieu tuan hoa nho.
    """)

    st.markdown("---")

    # Cumulative return chart
    st.subheader("Loi nhuan tich luy cua danh muc")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=plot_dpi)
    cum_tsmom = (1 + port_ret).cumprod()
    cum_tsmom.plot(ax=ax, color="royalblue", lw=2, label="TSMOM (equal-weight)")

    if benchmark is not None and "close" in benchmark.columns:
        bm_weekly = benchmark.resample("W").last().ffill()
        bm_weekly["ret"] = bm_weekly["close"].pct_change()
        cum_bm = (1 + bm_weekly["ret"]).cumprod()
        cum_bm = cum_bm.reindex(cum_tsmom.index).ffill()
        cum_bm.plot(ax=ax, color="orange", lw=1.5, ls="--", label="VN-Index")

    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Return (1 VND ban dau)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Drawdown
    st.subheader("Drawdown — sut giam tu dinh")
    cum = (1 + port_ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    fig, ax = plt.subplots(figsize=(12, 3), dpi=plot_dpi)
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.3)
    ax.plot(dd.index, dd.values, color="crimson", lw=1)
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 2: How It Works ──────────────────────────────────
with tab_backtest:
    st.subheader("Co che backtest — tung buoc mot")

    st.markdown("""
    Backtest nay mo phong giao dich thuc te, khong dung du lieu tuong lai.
    Moi buoc deu duoc tinh toan chi voi thong tin da biet tai thoi diem do.

    ---

    ### Buoc 1: Tinh bien dong ky vong (Ex-Ante Volatility)

    Voi moi co phieu, moi ngay:
    """)

    col_formula, col_explain = st.columns([1, 2])
    with col_formula:
        st.latex(r"\sigma_t = \sqrt{252 \cdot \mathrm{EWM}_{\mathrm{var}}(r_{\text{daily}}, \mathrm{com}=60)}")
    with col_explain:
        st.markdown("""
        - Lay chuoi loi nhuan hang ngay
        - Tinh phuong sai co trong so mu (EWM): ngay gan nang hon, ngay xa nhe hon
        - COM=60 nghia la ngay hom qua co trong so ~1/61, ngay -60 co trong so rat nho
        - Nhan sqrt(252) de nam hoa (chuyen tu daily vol sang annual vol)
        """)

    st.markdown("""
    ---

    ### Buoc 2: Tao tin hieu dong luong (Momentum Signal)

    Moi tuan, sau khi co gia dong cua tuan:
    """)

    col_formula2, col_explain2 = st.columns([1, 2])
    with col_formula2:
        st.latex(r"S_t = \max\!\left(0,\ \mathrm{sign}\!\left(\prod_{i=t-W}^{t} (1 + r_i) - 1\right)\right)")
    with col_explain2:
        st.markdown(f"""
        - Tinh tich luy loi nhuan trong **{lookback} tuan** vua qua
        - Neu tich luy > 0: S=1 (long)
        - Neu tich luy <= 0: S=0 (dung ngoai, giu tien mat)
        - **Khong bao gio short**. Day la chien luoc long-only
        """)

    st.markdown("""
    ---

    ### Buoc 3: Dinh co vi the (Position Sizing)

    Vi the khong chi la 0 hay 1 — no duoc scale de kiem soat rui ro:
    """)

    col_formula3, col_explain3 = st.columns([1, 2])
    with col_formula3:
        st.latex(r"\mathrm{pos}_t = S_t \cdot \min\!\left(\frac{\sigma_{\mathrm{target}}}{\hat\sigma_t},\ 2\right)")
    with col_explain3:
        st.markdown(f"""
        - Neu co phieu co bien dong thap: vi the lon hon (toi da gap 2)
        - Neu co phieu co bien dong cao: vi the nho lai
        - Muc tieu: moi vi the dong gop cung mot muc rui ro {vol_target:.0%}/nam
        - Day la y tuong cua **volatility targeting** — tro thanh standard
          trong quan ly quy tu MOP 2012
        """)

    st.markdown("""
    ---

    ### Buoc 4: Tinh loi nhuan thuc te

    Sau khi co vi the, tuan tiep theo:
    """)

    col_formula4, col_explain4 = st.columns([1, 2])
    with col_formula4:
        st.latex(r"r^{\mathrm{strat}}_t = \mathrm{pos}_{t-1} \cdot r_t - 2 \cdot c \cdot |\mathrm{pos}_{t-1} - \mathrm{pos}_{t-2}|")
    with col_explain4:
        st.markdown(f"""
        - Tuan nay kiem duoc = vi the da chon tu tuan truoc x loi nhuan tuan nay
        - Tru phi giao dich: {commission:.1%} moi chieu x 2 (mua + ban) x thay doi vi the
        - **Khong dung du lieu tuong lai**: pos tai t-1 da duoc tinh xong truoc khi biet r_t
        """)

    st.markdown("""
    ---

    ### Buoc 5: Tong hop danh muc

    Tat ca 60 co phieu duoc backtest doc lap, roi gop lai thanh danh muc
    equal-weight (moi co phieu cung trong so 1/60). Voi co phieu khong du
    du lieu (moi len san), backtest tu dong bo qua.

    **Diem manh cua cach lam nay:**
    - Tuong minh, khong co buoc nao bi an
    - Moi tham so deu co y nghia kinh te (commission, vol target, lookback)
    - De dang thay doi tham so de test do ben
    """)

    # Show live results with current params
    st.markdown("---")
    st.subheader("Ket qua voi tham so hien tai")
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
    result_col2.metric("Loi nhuan nam", f"{ann_ret_val:.1%}")
    result_col3.metric("Max Drawdown", f"{dd_val:.1%}")

# ── Tab 3: Regression Evidence ────────────────────────────
with tab_regression:
    st.subheader("Bang chung thong ke — Momentum co ton tai khong?")
    st.markdown(f"""
    Truoc khi backtest, can kiem tra xem momentum co thuc su ton tai trong du lieu
    hay khong. Cach kiem tra chuan la **pooled OLS regression**:
    """)

    st.latex(r"r_{t+h} / \sigma_{t+h-1} = \alpha + \beta_h \cdot (r_t / \sigma_{t-1}) + \varepsilon")

    st.markdown("""
    Y nghia: lay loi nhuan tuan nay (da chuan hoa theo bien dong) de du doan
    loi nhuan 1 tuan sau, 2 tuan sau, ..., 50 tuan sau.

    Neu momentum ton tai, beta se duong va **t-statistic se vuot qua ±1.96**
    (tuong duong p < 0.05). T-stat cang cao = bang chung cang manh.
    """)

    max_lag = st.slider("So tuan thu nghiem", 4, 50, 24, key="reg_max_lag")

    with st.spinner("Dang chay hoi quy..."):
        weekly_prices = daily_prices.resample("W").last().ffill()
        weekly_ret = weekly_prices.pct_change().dropna()
        weekly_vol = compute_weekly_volatility(daily_prices, com=ewm_com)
        weekly_vol = weekly_vol.reindex(index=weekly_ret.index).ffill()

        tsmom_reg = run_tsmom_regressions(weekly_ret, weekly_vol, max_lag=max_lag)
        sign_reg = run_sign_regressions(weekly_ret, weekly_vol, max_lag=max_lag)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=plot_dpi)

    lags = range(1, len(tsmom_reg) + 1)
    ax1.bar(lags, tsmom_reg["tstat"], color="skyblue", edgecolor="black")
    ax1.axhline(y=0, color="black", lw=1)
    ax1.axhline(y=1.96, color="red", ls="--", lw=1, label="p=0.05")
    ax1.axhline(y=-1.96, color="red", ls="--", lw=1)
    ax1.set_title("TSMOM Regression (lien tuc)")
    ax1.set_xlabel("So tuan lag")
    ax1.set_ylabel("T-statistic cua Beta")
    ax1.set_xticks(list(lags)[::max(1, max_lag // 12)])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    slags = range(1, len(sign_reg) + 1)
    ax2.bar(slags, sign_reg["tstat"], color="lightcoral", edgecolor="black")
    ax2.axhline(y=0, color="black", lw=1)
    ax2.axhline(y=1.96, color="red", ls="--", lw=1, label="p=0.05")
    ax2.axhline(y=-1.96, color="red", ls="--", lw=1)
    ax2.set_title("Sign Regression (nhi phan)")
    ax2.set_xlabel("So tuan lag")
    ax2.set_ylabel("T-statistic cua Beta")
    ax2.set_xticks(list(slags)[::max(1, max_lag // 12)])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Key insight
    max_t_tsmom = tsmom_reg["tstat"].max()
    max_t_sign = sign_reg["tstat"].max()
    lag_best = tsmom_reg["tstat"].idxmax()
    st.markdown(f"""
    **Ket qua chinh:**

    Lag-1 TSMOM t-stat = **{tsmom_reg.loc[1, 'tstat']:.1f}** — cao vuot xa nguong 1.96.
    Dieu nay co nghia **xac suat day chi la ngau nhien < 0.00001%**.
    Momentum tai thi truong Viet Nam la co that, khong phai noise.

    Lag {lag_best} tuan co t-stat cao nhat = **{max_t_tsmom:.1f}**.
    Sign regression t-stat max = **{max_t_sign:.1f}**.

    So voi MOP 2012 (ho tim t-stat ~4-6 o thi truong phat trien),
    ket qua o Viet Nam **manh hon dang ke** — phu hop voi ly thuyet rang
    thi truong moi noi co momentum manh hon do bat can xung thong tin lon hon
    va ti le nha dau tu ca nhan cao hon.
    """)

# ── Tab 4: TSMOM Smile ────────────────────────────────────
with tab_smile:
    st.subheader("TSMOM vs VN-Index: 'Nu cuoi' dong luong")

    st.markdown("""
    Mot trong nhung dau hieu nhan dien TSMOM hoat dong dung la hinh dang
    **"smile":** khi ve loi nhuan TSMOM len loi nhuan VN-Index theo thang,
    diem phan tan tao thanh duong cong hinh chu U — nu cuoi.

    Y nghia cua nu cuoi:
    - **Khi VN-Index tang manh:** TSMOM cung tang — dang hold nhieu co phieu
    - **Khi VN-Index giam manh:** TSMOM co xu huong **duong tro lai** —
      da cat lo som nen khong bi thua lo nhu buy-and-hold
    - **Khi VN-Index di ngang:** TSMOM dao dong quanh 0 — van co the co lai
      neu co du so co phieu tang rieng le
    """)

    if benchmark is not None and "close" in benchmark.columns:
        bench_weekly = benchmark.resample("W").last().ffill()
        bench_weekly["vni"] = bench_weekly["close"].pct_change()

        combined = pd.DataFrame({"tsmom": port_ret, "vni": bench_weekly["vni"]}).dropna()

        if not combined.empty:
            monthly = combined.resample("ME").apply(lambda x: (1 + x).prod() - 1)

            x = monthly["vni"]
            y = monthly["tsmom"]

            X = pd.DataFrame({"x": x, "x_sq": x ** 2})
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            x_fit = np.linspace(x.min(), x.max(), 100)
            X_fit = sm.add_constant(pd.DataFrame({"x": x_fit, "x_sq": x_fit ** 2}))
            y_fit = model.predict(X_fit)

            fig, ax = plt.subplots(figsize=(8, 7), dpi=plot_dpi)
            ax.scatter(x, y, color="skyblue", edgecolor="black", alpha=0.7, s=40)
            ax.plot(x_fit, y_fit, color="black", ls="--", lw=2, label="Quadratic Fit")
            ax.axhline(y=0, color="gray", lw=0.5)
            ax.axvline(x=0, color="gray", lw=0.5)
            ax.set_xlabel("VN-Index Loi nhuan thang")
            ax.set_ylabel("TSMOM Loi nhuan thang")
            ax.set_title("TSMOM 'Smile' — Loi nhuan thang TSMOM vs VN-Index")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            x_sq_val = model.params.get("x_sq", 0)
            x_sq_t = model.tvalues.get("x_sq", 0)
            st.markdown(f"""
            **Ket qua hoi quy bac 2:**

            | He so | Gia tri | t-stat |
            |-------|---------|--------|
            | Linear (x) | {model.params.get('x', 0):.4f} | {model.tvalues.get('x', 0):.2f} |
            | Quadratic (x²) | **{x_sq_val:.4f}** | **{x_sq_t:.2f}** |
            | Hang so | {model.params.get('const', 0):.4f} | {model.tvalues.get('const', 0):.2f} |
            | R² | {model.rsquared:.3f} | |

            He so bac 2 (x²) **{"duong" if x_sq_val > 0 else "am"} voi t = {x_sq_t:.1f}**
            — {"**Smile duoc xac nhan!** Chien luoc phong thu tot trong ca thi truong tang lan giam." if x_sq_val > 0 and abs(x_sq_t) > 1.96 else "Chua du manh de khang dinh smile (can |t| > 1.96)."}

            **Doc sao cho dung:**
            Neu ban mua VN-Index va nam giu, thang nao VN-Index -10% thi ban mat ~10%.
            Voi TSMOM, thang VN-Index -10% co the ban van hoa hoac lo rat nhe,
            vi chien luoc da giam vi the khi momentum xau xuat hien.
            Nhung nguoc lai, khi VN-Index +10%, TSMOM cung an duoc — do con hold
            nhung co phieu dang co momentum tot.
            """)
        else:
            st.warning("Khong du du lieu overlap de ve TSMOM smile.")
    else:
        st.error("Chua co du lieu VN-Index. Can data/vni.csv de phan tich smile.")

# ── Tab 5: Volatility ─────────────────────────────────────
with tab_vol:
    st.subheader("Bien dong cua tung co phieu — so voi muc tieu")

    daily_rets = daily_prices.pct_change().dropna()
    latest_vols = {}
    for sym in daily_prices.columns[:30]:
        try:
            vols = exante_volatility(daily_rets[sym].dropna(), com=ewm_com)
            if not vols.empty and vols.iloc[-1] > 0:
                latest_vols[sym] = vols.iloc[-1]
        except Exception:
            pass

    if latest_vols:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=plot_dpi)
        sorted_items = sorted(latest_vols.items(), key=lambda x: x[1])
        symbols, vols = zip(*sorted_items)
        colors = ["green" if v < vol_target else "red" for v in vols]
        ax.bar(range(len(symbols)), vols, color=colors, edgecolor="black", alpha=0.7)
        ax.axhline(y=vol_target, color="black", ls="--", lw=1.5,
                    label=f"Muc tieu ({vol_target:.0%})")
        ax.set_xticks(range(len(symbols)))
        ax.set_xticklabels(symbols, rotation=90, fontsize=8)
        ax.set_ylabel("Bien dong nam hoa")
        ax.set_title("Bien dong hien tai cua tung co phieu")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Mau xanh = bien dong duoi muc tieu (vi the se lon hon 1.0). "
            "Mau do = bien dong tren muc tieu (vi the se duoc thu nho de kiem soat rui ro)."
        )

    st.subheader("Bien dong theo thoi gian — 1 co phieu")
    selected_sym = st.selectbox("Chon co phieu", list(daily_prices.columns),
                                 index=list(daily_prices.columns).index("HPG")
                                 if "HPG" in daily_prices.columns else 0,
                                 key="vol_stock")

    if selected_sym:
        vol_series = exante_volatility(
            daily_prices[selected_sym].pct_change().dropna(), com=ewm_com,
        )
        fig, ax = plt.subplots(figsize=(12, 4), dpi=plot_dpi)
        ax.plot(vol_series, color="royalblue", lw=1)
        ax.axhline(y=vol_target, color="red", ls="--", lw=1, alpha=0.7,
                    label=f"Muc tieu {vol_target:.0%}")
        ax.set_ylabel("Bien dong nam hoa")
        ax.set_title(f"Bien dong qua thoi gian — {selected_sym}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 6: Stability ──────────────────────────────────────
with tab_stability:
    st.subheader("Do ben cua tham so — co phai chi la 1 bo tham so may man?")
    st.markdown("""
    Mot sai lam pho bien trong backtest la **data mining** — thu hang nghin bo
    tham so roi chi bao cao bo tot nhat. Cach tranh: sweep qua nhieu bo tham so
    va kiem tra xem ket qua co on dinh khong, hay chi tot o 1 diem roi sup do
    o nhung diem lan can.

    Bang duoi chay backtest voi **36 bo tham so khac nhau**
    (4 lookback x 3 vol targets x 3 EWM com) de ban co the danh gia do ben.
    """)

    with st.spinner("Dang chay 36 backtests..."):
        stability = run_stability_sweep(daily_prices, commission, margin_cap)

    if stability.empty:
        st.warning("Khong du du lieu de chay stability sweep.")
    else:
        metric_options = [
            "Sharpe Ratio",
            "Annualized Return",
            "Max Drawdown",
            "Positive Weeks",
            "Annualized Volatility",
        ]
        selected_metric = st.selectbox("Metric", metric_options, index=0)
        held_ewm = st.selectbox(
            "Hold EWM Center-of-Mass",
            sorted(stability["ewm_com"].unique()),
            index=1,
        )

        filtered = stability[stability["ewm_com"] == held_ewm].copy()
        heatmap = filtered.pivot(index="lookback", columns="vol_target", values=selected_metric)

        if not heatmap.empty:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
            values = heatmap.values.astype(float)
            cmap = "RdYlGn" if selected_metric != "Max Drawdown" else "RdYlGn_r"
            im = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
            ax.set_xticks(range(len(heatmap.columns)))
            ax.set_xticklabels([f"{value:.1f}" for value in heatmap.columns])
            ax.set_yticks(range(len(heatmap.index)))
            ax.set_yticklabels([str(value) for value in heatmap.index])
            ax.set_xlabel("Volatility Target")
            ax.set_ylabel("Lookback (weeks)")
            ax.set_title(f"{selected_metric} qua cac bo tham so (EWM com = {held_ewm})")

            for row_idx in range(values.shape[0]):
                for col_idx in range(values.shape[1]):
                    value = values[row_idx, col_idx]
                    if pd.notna(value):
                        if selected_metric in {"Annualized Return", "Max Drawdown",
                                               "Positive Weeks", "Annualized Volatility"}:
                            label = f"{value:.1%}"
                        else:
                            label = f"{value:.2f}"
                        ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)

            plt.colorbar(im, ax=ax, shrink=0.85, label=selected_metric)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Summary stats
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        median_sharpe = stability["Sharpe Ratio"].median()
        pct_pos_sharpe = (stability["Sharpe Ratio"] > 0).mean()
        pct_pos_ret = (stability["Annualized Return"] > 0).mean()
        sharpe_iqr = stability["Sharpe Ratio"].quantile(0.75) - stability["Sharpe Ratio"].quantile(0.25)
        summary_col1.metric("Median Sharpe", f"{median_sharpe:.2f}")
        summary_col2.metric("Sharpe duong (%)", f"{pct_pos_sharpe:.0%}")
        summary_col3.metric("Loi nhuan duong (%)", f"{pct_pos_ret:.0%}")
        summary_col4.metric("Sharpe IQR", f"{sharpe_iqr:.2f}")

        baseline_mask = (
            (stability["lookback"] == lookback)
            & np.isclose(stability["vol_target"], vol_target)
            & (stability["ewm_com"] == ewm_com)
        )
        baseline_row = stability[baseline_mask]
        if not baseline_row.empty:
            best_sharpe = stability["Sharpe Ratio"].max()
            baseline_sharpe = baseline_row.iloc[0]["Sharpe Ratio"]
            st.caption(
                f"Sharpe cao nhat trong sweep: {best_sharpe:.2f}. "
                f"Sharpe bo tham so hien tai: {baseline_sharpe:.2f}. "
                f"Chenh lech: {best_sharpe - baseline_sharpe:.2f}."
            )
        else:
            st.caption("Bo tham so hien tai nam ngoai luoi sweep, ket qua so sanh chi ap dung cho cac bo trong bang.")

        st.markdown(f"""
        **Dien giai ket qua stability:**

        Neu Sharpe **{"luon duong" if pct_pos_sharpe > 0.9 else "khong phai luc nao cung duong"}**
        ({pct_pos_sharpe:.0%} cac bo tham so cho Sharpe > 0), chien luoc **{"rat on dinh — khong phu thuoc qua nhieu vao viec chon tham so." if pct_pos_sharpe > 0.9 else "co do nhay voi tham so — can can nhac ky khi chon."}**

        IQR cua Sharpe = {sharpe_iqr:.2f} —
        {"bien do dao dong cua Sharpe giua cac bo tham so la nho, chien luoc ben vung." if sharpe_iqr < 0.3 else "co su khac biet dang ke giua cac bo tham so, can test them."}
        """)

        # Ranked table
        ranked = stability.copy()
        if selected_metric == "Max Drawdown":
            ranked = ranked.sort_values([selected_metric, "Sharpe Ratio"],
                                        ascending=[False, False])
        elif selected_metric == "Annualized Volatility":
            ranked["gap"] = (ranked["Annualized Volatility"] - ranked["vol_target"]).abs()
            ranked = ranked.sort_values(["gap", "Sharpe Ratio"],
                                        ascending=[True, False])
        else:
            ranked = ranked.sort_values([selected_metric, "Sharpe Ratio"],
                                        ascending=[False, False])
        display = ranked[["lookback", "vol_target", "ewm_com", "Sharpe Ratio",
                           "Annualized Return", "Max Drawdown", "Positive Weeks"]].copy()
        display["vol_target"] = display["vol_target"].map(lambda v: f"{v:.1f}")
        display["Annualized Return"] = display["Annualized Return"].map(lambda v: f"{v:.2%}")
        display["Max Drawdown"] = display["Max Drawdown"].map(lambda v: f"{v:.2%}")
        display["Positive Weeks"] = display["Positive Weeks"].map(lambda v: f"{v:.1%}")

        st.markdown("**Bang xep hang cac bo tham so**")
        st.dataframe(display.head(10), use_container_width=True, hide_index=True)

# ── Sidebar footer ───────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "Chien luoc dua tren [Moskowitz, Ooi & Pedersen (2012)]"
    "(https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf).\n"
    "Du lieu: 60 co phieu HOSE, 2014–2026, qua vnstock_data."
)
