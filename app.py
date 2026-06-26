# ============================================================
# Finance & Analytics Dashboard — Streamlit App
# Ferdy Febrian Iskandar
# Projects: IDX Sectoral Performance + LQ45 Portfolio Optimizer
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Finance & Analytics Dashboard | Ferdy Febrian Iskandar",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Analysis", [
    "🏠 Overview",
    "📊 IDX Sectoral Performance",
    "💼 LQ45 Portfolio Optimizer"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Ferdy Febrian Iskandar**")
st.sidebar.markdown("📧 iskandarferdy559@gmail.com")
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/ferdy-febrian-iskandar) | [GitHub](https://github.com/Ferdyfi24) | [Medium](https://medium.com/@iskandarferdy559)")

# ── Shared Data ───────────────────────────────────────────────
SECTORS = ['Energy','Basic Material','Industrials','Non-Cyclical',
           'Cyclical','Healthcare','Finance','Technology','Infrastructure','Transportation']

RETURN_DATA = {
    2022: [62.4,18.3,12.1,4.8,-8.2,-5.1,11.3,-24.6,2.1,9.8],
    2023: [11.2,4.6,18.4,6.3,14.7,8.2,22.1,15.3,7.4,13.2],
    2024: [-8.3,2.1,6.8,5.2,9.4,3.7,8.6,4.2,3.8,6.1],
}
VOL_DATA = {
    2022: [28.4,22.1,18.6,12.3,19.8,15.2,16.4,31.2,14.8,20.1],
    2023: [24.2,18.4,16.2,11.8,17.3,13.6,14.8,26.4,13.2,17.8],
    2024: [22.6,17.2,15.4,11.2,16.8,12.9,13.6,24.8,12.6,16.4],
}
returns_df = pd.DataFrame(RETURN_DATA, index=SECTORS)
vol_df     = pd.DataFrame(VOL_DATA,    index=SECTORS)

LQ45_STOCKS = ['BBCA','BBRI','BMRI','TLKM','ASII','UNVR','ICBP','KLBF',
               'EXCL','INDF','HMSP','GGRM','ANTM','PTBA','ADRO',
               'BBNI','SMGR','INCO','ITMG','PGAS']

@st.cache_data
def generate_lq45():
    np.random.seed(42)
    n = len(LQ45_STOCKS)
    exp_ret = np.random.uniform(0.05, 0.35, n)
    vols    = np.random.uniform(0.12, 0.45, n)
    corr    = np.eye(n)
    for i in range(n):
        for j in range(i+1,n):
            c = np.random.uniform(0.1,0.6)
            corr[i,j]=corr[j,i]=c
    cov = np.outer(vols,vols)*corr
    cov = (cov+cov.T)/2
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < 0: cov += (-min_eig+1e-8)*np.eye(n)
    return pd.Series(exp_ret, index=LQ45_STOCKS), pd.DataFrame(cov, index=LQ45_STOCKS, columns=LQ45_STOCKS)

annual_ret, cov_matrix = generate_lq45()
RISK_FREE = 0.065

# ══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📈 Finance & Analytics Dashboard")
    st.markdown("**Independent Equity Research | IDX Market Analysis | Portfolio Optimization**")
    st.markdown("---")

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Sectors Analyzed", "10", "IDX sectors 2022–2024")
    col2.metric("LQ45 Stocks", "20", "Portfolio universe")
    col3.metric("MC Simulations", "5,000", "Efficient frontier")
    col4.metric("Risk-Free Rate", "6.5%", "BI Rate avg 2022–2024")

    st.markdown("---")
    st.subheader("📌 About This Dashboard")
    st.markdown("""
    | Module | Description |
    |--------|-------------|
    | 📊 **IDX Sectoral Performance** | Annual return, volatility heatmap, and risk-return scatter across 10 IDX sectors (2022–2024) |
    | 💼 **LQ45 Portfolio Optimizer** | Markowitz mean-variance optimization — efficient frontier, Max Sharpe, and Min Variance portfolios |

    **Built with:** Python · Pandas · NumPy · SciPy · Matplotlib · Seaborn · Streamlit

    > *Data based on realistic simulation of IDX sector performance trends. For educational and portfolio purposes only. Not financial advice.*
    """)

# ══════════════════════════════════════════════════════════════
# PAGE 2: IDX SECTORAL
# ══════════════════════════════════════════════════════════════
elif page == "📊 IDX Sectoral Performance":
    st.title("📊 IDX Sectoral Performance Analysis (2022–2024)")
    st.markdown("---")

    year_sel = st.multiselect("Select Year(s)", [2022,2023,2024], default=[2022,2023,2024])
    sector_sel = st.multiselect("Select Sectors", SECTORS, default=SECTORS)

    filtered_ret = returns_df.loc[sector_sel, year_sel]
    filtered_vol = vol_df.loc[sector_sel, year_sel]

    col1,col2,col3 = st.columns(3)
    col1.metric("Best 3Y Avg Return", f"{returns_df.loc[sector_sel].mean(axis=1).max():.1f}%",
                returns_df.loc[sector_sel].mean(axis=1).idxmax())
    col2.metric("Worst 3Y Avg Return", f"{returns_df.loc[sector_sel].mean(axis=1).min():.1f}%",
                returns_df.loc[sector_sel].mean(axis=1).idxmin())
    col3.metric("Lowest Volatility", f"{vol_df.loc[sector_sel].mean(axis=1).min():.1f}%",
                vol_df.loc[sector_sel].mean(axis=1).idxmin())

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 Annual Returns", "🌡️ Heatmap", "⚖️ Risk-Return"])

    with tab1:
        if year_sel:
            fig, axes = plt.subplots(1, len(year_sel), figsize=(6*len(year_sel), 6))
            if len(year_sel)==1: axes=[axes]
            for ax, yr in zip(axes, year_sel):
                data = filtered_ret[yr].sort_values()
                colors = ['#F44336' if v<0 else '#2196F3' for v in data.values]
                bars = ax.barh(data.index, data.values, color=colors, edgecolor='white')
                ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
                for bar, val in zip(bars, data.values):
                    ax.text(val+(0.5 if val>=0 else -0.5), bar.get_y()+bar.get_height()/2,
                            f'{val:.1f}%', va='center', ha='left' if val>=0 else 'right', fontsize=8)
                ax.set_title(f'{yr}', fontweight='bold')
                ax.grid(axis='x', alpha=0.3); ax.spines[['top','right']].set_visible(False)
            plt.suptitle('Annual Returns by Sector (%)', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(filtered_ret, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    linewidths=0.5, linecolor='white', annot_kws={'size':10}, ax=ax)
        ax.set_title('Return Heatmap (%)', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        avg_ret = returns_df.loc[sector_sel].mean(axis=1)
        avg_vol = vol_df.loc[sector_sel].mean(axis=1)
        fig, ax = plt.subplots(figsize=(10,6))
        sc = ax.scatter(avg_vol, avg_ret, s=180, c=avg_ret, cmap='RdYlGn', edgecolors='#333', linewidths=0.8)
        for s in avg_ret.index:
            ax.annotate(s, (avg_vol[s], avg_ret[s]), textcoords='offset points', xytext=(8,4), fontsize=9)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(avg_vol.mean(), color='steelblue', linestyle=':', linewidth=0.8, label='Avg Vol')
        ax.axhline(avg_ret.mean(), color='darkorange', linestyle=':', linewidth=0.8, label='Avg Return')
        ax.set_xlabel('Avg Volatility (%)'); ax.set_ylabel('Avg Return (%)')
        ax.set_title('Risk-Return Profile (3Y Average)', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)
        plt.colorbar(sc, ax=ax, label='Avg Return (%)')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 3: LQ45 OPTIMIZER
# ══════════════════════════════════════════════════════════════
elif page == "💼 LQ45 Portfolio Optimizer":
    st.title("💼 LQ45 Portfolio Optimizer")
    st.markdown("**Markowitz Mean-Variance Optimization | 20 LQ45 Stocks | 2022–2024**")
    st.markdown("---")

    n_sim = st.slider("Monte Carlo Simulations", 1000, 5000, 3000, 500)
    rf    = st.slider("Risk-Free Rate (%)", 4.0, 8.0, 6.5, 0.5) / 100

    n = len(LQ45_STOCKS)
    bounds      = tuple((0,1) for _ in range(n))
    constraints = {'type':'eq','fun':lambda w: np.sum(w)-1}
    w0          = np.array([1/n]*n)

    def port_perf(w):
        r = np.dot(w, annual_ret.values)
        v = np.sqrt(w @ cov_matrix.values @ w)
        return r, v, (r-rf)/v

    res_s = minimize(lambda w: -port_perf(w)[2], w0, method='SLSQP', bounds=bounds, constraints=constraints)
    res_v = minimize(lambda w: port_perf(w)[1],  w0, method='SLSQP', bounds=bounds, constraints=constraints)

    w_s = res_s.x; w_v = res_v.x
    r_s,v_s,sh_s = port_perf(w_s)
    r_v,v_v,sh_v = port_perf(w_v)

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Max Sharpe Return", f"{r_s*100:.1f}%")
    col2.metric("Max Sharpe Ratio",  f"{sh_s:.2f}")
    col3.metric("Min Var Return",    f"{r_v*100:.1f}%")
    col4.metric("Min Var Volatility",f"{v_v*100:.1f}%")

    # MC simulation
    with st.spinner(f"Running {n_sim:,} simulations..."):
        sim_r,sim_v,sim_sh = [],[],[]
        for _ in range(n_sim):
            w = np.random.dirichlet(np.ones(n))
            r,v,s = port_perf(w)
            sim_r.append(r); sim_v.append(v); sim_sh.append(s)

    tab1,tab2,tab3 = st.tabs(["🎯 Efficient Frontier","📊 Portfolio Weights","🌡️ Correlation"])

    with tab1:
        fig, ax = plt.subplots(figsize=(11,6))
        sc = ax.scatter(np.array(sim_v)*100, np.array(sim_r)*100, c=sim_sh,
                        cmap='viridis', alpha=0.4, s=8)
        plt.colorbar(sc, ax=ax, label='Sharpe Ratio')
        ax.scatter(v_s*100,r_s*100,marker='*',color='#E53935',s=400,zorder=5,label=f'Max Sharpe (SR={sh_s:.2f})')
        ax.scatter(v_v*100,r_v*100,marker='D',color='#1565C0',s=150,zorder=5,label=f'Min Variance (SR={sh_v:.2f})')
        ax.set_xlabel('Volatility (%)'); ax.set_ylabel('Expected Return (%)')
        ax.set_title('Efficient Frontier — LQ45 Universe', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.25); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        col1,col2 = st.columns(2)
        for col, w, title, color in [
            (col1, w_s, f'Max Sharpe (SR={sh_s:.2f})', '#E53935'),
            (col2, w_v, f'Min Variance (Vol={v_v*100:.1f}%)', '#1565C0')
        ]:
            with col:
                df_w = pd.DataFrame({'Stock':LQ45_STOCKS,'Weight':w*100})
                df_w = df_w[df_w['Weight']>0.5].sort_values('Weight')
                fig,ax = plt.subplots(figsize=(5,5))
                bars = ax.barh(df_w['Stock'], df_w['Weight'], color=color, edgecolor='white')
                for bar,val in zip(bars,df_w['Weight']):
                    ax.text(val+0.2, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8)
                ax.set_title(title, fontweight='bold', fontsize=10)
                ax.set_xlabel('Weight (%)')
                ax.spines[['top','right']].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        fig,ax = plt.subplots(figsize=(11,8))
        corr = cov_matrix.div(np.outer(np.sqrt(np.diag(cov_matrix)), np.sqrt(np.diag(cov_matrix))))
        mask = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    linewidths=0.4, annot_kws={'size':7}, ax=ax)
        ax.set_title('LQ45 Correlation Matrix', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.caption("*Simulated data based on realistic LQ45 return assumptions. Not financial advice.*")
