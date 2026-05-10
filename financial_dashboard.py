"""
企業財務診断ダッシュボード Pro
機能: 個別分析 / 複数銘柄比較 / 業界平均比較 / AIコメント / スクリーニング
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
import time

# =============================================
# ページ設定
# =============================================
st.set_page_config(layout="wide", page_title="企業財務診断ダッシュボード Pro", page_icon="📊")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .section-header {
        font-size: 1.05rem; font-weight: 700; letter-spacing: 0.08em;
        color: #58a6ff; border-left: 3px solid #58a6ff;
        padding-left: 10px; margin: 24px 0 12px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2230 100%);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 14px 16px; text-align: center;
    }
    .metric-label { font-size: 0.68rem; color: #8b949e; letter-spacing: 0.1em; text-transform: uppercase; }
    .metric-value { font-size: 1.35rem; font-weight: 700; color: #e6edf3; }
    .metric-sub   { font-size: 0.72rem; color: #8b949e; margin-top: 2px; }
    .ai-box {
        background: linear-gradient(135deg, #0d1f12 0%, #0d1525 100%);
        border: 1px solid #238636; border-radius: 12px;
        padding: 20px 24px; margin-top: 8px; line-height: 1.9;
        font-size: 0.92rem; color: #c9d1d9;
    }
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border: none; border-radius: 8px;
        padding: 8px 20px; font-weight: 700; width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2ea043, #3fb950); }
    .stTextInput > div > div > input {
        background-color: #161b22; color: #e6edf3;
        border: 1px solid #30363d; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# 定数
# =============================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,27,34,0.8)',
    font=dict(family='Noto Sans JP', color='#c9d1d9', size=12),
    margin=dict(t=50, b=40, l=50, r=30),
    legend=dict(bgcolor='rgba(22,27,34,0.8)', bordercolor='#30363d', borderwidth=1, font=dict(size=11))
)
C = dict(
    blue='#58a6ff', green='#3fb950', orange='#f78166', yellow='#e3b341',
    purple='#bc8cff', teal='#39d353', red='#f85149', grid='#21262d', axis='#30363d',
)
PALETTE = [C['blue'], C['green'], C['yellow'], C['purple'], C['orange'], C['teal']]

INDUSTRY_GROUPS = {
    "自動車":   ["7203", "7267", "7269", "7270", "7201", "7202"],
    "電機・精密": ["6758", "6752", "6971", "6954", "7751", "6501"],
    "商社":     ["8058", "8053", "8001", "8002", "8031"],
    "銀行":     ["8306", "8316", "8411", "8354", "8331"],
    "小売":     ["3382", "8267", "8270", "2651", "9843"],
    "通信":     ["9432", "9433", "9984", "9437"],
    "不動産":   ["8801", "8802", "3289", "8830"],
    "食品":     ["2914", "2503", "2502", "2501", "2269"],
}

# =============================================
# ヘルパー
# =============================================
def AX():
    return dict(gridcolor=C['grid'], linecolor=C['axis'], tickfont=dict(size=11))

def safe(info, key, default=0):
    v = info.get(key, default)
    return v if v is not None else default

def pick_col(df, *keys):
    for k in keys:
        if k in df.columns:
            return df[k]
    return pd.Series([0] * len(df), index=df.index)

def score_stock(info):
    per       = safe(info, 'trailingPE', 99)
    pbr       = safe(info, 'priceToBook', 99)
    roe       = safe(info, 'returnOnEquity', 0) * 100
    div       = safe(info, 'dividendYield', 0) * 100
    op_margin = safe(info, 'operatingMargins', 0) * 100
    cr        = safe(info, 'currentRatio', 0)
    eg        = safe(info, 'earningsQuarterlyGrowth', 0) * 100
    scores = {
        "割安性": min(100, max(0, (1 / max(per, 1)) * 1500)),
        "収益性": min(100, max(0, op_margin * 4)),
        "安全性": min(100, max(0, (cr / 2) * 100)),
        "成長性": min(100, max(0, eg + 50)),
        "ROE":    min(100, max(0, roe * 5)),
        "配当":   min(100, max(0, div * 20)),
    }
    return scores, sum(scores.values()) / len(scores)

# =============================================
# データ取得
# =============================================
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        if not info or not info.get('shortName'):
            return None
        return dict(
            info       = info,
            hist       = stock.history(period="2y"),
            financials = stock.financials,
            balance    = stock.balance_sheet,
            cashflow   = stock.cashflow,
            dividends  = stock.dividends,
        )
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_info_only(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

# =============================================
# 個別グラフ
# =============================================
def plot_profitability(financials, info):
    if financials is None or financials.empty:
        return None
    df  = financials.T.sort_index()
    rev = pick_col(df, 'Total Revenue', 'Revenue') / 1e8
    op  = pick_col(df, 'Operating Income', 'Operating Revenue') / 1e8
    net = pick_col(df, 'Net Income', 'Net Income Common Stockholders') / 1e8
    labels     = [str(d.year) + '年' for d in df.index]
    op_margin  = (op  / rev.replace(0, np.nan) * 100).fillna(0)
    net_margin = (net / rev.replace(0, np.nan) * 100).fillna(0)
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('売上高・利益の推移（億円）', '利益率の推移（%）'),
        column_widths=[0.55, 0.45])
    fig.add_trace(go.Bar(name='売上高',   x=labels, y=rev, marker_color=C['blue'],   opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(name='営業利益', x=labels, y=op,  marker_color=C['green'],  opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(name='純利益',   x=labels, y=net, marker_color=C['purple'], opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(name='営業利益率', x=labels, y=op_margin,
        mode='lines+markers', line=dict(color=C['green'], width=2), marker=dict(size=7)), row=1, col=2)
    fig.add_trace(go.Scatter(name='純利益率', x=labels, y=net_margin,
        mode='lines+markers', line=dict(color=C['purple'], width=2), marker=dict(size=7)), row=1, col=2)
    roe = safe(info, 'returnOnEquity', 0) * 100
    roa = safe(info, 'returnOnAssets', 0) * 100
    fig.add_annotation(text=f"ROE: {roe:.1f}%  |  ROA: {roa:.1f}%",
        xref='paper', yref='paper', x=0.5, y=-0.15, showarrow=False,
        font=dict(size=12, color=C['yellow']))
    fig.update_layout(**PLOTLY_LAYOUT, title='📈 収益性分析', barmode='group', height=400)
    fig.update_xaxes(**AX()); fig.update_yaxes(**AX())
    return fig

def plot_safety(balance, info):
    if balance is None or balance.empty:
        return None
    df     = balance.T.sort_index()
    labels = [str(d.year) + '年' for d in df.index]
    total  = pick_col(df, 'Total Assets') / 1e8
    equity = pick_col(df, 'Stockholders Equity', 'Total Stockholder Equity') / 1e8
    ca     = pick_col(df, 'Current Assets') / 1e8
    cl     = pick_col(df, 'Current Liabilities') / 1e8
    ld     = pick_col(df, 'Long Term Debt') / 1e8
    sd     = pick_col(df, 'Short Long Term Debt', 'Short Term Debt') / 1e8
    eq_r   = (equity / total.replace(0, np.nan) * 100).fillna(0)
    cr     = (ca / cl.replace(0, np.nan) * 100).fillna(0)
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=('自己資本比率（%）', '流動比率（%）', '有利子負債（億円）'))
    fig.add_trace(go.Bar(name='自己資本比率', x=labels, y=eq_r, marker_color=C['teal']),   row=1, col=1)
    fig.add_hline(y=40,  line_dash='dot', line_color=C['yellow'], annotation_text='安全40%',  row=1, col=1)
    fig.add_trace(go.Bar(name='流動比率',   x=labels, y=cr,   marker_color=C['blue']),    row=1, col=2)
    fig.add_hline(y=100, line_dash='dot', line_color=C['yellow'], annotation_text='安全100%', row=1, col=2)
    fig.add_trace(go.Bar(name='短期負債', x=labels, y=sd, marker_color=C['orange']), row=1, col=3)
    fig.add_trace(go.Bar(name='長期負債', x=labels, y=ld, marker_color=C['red']),    row=1, col=3)
    fig.update_layout(**PLOTLY_LAYOUT, title='🛡️ 安全性分析', barmode='stack', height=400)
    fig.update_xaxes(**AX()); fig.update_yaxes(**AX())
    return fig

def plot_growth(financials, info):
    if financials is None or financials.empty:
        return None
    df    = financials.T.sort_index()
    rev   = pick_col(df, 'Total Revenue', 'Revenue')
    net   = pick_col(df, 'Net Income', 'Net Income Common Stockholders')
    rev_g = rev.pct_change() * 100
    net_g = net.pct_change() * 100
    labels = [str(d.year) + '年' for d in df.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='売上高成長率', x=labels, y=rev_g,
        marker_color=[C['green'] if v >= 0 else C['red'] for v in rev_g], opacity=0.85))
    fig.add_trace(go.Scatter(name='純利益成長率', x=labels, y=net_g,
        mode='lines+markers', line=dict(color=C['yellow'], width=2.5), marker=dict(size=8)))
    fig.add_hline(y=0, line_color=C['axis'])
    qg = safe(info, 'earningsQuarterlyGrowth', 0) * 100
    fig.add_annotation(text=f"直近四半期純利益成長率: {qg:.1f}%",
        xref='paper', yref='paper', x=1, y=1.08, showarrow=False,
        font=dict(size=12, color=C['teal']), xanchor='right')
    fig.update_layout(**PLOTLY_LAYOUT, title='🚀 成長性分析（前年比 %）', height=400,
        xaxis=dict(**AX()), yaxis=dict(**AX(), title='成長率（%）'))
    return fig

def plot_valuation(info):
    per   = safe(info, 'trailingPE', 0)
    pbr   = safe(info, 'priceToBook', 0)
    div_y = safe(info, 'dividendYield', 0) * 100
    fig = make_subplots(rows=1, cols=3,
        specs=[[{"type": "indicator"}] * 3],
        subplot_titles=('PER（倍）', 'PBR（倍）', '配当利回り（%）'))
    if per:
        fig.add_trace(go.Indicator(mode='gauge+number', value=round(per, 1),
            gauge=dict(axis=dict(range=[0, 50]), bar=dict(color=C['blue']),
                steps=[dict(range=[0, 15],  color='rgba(63,185,80,0.2)'),
                       dict(range=[15, 25], color='rgba(227,179,65,0.2)'),
                       dict(range=[25, 50], color='rgba(248,81,73,0.2)')],
                threshold=dict(line=dict(color=C['yellow'], width=3), value=20)),
            number=dict(suffix='倍', font=dict(size=28))), row=1, col=1)
    if pbr:
        fig.add_trace(go.Indicator(mode='gauge+number', value=round(pbr, 2),
            gauge=dict(axis=dict(range=[0, 5]), bar=dict(color=C['purple']),
                steps=[dict(range=[0, 1], color='rgba(63,185,80,0.2)'),
                       dict(range=[1, 2], color='rgba(227,179,65,0.2)'),
                       dict(range=[2, 5], color='rgba(248,81,73,0.2)')],
                threshold=dict(line=dict(color=C['yellow'], width=3), value=1)),
            number=dict(suffix='倍', font=dict(size=28))), row=1, col=2)
    fig.add_trace(go.Indicator(mode='gauge+number', value=round(div_y, 2),
        gauge=dict(axis=dict(range=[0, 8]), bar=dict(color=C['teal']),
            steps=[dict(range=[0, 2], color='rgba(248,81,73,0.15)'),
                   dict(range=[2, 4], color='rgba(227,179,65,0.2)'),
                   dict(range=[4, 8], color='rgba(63,185,80,0.25)')],
            threshold=dict(line=dict(color=C['yellow'], width=3), value=3)),
        number=dict(suffix='%', font=dict(size=28))), row=1, col=3)
    fig.update_layout(**PLOTLY_LAYOUT, title='💰 割安性分析', height=340)
    return fig

def plot_technical(hist):
    if hist is None or hist.empty:
        return None
    df = hist.copy()
    df['MA25']  = df['Close'].rolling(25).mean()
    df['MA75']  = df['Close'].rolling(75).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28],
        vertical_spacing=0.04, shared_xaxes=True,
        subplot_titles=('株価（ローソク足）', '出来高'))
    fig.add_trace(go.Candlestick(x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='株価', increasing_line_color=C['green'], decreasing_line_color=C['red'],
        increasing_fillcolor=C['green'], decreasing_fillcolor=C['red']), row=1, col=1)
    for ma, color, label in [('MA25', C['yellow'], '25日MA'), ('MA75', C['blue'], '75日MA'), ('MA200', C['orange'], '200日MA')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=label,
            line=dict(color=color, width=1.5)), row=1, col=1)
    vol_c = [C['green'] if c >= o else C['red'] for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='出来高',
        marker_color=vol_c, opacity=0.7), row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, title='📊 株価チャート', height=550,
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(**AX()); fig.update_yaxes(**AX())
    return fig

# =============================================
# 複数比較グラフ
# =============================================
def plot_comparison_radar(tickers_data: dict):
    fig = go.Figure()
    cats = ["割安性", "収益性", "安全性", "成長性", "ROE", "配当"]
    for i, (name, info) in enumerate(tickers_data.items()):
        sc, _ = score_stock(info)
        vals  = [sc[c] for c in cats] + [sc[cats[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]], fill='toself', name=name,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2), opacity=0.85))
    fig.update_layout(**PLOTLY_LAYOUT,
        title='🔄 複数銘柄 財務スコア比較（レーダー）',
        polar=dict(
            bgcolor='rgba(22,27,34,0.8)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=C['grid'],
                tickfont=dict(size=10), color=C['axis']),
            angularaxis=dict(gridcolor=C['grid'], linecolor=C['axis'])
        ), height=500)
    return fig

def plot_comparison_bar(tickers_data: dict):
    metrics = {
        'PER（倍）':      lambda i: safe(i, 'trailingPE', 0),
        'PBR（倍）':      lambda i: safe(i, 'priceToBook', 0),
        'ROE（%）':       lambda i: safe(i, 'returnOnEquity', 0) * 100,
        '営業利益率（%）': lambda i: safe(i, 'operatingMargins', 0) * 100,
        '配当利回り（%）': lambda i: safe(i, 'dividendYield', 0) * 100,
        '流動比率':       lambda i: safe(i, 'currentRatio', 0),
    }
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(metrics.keys()),
        vertical_spacing=0.16, horizontal_spacing=0.08)
    pos = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    names  = list(tickers_data.keys())
    colors = PALETTE[:len(names)]
    for (r, c), (mn, fn) in zip(pos, metrics.items()):
        vals = [fn(info) for info in tickers_data.values()]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors, showlegend=False,
            text=[f"{v:.1f}" for v in vals], textposition='auto',
            textfont=dict(size=11)), row=r, col=c)
    fig.update_layout(**PLOTLY_LAYOUT, title='📊 主要指標 一覧比較', height=520)
    fig.update_xaxes(**AX()); fig.update_yaxes(**AX())
    return fig

def plot_comparison_price(tickers_hist: dict):
    fig = go.Figure()
    for i, (name, hist) in enumerate(tickers_hist.items()):
        if hist is None or hist.empty:
            continue
        base = hist['Close'].iloc[0]
        perf = (hist['Close'] / base - 1) * 100
        fig.add_trace(go.Scatter(x=hist.index, y=perf, name=name,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig.add_hline(y=0, line_color=C['axis'], line_dash='dot')
    fig.update_layout(**PLOTLY_LAYOUT, title='📈 株価騰落率比較（基準日=0%）',
        height=400, xaxis=dict(**AX()), yaxis=dict(**AX(), title='騰落率（%）'))
    return fig

# =============================================
# 業界比較グラフ
# =============================================
def plot_industry_comparison(target_code, peer_codes, target_name):
    all_codes = [target_code] + peer_codes
    data = []
    prog = st.progress(0)
    for i, code in enumerate(all_codes):
        t    = code + ".T" if not code.endswith(".T") else code
        info = get_info_only(t)
        if info:
            data.append({
                'name':    info.get('shortName', code),
                'is_target': (code == target_code),
                'PER':     safe(info, 'trailingPE', 0),
                'PBR':     safe(info, 'priceToBook', 0),
                'ROE':     safe(info, 'returnOnEquity', 0) * 100,
                '営業利益率': safe(info, 'operatingMargins', 0) * 100,
                '配当利回り': safe(info, 'dividendYield', 0) * 100,
            })
        prog.progress((i + 1) / len(all_codes))
        time.sleep(0.3)
    prog.empty()
    if not data:
        return None

    df      = pd.DataFrame(data)
    metrics = ['PER', 'PBR', 'ROE', '営業利益率', '配当利回り']
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics, horizontal_spacing=0.06)
    for col_idx, metric in enumerate(metrics, 1):
        bar_colors = [C['yellow'] if row['is_target'] else PALETTE[col_idx % len(PALETTE)]
                      for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df['name'].tolist(), y=df[metric].tolist(),
            marker_color=bar_colors, showlegend=False,
            text=[f"{v:.1f}" for v in df[metric]], textposition='auto',
            textfont=dict(size=9)), row=1, col=col_idx)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=f'🏭 業界内比較（黄色 = {target_name}）', height=420)
    fig.update_xaxes(tickangle=30, tickfont=dict(size=9), gridcolor=C['grid'], linecolor=C['axis'])
    fig.update_yaxes(**AX())
    return fig

# =============================================
# ルールベース 財務コメント自動生成（API不要）
# =============================================
def generate_ai_comment(info, scores, total_score, company_name, api_key=None):
    per       = safe(info, 'trailingPE', 0)
    pbr       = safe(info, 'priceToBook', 0)
    roe       = safe(info, 'returnOnEquity', 0) * 100
    op_margin = safe(info, 'operatingMargins', 0) * 100
    div_y     = safe(info, 'dividendYield', 0) * 100
    cr        = safe(info, 'currentRatio', 0)
    eg        = safe(info, 'earningsQuarterlyGrowth', 0) * 100

    lines = []

    # ── 1. 総合評価 ──────────────────────────
    if total_score >= 75:
        overall = "財務全般にわたって非常にバランスが取れており、優良企業と判断できます。"
    elif total_score >= 55:
        overall = "いくつかの強みを持つ良好な財務状況です。一部に改善余地があります。"
    elif total_score >= 40:
        overall = "財務指標は平均的な水準です。特定の項目に注意が必要です。"
    else:
        overall = "複数の財務指標に懸念点があります。慎重な判断が求められます。"
    lines.append(f"**【総合評価】** {overall}")

    # ── 2. 強み ──────────────────────────────
    strengths = []
    if per and 0 < per < 15:
        strengths.append(f"PER {per:.1f}倍と市場平均を下回る割安水準")
    if pbr and 0 < pbr < 1:
        strengths.append(f"PBR {pbr:.2f}倍と純資産価値以下で放置されている")
    if roe > 15:
        strengths.append(f"ROE {roe:.1f}%と高い資本効率を実現")
    elif roe > 10:
        strengths.append(f"ROE {roe:.1f}%と優良水準の資本効率")
    if op_margin > 15:
        strengths.append(f"営業利益率 {op_margin:.1f}%と高い収益性")
    elif op_margin > 8:
        strengths.append(f"営業利益率 {op_margin:.1f}%と安定した収益力")
    if div_y > 3:
        strengths.append(f"配当利回り {div_y:.2f}%と高い株主還元")
    if cr > 2:
        strengths.append(f"流動比率 {cr:.2f}と財務的な余裕が大きい")
    if eg > 10:
        strengths.append(f"直近四半期の利益成長率 {eg:.1f}%と好調な成長軌道")

    if strengths:
        lines.append("**【強み】** " + "、".join(strengths[:3]) + "。")
    else:
        lines.append("**【強み】** 現時点で特筆すべき突出した指標は見られません。")

    # ── 3. 懸念点 ────────────────────────────
    concerns = []
    if per and per > 30:
        concerns.append(f"PER {per:.1f}倍と割高感があり、成長期待が株価に織り込まれている")
    if pbr and pbr > 3:
        concerns.append(f"PBR {pbr:.2f}倍と資産価値対比で高い水準")
    if roe < 5 and roe > 0:
        concerns.append(f"ROE {roe:.1f}%と資本効率が低め")
    if op_margin < 3 and op_margin >= 0:
        concerns.append(f"営業利益率 {op_margin:.1f}%と収益性の薄さが気になる")
    if 0 < cr < 1:
        concerns.append(f"流動比率 {cr:.2f}と短期の支払い能力に注意が必要")
    if eg < -10:
        concerns.append(f"直近四半期の利益が {eg:.1f}%と大幅に落ち込んでいる")
    if div_y == 0:
        concerns.append("配当なし（無配）のため、インカム投資家には不向き")

    if concerns:
        lines.append("**【懸念点】** " + "、".join(concerns[:3]) + "。")
    else:
        lines.append("**【懸念点】** 現時点で重大な懸念指標は見当たりません。")

    # ── 4. 投資判断のポイント ──────────────────
    investor_types = []
    if div_y > 2.5:
        investor_types.append("配当収入を重視するインカム投資家")
    if per and per < 15 and pbr and pbr < 1.5:
        investor_types.append("割安株を狙うバリュー投資家")
    if eg > 10 and roe > 12:
        investor_types.append("成長と収益性を重視するグロース投資家")
    if cr > 1.5 and op_margin > 5:
        investor_types.append("安定性を重視する長期保有志向の投資家")
    if not investor_types:
        investor_types.append("財務改善の進捗を見守りながら慎重に判断したい投資家")

    lines.append("**【投資判断のポイント】** " + "、".join(investor_types) + "に向いている銘柄と考えられます。")
    lines.append("<br><small>※ 本コメントは財務指標のルールベース分析であり、投資助言ではありません。</small>")

    return "<br><br>".join(lines)

# =============================================
# スクリーニング
# =============================================
def run_screening(industry, per_max, pbr_max, roe_min, div_min, op_margin_min):
    tickers = INDUSTRY_GROUPS.get(industry, [])
    results = []
    prog = st.progress(0)
    for i, code in enumerate(tickers):
        info = get_info_only(code + ".T")
        if info:
            per       = safe(info, 'trailingPE', 999)
            pbr       = safe(info, 'priceToBook', 999)
            roe       = safe(info, 'returnOnEquity', 0) * 100
            div       = safe(info, 'dividendYield', 0) * 100
            op_margin = safe(info, 'operatingMargins', 0) * 100
            price     = safe(info, 'currentPrice', 0)
            _, total  = score_stock(info)
            if per <= per_max and pbr <= pbr_max and roe >= roe_min and div >= div_min and op_margin >= op_margin_min:
                results.append({
                    '銘柄': info.get('shortName', code), 'コード': code,
                    '株価': f"¥{price:,.0f}",
                    'PER': f"{per:.1f}倍", 'PBR': f"{pbr:.2f}倍",
                    'ROE': f"{roe:.1f}%", '配当利回り': f"{div:.2f}%",
                    '営業利益率': f"{op_margin:.1f}%",
                    '総合スコア': f"{total:.0f}点",
                    '_score': total,
                })
        prog.progress((i + 1) / len(tickers))
        time.sleep(0.2)
    prog.empty()
    return sorted(results, key=lambda x: x['_score'], reverse=True)

# =============================================
# サイドバー
# =============================================
with st.sidebar:
    st.markdown("### 📋 設定")
    mode = st.radio("モード選択", ["📊 個別分析", "🔄 複数比較", "🏭 業界比較", "🔍 スクリーニング"])
    api_key = None  # ルールベース生成のためAPIキー不要
    st.markdown("---")

    if mode == "📊 個別分析":
        ticker_input   = st.text_input("証券コード（例：7203）", "7203")
        ticker         = ticker_input.strip() + ".T"
        show_technical = st.checkbox("株価チャートを表示", True)
        show_ai        = st.checkbox("🤖 AIコメント生成", True)
        run_btn        = st.button("🔍 分析実行")

    elif mode == "🔄 複数比較":
        st.caption("カンマ区切りで最大6銘柄")
        tickers_input = st.text_area("証券コード", "7203, 7267, 7269")
        run_btn       = st.button("🔍 比較分析実行")

    elif mode == "🏭 業界比較":
        target_input   = st.text_input("対象銘柄コード", "7203")
        industry_group = st.selectbox("業界グループ", list(INDUSTRY_GROUPS.keys()))
        run_btn        = st.button("🔍 業界比較実行")

    elif mode == "🔍 スクリーニング":
        screen_industry = st.selectbox("業界", list(INDUSTRY_GROUPS.keys()))
        st.markdown("##### 条件設定")
        per_max       = st.slider("PER 上限（倍）",        5.0, 50.0, 20.0, 0.5)
        pbr_max       = st.slider("PBR 上限（倍）",        0.1,  5.0,  2.0, 0.1)
        roe_min       = st.slider("ROE 下限（%）",         0.0, 30.0,  8.0, 0.5)
        div_min       = st.slider("配当利回り 下限（%）",  0.0,  6.0,  1.5, 0.1)
        op_margin_min = st.slider("営業利益率 下限（%）",  0.0, 20.0,  5.0, 0.5)
        run_btn       = st.button("🔍 スクリーニング実行")

    st.markdown("---")
    st.caption("データ: Yahoo Finance\nキャッシュ: 10分\n※投資助言ではありません")

# =============================================
# タイトル
# =============================================
st.title("📊 企業財務診断ダッシュボード Pro")

# =============================================
# メイン処理
# =============================================

# ── 個別分析 ──────────────────────────────
if mode == "📊 個別分析" and run_btn:
    with st.spinner("データ取得中..."):
        d = get_stock_data(ticker)
    if not d:
        st.error("データ取得失敗。証券コードを確認してください。"); st.stop()

    info = d['info']
    scores, total = score_stock(info)
    name = info.get('shortName', ticker)

    if total >= 75:   judge = "⭐ 超優良"
    elif total >= 55: judge = "✅ 優良"
    elif total >= 40: judge = "⚠️ 普通"
    else:             judge = "❌ 要注意"

    mdata = [
        ("企業名",     name,                                                                ""),
        ("現在株価",   f"¥{safe(info,'currentPrice',0):,.0f}",                            f"52週高値 ¥{safe(info,'fiftyTwoWeekHigh',0):,.0f}"),
        ("時価総額",   f"¥{safe(info,'marketCap',0)/1e12:.2f}兆円",                       ""),
        ("PER / PBR",  f"{safe(info,'trailingPE',0):.1f}倍 / {safe(info,'priceToBook',0):.2f}倍", ""),
        ("ROE",        f"{safe(info,'returnOnEquity',0)*100:.1f}%",                        ""),
        ("配当利回り", f"{safe(info,'dividendYield',0)*100:.2f}%",                        ""),
        ("総合判定",   judge,                                                               f"スコア: {total:.0f}/100"),
    ]
    cols = st.columns(len(mdata))
    for col, (lbl, val, sub) in zip(cols, mdata):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                     f'<div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for title, fig in [
        ("📈 収益性分析", plot_profitability(d['financials'], info)),
        ("🛡️ 安全性分析", plot_safety(d['balance'], info)),
        ("🚀 成長性分析", plot_growth(d['financials'], info)),
        ("💰 割安性分析", plot_valuation(info)),
    ]:
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
        if fig: st.plotly_chart(fig, use_container_width=True)
        else:   st.info("データが取得できませんでした。")

    if show_technical:
        st.markdown('<div class="section-header">📊 株価チャート（テクニカル）</div>', unsafe_allow_html=True)
        fig = plot_technical(d['hist'])
        if fig: st.plotly_chart(fig, use_container_width=True)

    if show_ai:
        st.markdown('<div class="section-header">🤖 AI財務アナリストコメント</div>', unsafe_allow_html=True)
        with st.spinner("AIが財務データを分析中..."):
            comment = generate_ai_comment(info, scores, total, name, api_key)
        st.markdown(f'<div class="ai-box">{comment}</div>', unsafe_allow_html=True)


# ── 複数比較 ──────────────────────────────
elif mode == "🔄 複数比較" and run_btn:
    raw = [t.strip() for t in tickers_input.replace("、", ",").split(",") if t.strip()]
    if len(raw) < 2:
        st.warning("2銘柄以上入力してください。"); st.stop()

    tickers_data = {}
    tickers_hist = {}
    prog = st.progress(0)
    for i, code in enumerate(raw[:6]):
        with st.spinner(f"{code}.T 取得中..."):
            d = get_stock_data(code + ".T")
        if d:
            name = d['info'].get('shortName', code)
            tickers_data[name] = d['info']
            tickers_hist[name] = d['hist']
        prog.progress((i + 1) / len(raw)); time.sleep(0.3)
    prog.empty()

    if not tickers_data:
        st.error("データ取得失敗。"); st.stop()

    st.subheader(f"🔄 比較: {' / '.join(tickers_data.keys())}")

    # スコアサマリー表
    rows = []
    for name, info in tickers_data.items():
        sc, tot = score_stock(info)
        row = {'銘柄': name, '総合スコア': f"{tot:.0f}点"}
        row.update({k: f"{v:.0f}" for k, v in sc.items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index('銘柄'), use_container_width=True)

    st.markdown('<div class="section-header">レーダーチャート比較</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_comparison_radar(tickers_data), use_container_width=True)

    st.markdown('<div class="section-header">主要指標 一覧比較</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_comparison_bar(tickers_data), use_container_width=True)

    st.markdown('<div class="section-header">株価騰落率比較</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_comparison_price(tickers_hist), use_container_width=True)


# ── 業界比較 ──────────────────────────────
elif mode == "🏭 業界比較" and run_btn:
    target_t    = target_input.strip() + ".T"
    target_info = get_info_only(target_t)
    target_name = target_info.get('shortName', target_input) if target_info else target_input
    peers       = [p for p in INDUSTRY_GROUPS[industry_group] if p != target_input.strip()]

    st.subheader(f"🏭 {industry_group}業界内比較：{target_name}")
    with st.spinner("業界データ取得中（しばらくお待ちください）..."):
        fig = plot_industry_comparison(target_input.strip(), peers, target_name)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
        if target_info:
            st.markdown('<div class="section-header">対象企業の割安性詳細</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_valuation(target_info), use_container_width=True)
    else:
        st.error("データ取得に失敗しました。")


# ── スクリーニング ─────────────────────────
elif mode == "🔍 スクリーニング" and run_btn:
    st.subheader(f"🔍 スクリーニング — {screen_industry}業界")
    st.caption(f"条件: PER≤{per_max} / PBR≤{pbr_max} / ROE≥{roe_min}% / 配当≥{div_min}% / 営業利益率≥{op_margin_min}%")

    with st.spinner("銘柄データを取得中..."):
        results = run_screening(screen_industry, per_max, pbr_max, roe_min, div_min, op_margin_min)

    if results:
        st.success(f"✅ {len(results)}銘柄が条件を満たしました")
        display = [{k: v for k, v in r.items() if k != '_score'} for r in results]
        st.dataframe(pd.DataFrame(display).set_index('銘柄'), use_container_width=True)

        # 総合スコア棒グラフ
        names_s  = [r['銘柄'] for r in results]
        scores_s = [r['_score'] for r in results]
        fig = go.Figure(go.Bar(
            x=names_s, y=scores_s,
            marker_color=[C['teal'] if s >= 60 else C['yellow'] if s >= 40 else C['red'] for s in scores_s],
            text=[f"{s:.0f}点" for s in scores_s], textposition='auto'
        ))
        fig.update_layout(**PLOTLY_LAYOUT,
            title='📊 条件適合銘柄 総合スコア（高い順）',
            height=350, xaxis=dict(**AX()), yaxis=dict(**AX(), title='総合スコア'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("条件を満たす銘柄が見つかりませんでした。条件を緩めて再試行してください。")


# ── 初期画面 ──────────────────────────────
else:
    if not run_btn:
        st.markdown("""
        <div style="text-align:center; padding: 60px 0; color: #8b949e;">
            <div style="font-size: 3.5rem;">📊</div>
            <div style="font-size: 1.15rem; margin-top: 16px; color: #c9d1d9; font-weight: 600;">
                企業財務診断ダッシュボード Pro
            </div>
            <div style="margin-top: 20px; font-size: 0.88rem; line-height: 2.2;">
                📊 <b>個別分析</b> — 収益性・安全性・成長性・割安性 + AIコメント自動生成<br>
                🔄 <b>複数比較</b> — 最大6銘柄をレーダー・棒グラフ・騰落率で一括比較<br>
                🏭 <b>業界比較</b> — 同業他社との主要指標を並べて比較<br>
                🔍 <b>スクリーニング</b> — PER/PBR/ROE等の条件で銘柄を絞り込み
            </div>
            <div style="margin-top: 28px; font-size: 0.82rem; color: #6e7681;">
                例: トヨタ→7203 / ソニー→6758 / 任天堂→7974 / SoftBank→9984<br>
                ※ 本ツールは投資助言を目的としていません
            </div>
        </div>
        """, unsafe_allow_html=True)
