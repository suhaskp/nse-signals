"""Shared chart helpers for the dashboard views."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config.config import BreakoutConfig
from screeners.breakout import breakout_frame

C = {"ink": "#1B2A41", "muted": "#5B6B82", "panel": "#FFFFFF", "grid": "#D8DEE7",
     "long": "#1F7A5C", "stop": "#B23A48", "watch": "#7A8699"}


def range_chart(history: pd.DataFrame, cfg: BreakoutConfig, plan: pd.Series | None = None,
                sessions: int = 250) -> go.Figure:
    """Candles with the prior N-day range, past BUY/SELL markers and (optionally) today's stop/target."""
    df = history.tail(sessions)
    f = breakout_frame(history, cfg).loc[df.index]
    fig = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                   increasing=dict(line=dict(color=C["long"]), fillcolor=C["long"]),
                                   decreasing=dict(line=dict(color=C["stop"]), fillcolor=C["stop"]), name="Price"))
    fig.add_scatter(x=f.index, y=f["prior_high"], name=f"Prior {cfg.lookback}-day high",
                    line=dict(color=C["ink"], width=1.2, dash="dot"))
    fig.add_scatter(x=f.index, y=f["prior_low"], name=f"Prior {cfg.lookback}-day low",
                    line=dict(color=C["muted"], width=1.2, dash="dot"))
    for flag, sym, color, name in (("buy", "triangle-up", C["long"], "BUY signal"),
                                   ("sell", "triangle-down", C["stop"], "SELL signal")):
        pts = f[f[flag]]
        if not pts.empty:
            fig.add_scatter(x=pts.index, y=pts["close"], mode="markers", name=name,
                            marker=dict(symbol=sym, size=12, color=color, line=dict(color="white", width=1)))
    if plan is not None and pd.notna(plan.get("Stop Loss")):
        x0, x1 = df.index[-20], df.index[-1] + pd.Timedelta(days=5)
        for y, label, color in ((plan["Target Price"], "Target", C["long"]), (plan["Close"], "Price", C["ink"]),
                                (plan["Stop Loss"], "Stop", C["stop"])):
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=color, width=1.5))
            fig.add_annotation(x=x1, y=y, text=f"{label} ₹{y:,.2f}", showarrow=False, xanchor="left",
                               xshift=6, font=dict(color=color, size=12))
    fig.update_layout(height=500, plot_bgcolor=C["panel"], paper_bgcolor=C["panel"], hovermode="x unified",
                      font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=C["ink"]),
                      xaxis=dict(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])],
                                 gridcolor=C["grid"]),
                      yaxis=dict(tickprefix="₹", gridcolor=C["grid"]), margin=dict(l=10, r=110, t=10, b=10),
                      legend=dict(orientation="h", y=1.06))
    return fig
