
# Play Animation Viewer (Streamlit) — no coverage version
# -------------------------------------------------------
# Simplified to remove all player coverage logic & UI.
# Keeps: tracked-only dropdowns, true field aspect, field lines,
# Parquet-backed cached loading, deep links, and robust guards.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="NFL Play Animation Viewer (No Coverage)", layout="wide")

# ------------------------------
# Team colors & helpers
# ------------------------------

colors = {
    'ARI':["#97233F","#000000","#FFB612"],
    'ATL':["#A71930","#000000","#A5ACAF"],
    'BAL':["#241773","#000000"],
    'BUF':["#00338D","#C60C30"],
    'CAR':["#0085CA","#101820","#BFC0BF"],
    'CHI':["#0B162A","#C83803"],
    'CIN':["#FB4F14","#000000"],
    'CLE':["#311D00","#FF3C00"],
    'DAL':["#003594","#041E42","#869397"],
    'DEN':["#FB4F14","#002244"],
    'DET':["#0076B6","#B0B7BC","#000000"],
    'GB' :["#203731","#FFB612"],
    'HOU':["#03202F","#A71930"],
    'IND':["#002C5F","#A2AAAD"],
    'JAX':["#101820","#D7A22A","#9F792C"],
    'KC' :["#E31837","#FFB81C"],
    'LA' :["#003594","#FFA300","#FF8200"],
    'LAC':["#0080C6","#FFC20E","#FFFFFF"],
    'LV' :["#000000","#A5ACAF"],
    'MIA':["#008E97","#FC4C02","#005778"],
    'MIN':["#4F2683","#FFC62F"],
    'NE' :["#002244","#C60C30","#B0B7BC"],
    'NO' :["#101820","#D3BC8D"],
    'NYG':["#0B2265","#A71930","#A5ACAF"],
    'NYJ':["#125740","#000000","#FFFFFF"],
    'PHI':["#004C54","#A5ACAF","#ACC0C6"],
    'PIT':["#FFB612","#101820"],
    'SEA':["#002244","#69BE28","#A5ACAF"],
    'SF' :["#AA0000","#B3995D"],
    'TB' :["#D50A0A","#FF7900","#0A0A08"],
    'TEN':["#0C2340","#4B92DB","#C8102E"],
    'WAS':["#5A1414","#FFB612"],
    'football':["#CBB67C","#663831"]
}

def ColorDistance(hex1,hex2):
    if hex1 == hex2:
        return 0
    def hex_to_rgb_array(h):
        return np.array(tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
    rgb1 = hex_to_rgb_array(hex1)
    rgb2 = hex_to_rgb_array(hex2)
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = abs(sum((2+rm,4,3-rm)*(rgb1-rgb2)**2))**0.5
    return d

def ColorPairs(team1,team2):
    c1 = colors.get(team1, ["#222222","#aaaaaa"])
    c2 = colors.get(team2, ["#444444","#cccccc"])
    if ColorDistance(c1[0],c2[0])<500:
        return {team1:[c1[0],c1[1]], team2:[c2[1],c2[0]], 'football':colors['football']}
    else:
        return {team1:[c1[0],c1[1]], team2:[c2[0],c2[1]], 'football':colors['football']}

# ------------------------------
# Animation builder (coverage-free)
# ------------------------------

def animate_play(games, tracking_df, play_df, players, gameId, playId, play_speed='regular'):
    selected_game_df = games.loc[games['gameId']==gameId].copy()
    selected_play_df = play_df.loc[(play_df['playId']==playId) & (play_df['gameId']==gameId)].copy()
    selected_tracking_df = tracking_df.loc[(tracking_df['playId']==playId)&(tracking_df['gameId']==gameId)].copy()

    if selected_play_df.empty:
        raise ValueError(f"Play metadata missing for gameId={gameId}, playId={playId}.")
    if selected_tracking_df.empty:
        raise ValueError(f"No tracking data for gameId={gameId}, playId={playId}.")

    # Sort frames
    sorted_frame_list = selected_tracking_df['frameId'].dropna().astype(int).unique()
    sorted_frame_list.sort()

    # Team colors
    team_combos = [t for t in selected_tracking_df['club'].dropna().unique() if t!='football']
    if len(team_combos) < 2:
        team_combos = [selected_game_df['homeTeamAbbr'].iloc[0], selected_game_df['visitorTeamAbbr'].iloc[0]]
    color_orders = ColorPairs(team_combos[0],team_combos[1])

    # Yard markers
    line_of_scrimmage = float(selected_play_df['absoluteYardlineNumber'].iloc[0])
    if 'playDirection' in selected_tracking_df.columns and selected_tracking_df['playDirection'].notna().any():
        play_dir = str(selected_tracking_df['playDirection'].dropna().iloc[0])
    else:
        play_dir = 'right'
    if play_dir == 'right':
        first_down_marker = line_of_scrimmage + float(selected_play_df['yardsToGo'].iloc[0])
    else:
        first_down_marker = line_of_scrimmage - float(selected_play_df['yardsToGo'].iloc[0])

    down = selected_play_df.get('down', pd.Series([None])).iloc[0]
    quarter = selected_play_df.get('quarter', pd.Series([None])).iloc[0]
    gameClock = selected_play_df.get('gameClock', pd.Series([''])).iloc[0]

    # Button speed
    fr_duration = {"fast":65, "regular":100, "slow":135}.get(play_speed, 100)

    updatemenus_dict = [{
        "buttons": [
            {"args": [None, {"frame": {"duration": fr_duration, "redraw": False},
                              "fromcurrent": True, "transition": {"duration": 0}}],
             "label": "Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate",
                               "transition": {"duration": 0}}],
             "label": "Pause", "method": "animate"}
        ],
        "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False,
        "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
    }]

    sliders_dict = {"active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 20}, "prefix": "Frame:", "visible": True, "xanchor": "right"},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": []}

    # --- Static field shapes (under all frames) ---
    field_shapes = []
    # Outer boundary
    field_shapes.append(dict(type="rect", x0=0, x1=120, y0=0, y1=53.3,
                             line=dict(color="white", width=3), fillcolor="#00B140", layer="below"))
    # 5- and 10-yard lines
    for x in np.arange(10, 120, 5):
        field_shapes.append(dict(type="line", x0=float(x), x1=float(x), y0=0, y1=53.3,
                                 line=dict(color="white", width=2.5 if x % 10 == 0 else 1),
                                 layer="below"))
    # Goal lines emphasized
    for x in [10, 110]:
        field_shapes.append(dict(type="line", x0=float(x), x1=float(x), y0=0, y1=53.3,
                                 line=dict(color="white", width=3), layer="below"))
    # Endzones
    home = selected_game_df['homeTeamAbbr'].iloc[0]
    visitor = selected_game_df['visitorTeamAbbr'].iloc[0]
    field_shapes.append(dict(type="rect", x0=0, x1=10, y0=0, y1=53.3,
                             line=dict(color="white", width=3), fillcolor=color_orders[home][0], layer="below"))
    field_shapes.append(dict(type="rect", x0=110, x1=120, y0=0, y1=53.3,
                             line=dict(color="white", width=3), fillcolor=color_orders[visitor][0], layer="below"))

    frames = []
    clubs = [c for c in selected_tracking_df['club'].dropna().unique()]
    for frameId in sorted_frame_list:
        data = []

        # Yard numbers as non-interactive text
        data.append(go.Scatter(x=np.arange(20,110,10), y=[5]*9,
                               mode='text',
                               text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                               textfont_size = 30, textfont_family = "Courier New, monospace",
                               textfont_color = "#ffffff", showlegend=False, hoverinfo='skip'))
        data.append(go.Scatter(x=np.arange(20,110,10), y=[53.5-5]*9,
                               mode='text',
                               text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                               textfont_size = 30, textfont_family = "Courier New, monospace",
                               textfont_color = "#ffffff", showlegend=False, hoverinfo='skip'))

        # LOS & first down (dashed)
        data.append(go.Scatter(x=[line_of_scrimmage,line_of_scrimmage], y=[0,53.3],
                               mode='lines', line=dict(dash='dash', color='blue'), showlegend=False, hoverinfo='skip'))
        data.append(go.Scatter(x=[first_down_marker,first_down_marker], y=[0,53.3],
                               mode='lines', line=dict(dash='dash', color='yellow'), showlegend=False, hoverinfo='skip'))

        # Entities per club (consistent order)
        for team in clubs:
            plot_df = selected_tracking_df.loc[(selected_tracking_df['club']==team) & (selected_tracking_df['frameId']==frameId)].copy()
            if plot_df.empty:
                # placeholder to keep trace order
                data.append(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=10), showlegend=False, hoverinfo='skip', name=str(team)))
                continue

            if team != 'football':
                mph = (plot_df['s'] * 2.23693629205).round(2).astype(str) + " MPH"
                hover_text_array = ("nflId:" + plot_df['nflId'].astype('Int64').astype(str) +
                                    "<br>displayName:" + plot_df['displayName'].astype(str) +
                                    "<br>Player Speed:" + mph).tolist()

                club = plot_df['club'].iloc[0]
                marker_colors = [colors.get(club, ['#888888'])[0]] * len(plot_df)

                data.append(go.Scatter(
                    x=plot_df['x'], y=plot_df['y'], mode='markers',
                    marker=dict(color=marker_colors, line=dict(width=2, color='white'), size=10),
                    name=str(team), hovertext=hover_text_array, hoverinfo='text', showlegend=False
                ))
            else:
                data.append(go.Scatter(
                    x=plot_df['x'], y=plot_df['y'], mode='markers',
                    marker=dict(color=colors['football'][0],
                                line=dict(width=2, color=colors['football'][1]),
                                size=7, symbol='diamond-wide'),
                    name='football', hoverinfo='skip', showlegend=False
                ))

        slider_step = {'args': [[str(frameId)], {'frame': {'duration': 100, 'redraw': False},
                                                 'mode': 'immediate','transition': {'duration': 0}}],
                       'label': str(frameId),'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frameId)))

    layout = go.Layout(
        autosize=True,
        height=600,
        xaxis=dict(range=[0, 120], showgrid=False, showticklabels=False, zeroline=False, constrain='domain'),
        yaxis=dict(range=[0, 53.3], showgrid=False, showticklabels=False, zeroline=False,
                   scaleanchor='x', scaleratio=1, constrain='domain'),
        plot_bgcolor='#00B140',
        title=dict(text=f"Game {gameId}, Play {playId} — {gameClock} Q{quarter}", x=0.5, xanchor='center', y=0.98, yanchor='top'),
        margin=dict(l=10, r=10, t=70, b=10),
        updatemenus=updatemenus_dict, sliders=[sliders_dict],
        shapes=field_shapes
    )

    fig = go.Figure(data=frames[0]['data'], layout=layout, frames=frames[1:])

    # Down markers
    for y_val in [0,53.3]:
        fig.add_annotation(x=first_down_marker, y=y_val, text=str(down), showarrow=False,
                           font=dict(family="Courier New, monospace", size=16, color="black"),
                           align="center", bordercolor="black", borderwidth=2, borderpad=4, bgcolor="#ff7f0e", opacity=1)

    # Endzone team labels
    fig.add_annotation(x=5, y=53.3/2, text=home, showarrow=False,
                       font=dict(family="Courier New, monospace", size=32, color="White"),
                       textangle=270)
    fig.add_annotation(x=115, y=53.3/2, text=visitor, showarrow=False,
                       font=dict(family="Courier New, monospace", size=32, color="White"),
                       textangle=90)

    return fig

# ------------------------------
# Data loading (Parquet-backed, cached)
# ------------------------------

CSV_DTYPES = {
    "gameId": "Int64", "playId": "Int64",
    "frameId": "Int64", "nflId": "Int64",
    "x": "float32", "y": "float32", "s": "float32"
}

@st.cache_data(show_spinner=False)
def load_data(data_dir: Path):
    pq_dir = data_dir / "_parquet"
    pq_dir.mkdir(exist_ok=True)

    def cvt(name):
        csv = data_dir / f"{name}.csv"
        pq  = pq_dir / f"{name}.parquet"
        if pq.exists() and csv.exists() and pq.stat().st_mtime > csv.stat().st_mtime:
            return pd.read_parquet(pq)
        if not csv.exists() and pq.exists():
            return pd.read_parquet(pq)
        df = pd.read_csv(csv, dtype=CSV_DTYPES, low_memory=False)
        df.to_parquet(pq, index=False)
        return df

    games   = cvt("games")
    plays   = cvt("plays")
    players = cvt("players")
    tracking= cvt("test_tracking_data1")
    return games, plays, players, tracking

@st.cache_data(show_spinner=False)
def build_fig_cached(games, plays, players, tracking_df, gameId, playId):
    return animate_play(games, tracking_df, plays, players, gameId, playId, play_speed='regular')

# ------------------------------
# App UI
# ------------------------------

st.sidebar.title("Controls")

# Query params (deep links)
qp = st.query_params
try:
    preset_game = int(qp.get("gameId")) if "gameId" in qp else None
except Exception:
    preset_game = None
try:
    preset_play = int(qp.get("playId")) if "playId" in qp else None
except Exception:
    preset_play = None

# Data source
data_dir_default = Path(__file__).parent / "data"
use_local = st.sidebar.toggle("Use local ./data files", value=True, help="Uncheck to upload CSVs instead.")

if use_local:
    if not data_dir_default.exists():
        st.error("Missing ./data folder. Create a 'data' folder with games.csv, plays.csv, players.csv, test_tracking_data1.csv")
        st.stop()
    games, plays, players, tracking_df = load_data(data_dir_default)
else:
    st.info("Upload four CSVs: games.csv, plays.csv, players.csv, test_tracking_data1.csv")
    g_file = st.file_uploader("games.csv", type="csv")
    p_file = st.file_uploader("plays.csv", type="csv")
    pl_file = st.file_uploader("players.csv", type="csv")
    t_file = st.file_uploader("test_tracking_data1.csv", type="csv")
    if not all([g_file, p_file, pl_file, t_file]):
        st.stop()
    games   = pd.read_csv(g_file, dtype=CSV_DTYPES, low_memory=False)
    plays   = pd.read_csv(p_file, dtype=CSV_DTYPES, low_memory=False)
    players = pd.read_csv(pl_file, dtype=CSV_DTYPES, low_memory=False)
    tracking_df = pd.read_csv(t_file, dtype=CSV_DTYPES, low_memory=False)

# Normalize ID types
for df in (games, plays, players, tracking_df):
    for col in ("gameId", "playId"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# --- Games dropdown (tracked-only) ---
tracked_game_ids = tracking_df["gameId"].dropna().astype("Int64").drop_duplicates()
games_tracked = games[games["gameId"].isin(tracked_game_ids)].copy()

if games_tracked.empty:
    st.warning("No games found in tracking data. Upload a different tracking file or check your /data folder.")
    st.stop()

def game_label_row(r):
    date = r.get("gameDate", "")
    return f"{int(r['gameId'])} — {date} — {r['visitorTeamAbbr']} @ {r['homeTeamAbbr']}"

games_tracked["label"] = games_tracked.apply(game_label_row, axis=1)
labels_sorted = games_tracked["label"].sort_values().tolist()

if preset_game and preset_game in games_tracked["gameId"].astype(int).tolist():
    idx = labels_sorted.index(games_tracked.loc[games_tracked["gameId"]==preset_game, "label"].iloc[0])
else:
    idx = 0

game_label = st.sidebar.selectbox("Game", options=labels_sorted, index=idx)
gameId = int(games_tracked.loc[games_tracked["label"] == game_label, "gameId"].iloc[0])

# Pre-filter tracking to selected game
tracking_game = tracking_df[tracking_df["gameId"] == gameId].copy()

# --- Plays dropdown (tracked-only for the selected game) ---
tracked_play_ids = tracking_game["playId"].dropna().astype("Int64").drop_duplicates()
game_plays = plays[(plays["gameId"] == gameId) & (plays["playId"].isin(tracked_play_ids))].copy()

if game_plays.empty:
    st.warning("No tracked plays for this game.")
    st.stop()

def short_desc(row):
    d = f"{int(row['playId'])} — {int(row.get('down',0))} & {int(row.get('yardsToGo',0))} — {row.get('gameClock','')}"
    txt = str(row.get('playDescription',''))[:110].replace("\n"," ")
    return f"{d} — {txt}"

game_plays["label"] = game_plays.apply(short_desc, axis=1)
play_labels = game_plays["label"].tolist()

if preset_play and preset_play in game_plays["playId"].astype(int).tolist():
    pidx = play_labels.index(game_plays.loc[game_plays["playId"]==preset_play, "label"].iloc[0])
else:
    pidx = 0

play_label = st.sidebar.selectbox("Play", options=play_labels, index=pidx)
playId = int(game_plays.loc[game_plays["label"] == play_label, "playId"].iloc[0])

play_speed = st.sidebar.radio("Speed", ["fast","regular","slow"], index=1, horizontal=True)

# Reflect selection in URL
st.query_params.update({"gameId": str(gameId), "playId": str(playId)})

st.title("NFL Play Animation Viewer (no coverage)")
st.caption("Pick a tracked game and play, then render the animation.")

if st.sidebar.button("Render animation", type="primary"):
    # Description above chart
    desc = plays.loc[(plays["gameId"] == gameId) & (plays["playId"] == playId), "playDescription"].astype(str).iloc[0]
    st.markdown(f"**Play description:** {desc}")

    try:
        with st.spinner("Building animation…"):
            fig = build_fig_cached(games, plays, players, tracking_game, gameId, playId)

        # Apply speed without rebuilding
        fr_duration = {"fast":65, "regular":100, "slow":135}[play_speed]
        if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
            btn = fig.layout.updatemenus[0].buttons[0]
            btn.args[1]["frame"]["duration"] = fr_duration

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception as e:
        st.exception(e)
else:
    st.info("Select a play and click **Render animation**.")
