import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to python path to load engines
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict_engine import predict_match
from explain import explain_prediction
from lineup_engine import get_starting_xi
from tournament_sim import run_tournament_simulation
from weak_link_detector import find_weak_links
from report_generator import generate_match_report

# Set up page config
st.set_page_config(
    page_title="Beat the Table — Football Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium design system
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        /* ── Base ─────────────────────────────────── */
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        .main, .block-container     { background-color: #080e1c !important; color: #e2e8f0; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(160deg, #0f172a 0%, #0a1628 100%) !important;
            border-right: 1px solid rgba(99,179,237,0.12);
        }

        /* ── Typography ───────────────────────────── */
        h1, h2, h3 { font-family: 'Outfit', sans-serif !important; letter-spacing: -0.02em; }
        h1 { font-weight: 900 !important; }
        h2 { font-weight: 700 !important; color: #e2e8f0 !important; font-size: 1.25rem !important; }
        h3 { font-weight: 600 !important; color: #cbd5e1 !important; }

        /* ── Hero header ──────────────────────────── */
        .hero-header {
            background: linear-gradient(135deg, #0f172a 0%, #1a2744 50%, #0f172a 100%);
            border: 1px solid rgba(99,179,237,0.18);
            border-radius: 16px;
            padding: 32px 36px 28px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero-header::before {
            content: '';
            position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #3b82f6, #60a5fa, #38bdf8, #818cf8);
        }
        .hero-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2.4rem; font-weight: 900;
            background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #818cf8 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; line-height: 1.1; margin: 0;
        }
        .hero-sub {
            color: #64748b; font-size: 1rem; margin-top: 8px; font-weight: 400;
            font-family: 'Inter', sans-serif;
        }
        .hero-badge {
            display: inline-block; background: rgba(59,130,246,0.15);
            border: 1px solid rgba(59,130,246,0.3); color: #60a5fa;
            padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
            font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
            margin-bottom: 12px; font-family: 'Inter', sans-serif;
        }

        /* ── Cards ────────────────────────────────── */
        .section-card {
            background: #0f172a;
            border: 1px solid rgba(99,179,237,0.1);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        }
        .card-label {
            font-family: 'Outfit', sans-serif;
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: #475569; margin-bottom: 14px;
        }

        /* ── Probability bars ─────────────────────── */
        .prob-bar-wrap { margin-bottom: 14px; }
        .prob-bar-label {
            display: flex; justify-content: space-between;
            font-size: 0.85rem; font-weight: 600; margin-bottom: 5px;
            color: #cbd5e1;
        }
        .prob-bar-track {
            height: 10px; background: #1e293b;
            border-radius: 99px; overflow: hidden;
        }
        .prob-bar-fill {
            height: 100%; border-radius: 99px;
            transition: width 0.8s cubic-bezier(.4,0,.2,1);
        }

        /* ── Commentary ───────────────────────────── */
        .commentary-box {
            background: #0f172a;
            border: 1px solid rgba(99,179,237,0.12);
            border-left: 4px solid #3b82f6;
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 16px;
            font-family: 'Inter', sans-serif;
            font-size: 0.96rem;
            line-height: 1.75;
            color: #cbd5e1;
        }
        .commentary-icon {
            font-size: 1.4rem; margin-bottom: 10px; display: block;
        }

        /* ── Advisor box ──────────────────────────── */
        .advisor-box {
            background: linear-gradient(135deg, #1a1a0e 0%, #1c1a0a 100%);
            border: 1px solid rgba(245,158,11,0.25);
            border-left: 4px solid #f59e0b;
            padding: 20px 22px; border-radius: 10px;
            margin-top: 12px; font-size: 0.96rem;
            line-height: 1.7; color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }

        /* ── Sidebar polish ───────────────────────── */
        .sidebar-section-label {
            font-family: 'Outfit', sans-serif;
            font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: #334155;
            margin: 16px 0 8px; padding-bottom: 6px;
            border-bottom: 1px solid rgba(99,179,237,0.1);
        }
        .matchup-display {
            background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(239,68,68,0.1));
            border: 1px solid rgba(99,179,237,0.15);
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            margin: 12px 0;
            font-family: 'Outfit', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #e2e8f0;
            letter-spacing: 0.01em;
        }

        /* ── Table polish ─────────────────────────── */
        .stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
        .stDataFrame thead tr th {
            background-color: #1e293b !important;
            color: #60a5fa !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.78rem !important; font-weight: 700 !important;
            letter-spacing: 0.06em !important; text-transform: uppercase !important;
        }
        .stDataFrame tbody tr:hover td { background: rgba(59,130,246,0.08) !important; }
        .stDataFrame tbody tr td { color: #e2e8f0 !important; font-size: 0.9rem !important; }

        /* ── Metrics ──────────────────────────────── */
        div[data-testid="stMetricValue"] {
            color: #38bdf8 !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }

        /* ── Button ───────────────────────────────── */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
            border: none !important; border-radius: 10px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important; font-size: 1rem !important;
            letter-spacing: 0.02em !important;
            padding: 14px 20px !important;
            box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important;
            transition: all 0.2s !important;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 28px rgba(59,130,246,0.5) !important;
        }
        div.stButton > button[kind="primary"]:disabled {
            background: #1e293b !important; box-shadow: none !important;
            color: #475569 !important;
        }

        /* ── Section dividers ─────────────────────── */
        hr { border-color: rgba(99,179,237,0.1) !important; margin: 28px 0 !important; }

        /* ── Selectbox + inputs ───────────────────── */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextInput"] label { color: #64748b !important; font-size: 0.82rem !important; font-weight: 500 !important; }
        
        div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            border-color: rgba(99,179,237,0.2) !important;
            color: #e2e8f0 !important;
            border-radius: 8px !important;
        }
        
        div[data-baseweb="select"] ul {
            background-color: #0f172a !important;
        }
        
        div[data-baseweb="select"] li {
            color: #e2e8f0 !important;
        }
        div[data-baseweb="select"] li:hover {
            background-color: rgba(59,130,246,0.15) !important;
        }
        
        /* Ensure table wrapper blends */
        [data-testid="stDataFrame"] > div {
            border: 1px solid rgba(99,179,237,0.15) !important;
            border-radius: 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load list of teams from team_stats.csv
BASE_DIR = os.path.dirname(__file__)
TEAM_STATS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'team_stats.csv')

@st.cache_data
def load_teams():
    if os.path.exists(TEAM_STATS_PATH):
        df = pd.read_csv(TEAM_STATS_PATH)
        return sorted(df['team'].tolist())
    return ["France", "Argentina", "Brazil", "England", "Spain", "Germany", "Croatia", "Morocco"]

teams_list = load_teams()

# ── ALL FOOTBALL FORMATIONS ─────────────────────────────────────────────────────
# Each value: list of (role, x_horizontal 0-100, y_depth 0-100)
FORMATIONS = {
    '4-4-2': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('LM',   8, 52), ('CM', 35, 55), ('CM', 65, 55), ('RM', 92, 52),
        ('ST',  35, 85), ('ST', 65, 85),
    ],
    '4-3-3': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('CM',  28, 58), ('CM', 50, 52), ('CM', 72, 58),
        ('LW',  10, 82), ('ST', 50, 88), ('RW', 90, 82),
    ],
    '4-2-3-1': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('CDM', 35, 44), ('CDM', 65, 44),
        ('LW',  10, 65), ('CAM', 50, 62), ('RW', 90, 65),
        ('ST',  50, 88),
    ],
    '5-3-2': [
        ('GK',  50,  8),
        ('LWB',  5, 35), ('CB', 28, 24), ('CB', 50, 22), ('CB', 72, 24), ('RWB', 95, 35),
        ('CM',  25, 60), ('CM', 50, 55), ('CM', 75, 60),
        ('ST',  35, 87), ('ST', 65, 87),
    ],
    '3-5-2': [
        ('GK',  50,  8),
        ('CB',  25, 24), ('CB', 50, 22), ('CB', 75, 24),
        ('LM',   6, 55), ('CDM', 32, 48), ('CM', 50, 55), ('CDM', 68, 48), ('RM', 94, 55),
        ('ST',  35, 87), ('ST', 65, 87),
    ],
    '3-4-3': [
        ('GK',  50,  8),
        ('CB',  25, 24), ('CB', 50, 22), ('CB', 75, 24),
        ('LM',   8, 55), ('CM', 35, 55), ('CM', 65, 55), ('RM', 92, 55),
        ('LW',  10, 83), ('ST', 50, 88), ('RW', 90, 83),
    ],
    '4-5-1': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('LM',   6, 55), ('CM', 28, 58), ('CM', 50, 52), ('CM', 72, 58), ('RM', 94, 55),
        ('ST',  50, 88),
    ],
    '5-4-1': [
        ('GK',  50,  8),
        ('LWB',  5, 35), ('CB', 28, 24), ('CB', 50, 22), ('CB', 72, 24), ('RWB', 95, 35),
        ('LM',  10, 62), ('CM', 35, 62), ('CM', 65, 62), ('RM', 90, 62),
        ('ST',  50, 88),
    ],
    '4-1-4-1': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('CDM', 50, 42),
        ('LM',   8, 64), ('CM', 33, 66), ('CM', 67, 66), ('RM', 92, 64),
        ('ST',  50, 88),
    ],
    '4-3-2-1': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('CM',  25, 50), ('CM', 50, 45), ('CM', 75, 50),
        ('LW',  30, 68), ('RW', 70, 68),
        ('ST',  50, 88),
    ],
    '4-4-1-1': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('LM',   8, 52), ('CM', 35, 55), ('CM', 65, 55), ('RM', 92, 52),
        ('CAM', 50, 70),
        ('ST',  50, 88),
    ],
    '3-6-1': [
        ('GK',  50,  8),
        ('CB',  25, 24), ('CB', 50, 22), ('CB', 75, 24),
        ('LM',   5, 52), ('CDM', 28, 48), ('CM', 44, 56), ('CM', 56, 56), ('CDM', 72, 48), ('RM', 95, 52),
        ('ST',  50, 88),
    ],
    '4-2-4': [
        ('GK',  50,  8),
        ('LB',  10, 26), ('CB', 35, 24), ('CB', 65, 24), ('RB', 90, 26),
        ('CDM', 35, 48), ('CDM', 65, 48),
        ('LW',   8, 82), ('ST', 35, 88), ('ST', 65, 88), ('RW', 92, 82),
    ],
}

# ── TACTICAL ADVISOR NOTES ──────────────────────────────────────────────────────
FORMATION_NOTES = {
    '4-4-2': {
        'high_press':      ("⚠️ The flat 4-4-2 struggles against a high press. Your two strikers are isolated up top and can't help press back. Midfield can get overrun.", -4),
        'counter_pace':    ("⚠️ Wide flanks in a 4-4-2 are exposed to fast counter attacks — fast fullbacks or wingers from the opponent can hurt you on the break.", -5),
        'aerial_threat':   ("✅ The 4-4-2 offers good aerial cover with two strikers pressing and two CBs defending. You should win most aerial duels in both boxes.", +3),
        'possession_heavy':("⚠️ Against possession-heavy sides, your flat midfield four can be stretched. The opponent can overload central midfield zones.", -4),
    },
    '4-3-3': {
        'high_press':      ("✅ The 4-3-3 is well suited against a high press. Your front three press back and the CM triangle provides short pass options to escape the press.", +3),
        'counter_pace':    ("⚠️ High fullbacks in 4-3-3 leave you exposed to counter attacks on the flanks. Track back wingers are essential.", -4),
        'aerial_threat':   ("⚠️ Only two CBs and no extra aerial coverage. Crosses from wide areas can be dangerous against a tall front line.", -3),
        'possession_heavy':("✅ The CM triangle is excellent at dominating ball and press in central zones. You have good compactness to win the ball back.", +3),
    },
    '4-2-3-1': {
        'high_press':      ("✅ The double CDM pivot provides a solid press-escape route. Ball is recycled deep and quickly up the pitch. One of the best formations vs high press.", +5),
        'counter_pace':    ("✅ Double CDM drops quickly to block counter attacks in behind your fullbacks. Solid base to defend transitions.", +3),
        'aerial_threat':   ("⚠️ CAM is small and technically-focused. Aerial duels in midfield can be a problem.", -2),
        'possession_heavy':("✅ Double pivot dominates the middle third and compresses space. CAM presses effectively to win the ball back.", +4),
    },
    '5-3-2': {
        'high_press':      ("✅ Five defenders give plenty of ball retention options. Short passes can be played back safely through the CBs and the WBs have width to beat the press.", +2),
        'counter_pace':    ("✅ Five at the back is specifically designed to neutralize fast counter attacks. Three CBs cover the width, WBs provide defensive redundancy.", +6),
        'aerial_threat':   ("✅ Three CBs dominate aerial duels in the box. Very difficult to score from crosses against this backline.", +5),
        'possession_heavy':("⚠️ Only three midfielders. You may struggle to match their density in the middle and could cede possession for long periods.", -5),
    },
    '3-5-2': {
        'high_press':      ("✅ Five midfielders give lots of press-escape options. The wide midfielders can switch the ball and bypass the press easily.", +3),
        'counter_pace':    ("⚠️ Only three at the back. If wide midfielders are caught high, you're exposed to 2v3 situations on the counter.", -4),
        'aerial_threat':   ("⚠️ A back three offers decent aerial cover but the lack of full-backs means wide crossing areas are open.", -3),
        'possession_heavy':("✅ Five midfielders directly compete with possession-heavy teams for control. You can match them player-for-player in midfield.", +5),
    },
    '3-4-3': {
        'high_press':      ("✅ Front three presses from the top and disrupts the opponent's build-up. High intensity and great for counter pressing.", +4),
        'counter_pace':    ("⚠️ High and wide shape leaves huge gaps behind the midfield. Quick transitions can easily expose a back three.", -6),
        'aerial_threat':   ("⚠️ Highest-risk formation against aerial threats — only three CBs and your wide forwards offer zero defensive cover.", -5),
        'possession_heavy':("✅ Aggressive front three wins the ball high up the pitch and stops the opponent from building comfortably.", +3),
    },
    '4-5-1': {
        'high_press':      ("⚠️ Lone striker will be isolated in the press. Five midfielders can absorb pressure but it's hard to progress the ball.", -2),
        'counter_pace':    ("✅ Compact mid-block with five midfielders makes counter attacks very difficult to execute through the middle.", +3),
        'aerial_threat':   ("✅ Five midfielders help win second balls from aerial duels throughout the pitch.", +2),
        'possession_heavy':("✅ Disciplined 5-man midfield makes it extremely hard for possession teams to find gaps through the centre.", +5),
    },
    '5-4-1': {
        'high_press':      ("✅ Back five and four midfielders create an extremely compact block. Very hard to press through.", +2),
        'counter_pace':    ("✅ Defensive solidity is maximum. Lone striker can hold the ball while the rest of the team defends deep.", +7),
        'aerial_threat':   ("✅ Five defenders dominate all aerial situations. Almost impenetrable from crosses.", +6),
        'possession_heavy':("✅ Nine outfield players behind the ball makes it extremely difficult to break you down. Excellent against possession-dominant teams.", +5),
    },
    '4-1-4-1': {
        'high_press':      ("✅ Lone CDM as a screen + four midfielders provides excellent press resistance. Ball can be recycled to wide areas easily.", +3),
        'counter_pace':    ("✅ CDM sits between the defensive line and midfield to intercept counter-attack channels.", +4),
        'aerial_threat':   ("⚠️ Four midfielders are mostly technical players, not aerial specialists.", -2),
        'possession_heavy':("✅ CDM controls the midfield zone and the four can press and recover compactly.", +3),
    },
    '4-3-2-1': {
        'high_press':      ("✅ The Christmas Tree uses the CAM/shadow strikers to press in pairs from the top. Good press disruption.", +2),
        'counter_pace':    ("⚠️ The two narrow CAMs don't track back well and leave the wide areas in behind your fullbacks open to quick attacks.", -3),
        'aerial_threat':   ("⚠️ No wide players — crosses from the flanks will be contested only by two CBs.", -3),
        'possession_heavy':("✅ Narrow shape forces play wide, where possession teams are less comfortable. Good compactness centrally.", +3),
    },
    '4-4-1-1': {
        'high_press':      ("⚠️ CAM behind a lone striker creates an uneven press structure. Can be bypassed with a simple ball over the first line.", -2),
        'counter_pace':    ("✅ Solid defensive four with wide midfielders who can track back. Decent counter-attack protection.", +2),
        'aerial_threat':   ("✅ A flat four is compact and good at heading clearances.", +2),
        'possession_heavy':("⚠️ CAM leaves a gap between midfield and the lone striker, which possession teams can exploit through the middle.", -3),
    },
    '3-6-1': {
        'high_press':      ("✅ Six midfielders completely overload the press. Almost impossible to shut down all passing lanes simultaneously.", +5),
        'counter_pace':    ("⚠️ Only three at the back. Extremely risky with fast teams — counters can reach the goal line quickly.", -7),
        'aerial_threat':   ("⚠️ Three CBs offer standard aerial cover but the extremely high shape means lots of space behind the line.", -4),
        'possession_heavy':("✅ Six midfielders means you will always have the numerical advantage in possession. Total midfield dominance.", +7),
    },
    '4-2-4': {
        'high_press':      ("✅ Four attackers press from the top constantly. Overwhelming in the opponent's half. Opponent can barely build up.", +5),
        'counter_pace':    ("⚠️ Ultra-high and wide shape leaves the back four massively exposed to 4v4 counter attacks. Extremely risky.", -8),
        'aerial_threat':   ("⚠️ No extra aerial cover. Crosses and set pieces can be very dangerous against this shape.", -4),
        'possession_heavy':("✅ Four attackers disrupt ball circulation and force mistakes. Aggressive press to win possession in their half.", +3),
    },
}

def map_players_to_formation(starting_xi, formation_name):
    formation_slots = FORMATIONS.get(formation_name)
    if not formation_slots or len(formation_slots) != 11:
        return None

    DEFENDER_ROLES  = {'CB', 'LB', 'RB', 'LWB', 'RWB'}
    MID_ROLES       = {'CDM', 'CM', 'CAM', 'LM', 'RM'}
    ATTACKER_ROLES  = {'LW', 'RW', 'ST'}

    def role_cat(role):
        if role == 'GK':          return 'GK'
        if role in DEFENDER_ROLES: return 'DEF'
        if role in MID_ROLES:      return 'MID'
        return 'ATK'

    pool = sorted(starting_xi, key=lambda p: p['adjusted_rating'], reverse=True)
    assigned = []
    used = [False] * len(pool)
    slot_assignments = [None] * len(formation_slots)

    for si, (slot_role, sx, sy) in enumerate(formation_slots):
        for pi, p in enumerate(pool):
            if not used[pi] and p['role'] == slot_role:
                slot_assignments[si] = pi
                used[pi] = True
                break

    for si, (slot_role, sx, sy) in enumerate(formation_slots):
        if slot_assignments[si] is not None: continue
        want_cat = role_cat(slot_role)
        for pi, p in enumerate(pool):
            if not used[pi] and role_cat(p['role']) == want_cat:
                slot_assignments[si] = pi
                used[pi] = True
                break

    for si, (slot_role, sx, sy) in enumerate(formation_slots):
        if slot_assignments[si] is not None: continue
        for pi, p in enumerate(pool):
            if not used[pi]:
                slot_assignments[si] = pi
                used[pi] = True
                break

    result = []
    for num, (si, (slot_role, sx, sy)) in enumerate(zip(slot_assignments, formation_slots), start=1):
        p = pool[si] if si is not None else {'name': '???', 'role': slot_role, 'adjusted_rating': 60}
        result.append({'name': p['name'], 'role': slot_role, 'num': num, 'adjusted_rating': p['adjusted_rating'], 'x': sx, 'y': sy})
    return result

# ── Hero Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚽ AI-Powered Football Analytics</div>
    <div class="hero-title">Beat the Table</div>
    <div class="hero-sub">Select two teams, pick their playstyle, and get deep tactical intelligence — Win probabilities, Starting XIs, Weak links &amp; Formation advice, all in one click.</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="hero-title" style="font-size:1.4rem;padding:4px 0 16px;">Beat<span style="color:#60a5fa;"> the </span>Table</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-label">Match Setup</div>', unsafe_allow_html=True)
team_a = st.sidebar.selectbox("Your Team (Team A)", teams_list, index=teams_list.index("France") if "France" in teams_list else 0)
team_b = st.sidebar.selectbox("Opponent (Team B)", teams_list, index=teams_list.index("Argentina") if "Argentina" in teams_list else 1)

if team_a != team_b:
    st.sidebar.markdown(
        f'<div class="matchup-display">{team_a} <span style="color:#64748b;font-size:0.85rem;">vs</span> {team_b}</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown('<div class="sidebar-section-label">Opponent Playstyle</div>', unsafe_allow_html=True)
opp_style = st.sidebar.selectbox(
    "How does the opponent play?",
    [
        "High Pressing / Aggressive Press",
        "Aerial Threat / Long Balls",
        "Fast Counter Attacks",
        "Possession Heavy / Tika-Taka",
        "Balanced",
    ]
)
STYLE_TO_PROFILE = {
    "High Pressing / Aggressive Press": {'high_press': True,  'aerial_threat': False, 'counter_pace': False, 'possession_heavy': False},
    "Aerial Threat / Long Balls":       {'high_press': False, 'aerial_threat': True,  'counter_pace': False, 'possession_heavy': False},
    "Fast Counter Attacks":             {'high_press': False, 'aerial_threat': False, 'counter_pace': True,  'possession_heavy': False},
    "Possession Heavy / Tika-Taka":     {'high_press': False, 'aerial_threat': False, 'counter_pace': False, 'possession_heavy': True},
    "Balanced":                         {'high_press': False, 'aerial_threat': False, 'counter_pace': False, 'possession_heavy': False},
}
opponent_profile = STYLE_TO_PROFILE[opp_style]

# ── Button: trigger predictions ──────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-section-label">Run Analysis</div>', unsafe_allow_html=True)

# Auto-reset when the user changes teams or playstyle
_selection_key = f"{team_a}|{team_b}|{opp_style}"
if st.session_state.get('_sel_key') != _selection_key:
    st.session_state['_sel_key']          = _selection_key
    st.session_state['prediction_active'] = False

if st.sidebar.button(
    "🔮 Analyse This Match",
    use_container_width=True,
    type="primary",
    disabled=(team_a == team_b),
):
    st.session_state['prediction_active'] = True

if team_a == team_b:
    st.sidebar.warning("Please select two different teams.")

def estimate_team_profile(team_name):
    """Estimate a team's playstyle based on their historical stats."""
    try:
        df = pd.read_csv(TEAM_STATS_PATH)
        row = df[df['team'] == team_name]
        if not row.empty:
            avg_gf = row.iloc[0]['avg_gf']
            avg_ga = row.iloc[0]['avg_ga']
            squad_rating = row.iloc[0]['squad_rating']
            return {
                'high_press': avg_gf > 1.5,
                'aerial_threat': avg_gf > 1.2 and squad_rating > 75,
                'counter_pace': avg_ga > 1.2,
                'possession_heavy': squad_rating > 78
            }
    except Exception:
        pass
    return {'high_press': True, 'aerial_threat': False, 'counter_pace': True, 'possession_heavy': False}

def draw_pitch_portrait(players_with_xy, dot_color='#38bdf8', fig_title='', highlights=None):
    """Draw a portrait (vertical) football pitch with players at exact formation positions.

    Args:
        players_with_xy : list of dicts with keys 'name', 'role', 'x' (0-100 horiz), 'y' (0-100 depth)
        dot_color       : default dot colour
        fig_title       : optional title above the pitch
        highlights      : dict {player_name: hex_colour} for danger-zone players
    """
    fig, ax = plt.subplots(figsize=(5, 7))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0d7a3e')          # grass green

    # ── Pitch markings ───────────────────────────────────────────────────────
    W, H = 100, 100
    lw, lc = 1.8, 'rgba(255,255,255,0.75)'
    line_color = '#ffffffbb'

    # Outer border
    ax.add_patch(patches.Rectangle((0, 0), W, H, linewidth=2,
                                   edgecolor='white', facecolor='none'))
    # Centre line & circle
    ax.plot([0, W], [H/2, H/2], color='white', linewidth=lw, alpha=0.7)
    ax.add_patch(patches.Circle((W/2, H/2), 9.5, edgecolor='white',
                                facecolor='none', linewidth=lw, alpha=0.7))
    ax.plot(W/2, H/2, 'o', color='white', markersize=3, alpha=0.7)

    # Penalty areas (top & bottom)
    pa_w, pa_h = 56, 16
    ax.add_patch(patches.Rectangle(((W - pa_w)/2, H - pa_h), pa_w, pa_h,
                                   linewidth=lw, edgecolor='white', facecolor='none', alpha=0.7))
    ax.add_patch(patches.Rectangle(((W - pa_w)/2, 0), pa_w, pa_h,
                                   linewidth=lw, edgecolor='white', facecolor='none', alpha=0.7))
    # 6-yard boxes
    sb_w, sb_h = 28, 6
    ax.add_patch(patches.Rectangle(((W - sb_w)/2, H - sb_h), sb_w, sb_h,
                                   linewidth=lw, edgecolor='white', facecolor='none', alpha=0.5))
    ax.add_patch(patches.Rectangle(((W - sb_w)/2, 0), sb_w, sb_h,
                                   linewidth=lw, edgecolor='white', facecolor='none', alpha=0.5))
    # Penalty spots
    ax.plot(W/2, H - 11, 'o', color='white', markersize=2.5, alpha=0.6)
    ax.plot(W/2,      11, 'o', color='white', markersize=2.5, alpha=0.6)

    # Stripe effect (alternating dark/light bands) — optional but looks great
    for i in range(10):
        alpha = 0.03 if i % 2 == 0 else 0.0
        ax.add_patch(patches.Rectangle((0, i * 10), W, 10,
                                       facecolor='black', alpha=alpha, linewidth=0))

    # ── Title ─────────────────────────────────────────────────────────────────
    if fig_title:
        ax.set_title(fig_title, color='white', fontsize=9, fontweight='bold',
                     fontfamily='DejaVu Sans', pad=6)

    # ── Players ───────────────────────────────────────────────────────────────
    for p in players_with_xy:
        px = p['x']           # horizontal 0-100 (left→right)
        py = p['y']           # depth 0-100 (GK=low, ST=high)

        # Highlight override
        color = dot_color
        ring_w = 1.2
        if highlights and p['name'] in highlights:
            color = highlights[p['name']]
            ring_w = 2.5

        # Glowing outer ring for highlighted players
        if highlights and p['name'] in highlights:
            ax.plot(px, py, 'o', color=color, markersize=20, alpha=0.25)

        ax.plot(px, py, 'o', color=color, markersize=14,
                markeredgecolor='white', markeredgewidth=ring_w, zorder=5)

        # Short surname label below the dot
        clean_name = p['name'].split(' ')[-1] if ' ' in p['name'] else p['name']
        label = f"{clean_name}\n{p['role']}"
        ax.text(px, py - 6.5, label,
                color='white', fontsize=6.2, ha='center', va='top',
                fontweight='bold', zorder=6,
                bbox=dict(facecolor='#00000088', edgecolor='none',
                          boxstyle='round,pad=0.25'))

    ax.set_xlim(-3, 103)
    ax.set_ylim(-10, 108)
    ax.axis('off')
    plt.tight_layout(pad=0.3)
    return fig


def draw_pitch(starting_xi, mirror=False, dot_color='#38bdf8'):
    """Draw a soccer pitch with players dynamically positioned by formation lines.
    
    Groups roles into defensive / midfield / attacking lines, assigns x by line,
    and distributes y evenly within each line so the visual matches the formation.
    
    Args:
        starting_xi: list of player dicts with 'name', 'role' keys
        mirror:      if True, flips x-axis so team attacks right-to-left (opponent side)
        dot_color:   color of player dots
    """
    fig_pitch, ax = plt.subplots(figsize=(6, 4.2))
    ax.set_facecolor('#1e293b')
    fig_pitch.patch.set_facecolor('#0f172a')

    # --- Pitch markings ---
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, linewidth=2, edgecolor='#38bdf8', facecolor='#0f172a'))
    ax.plot([50, 50], [0, 100], color='#38bdf8', linewidth=1.5)
    ax.add_patch(patches.Circle((50, 50), 12, edgecolor='#38bdf8', facecolor='none', linewidth=1.5))
    ax.plot(50, 50, 'o', color='#38bdf8')
    ax.add_patch(patches.Rectangle((0,   20), 16.5, 60, edgecolor='#38bdf8', facecolor='none', linewidth=1.5))
    ax.add_patch(patches.Rectangle((83.5,20), 16.5, 60, edgecolor='#38bdf8', facecolor='none', linewidth=1.5))

    # --- Role → line mapping (lower number = closer to own goal) ---
    ROLE_LINE = {
        'GK':  0,
        'CB':  1, 'LB': 1, 'RB': 1, 'LWB': 1, 'RWB': 1,
        'CDM': 2,
        'CM':  3,
        'CAM': 4,
        'LW':  5, 'RW': 5, 'ST': 5,
    }

    # --- Y-ordering within a line (left flank → right flank) ---
    ROLE_Y_ORDER = {
        'LB': 0, 'LWB': 0, 'LW': 0,          # left flank → lowest y
        'CB': 1, 'CDM': 1, 'CM': 1, 'CAM': 1, 'GK': 1,   # centre
        'ST': 1,                                # centre strikers
        'RB': 2, 'RWB': 2, 'RW': 2,           # right flank → highest y
    }

    # --- Group players by line ---
    from collections import defaultdict
    lines = defaultdict(list)
    for p in starting_xi:
        line_idx = ROLE_LINE.get(p['role'], 3)
        lines[line_idx].append(p)

    # --- Collapse empty lines so spacing is proportional to actual lines used ---
    present_lines = sorted(lines.keys())          # e.g. [0, 1, 3, 5]
    n_lines = len(present_lines)

    # x positions: GK near own goal, attackers near opponent goal
    # Map each present line index to an x coordinate 8..88
    def line_to_x(rank, total):
        """rank = 0-based index in present_lines, total = number of lines."""
        return 8 + rank * (80 / max(total - 1, 1))

    line_x = {line_idx: line_to_x(rank, n_lines) for rank, line_idx in enumerate(present_lines)}

    # --- Draw each player ---
    for line_idx, players in lines.items():
        x_base = line_x[line_idx]

        # Sort players within line: left-flank first, centre next, right-flank last
        players_sorted = sorted(players, key=lambda p: (ROLE_Y_ORDER.get(p['role'], 1),
                                                          p['name']))

        n = len(players_sorted)
        # Distribute y evenly between margins 15..85
        margin = 15
        if n == 1:
            y_positions = [50]
        else:
            step = (100 - 2 * margin) / (n - 1)
            y_positions = [margin + i * step for i in range(n)]

        for p, y in zip(players_sorted, y_positions):
            plot_x = (100 - x_base) if mirror else x_base

            # dot
            ax.plot(plot_x, y, 'o', color=dot_color, markersize=13,
                    markeredgecolor='white', markeredgewidth=1.2)

            # label: above dot for mirrored team, below for normal
            label_dy = +5.5 if mirror else -5.5
            clean_name = p['name'].split(' ')[-1] if ' ' in p['name'] else p['name']
            ax.text(plot_x, y + label_dy,
                    f"{clean_name}\n({p['role']})",
                    color='white', fontsize=7, ha='center', fontweight='bold',
                    bbox=dict(facecolor='#0f172a', alpha=0.85, edgecolor='none',
                              boxstyle='round,pad=0.2'))

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.axis('off')
    return fig_pitch

# ── MAIN BODY ────────────────────────────────────────────────────────────────────────────────
if team_a == team_b:
    st.error("Please select two different teams to run the analysis.")
elif not st.session_state.get('prediction_active', False):
    st.markdown(f"""
    <div class="section-card" style="text-align:center; padding: 48px 36px;">
        <div style="font-size:3rem; margin-bottom:16px;"></div>
        <div style="font-family:'Outfit',sans-serif; font-size:1.6rem; font-weight:700; color:#e2e8f0; margin-bottom:8px;">
            Ready to Analyse <span style="color:#60a5fa;">{team_a}</span> vs <span style="color:#f43f5e;">{team_b}</span>
        </div>
        <div style="color:#64748b; font-size:0.95rem; max-width:480px; margin:0 auto; line-height:1.6;">
            Set the opponent&apos;s playstyle in the sidebar, then click
            <b style="color:#60a5fa;">Analyse This Match</b> to get the full tactical breakdown.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # 1. Predict Outcome
    probs = predict_match(team_a, team_b)
    
    # 2. Get Commentary
    explanation = explain_prediction(team_a, team_b)
    
    # 3. Get Team A Starting XI
    lineup_res_a = get_starting_xi(team_a, opponent_profile)
    formation_a = lineup_res_a['formation']
    starting_xi_a = lineup_res_a['starting_xi']
    
    # 4. Get Team B Starting XI (facing Team A's dynamic style profile)
    profile_a = estimate_team_profile(team_a)
    lineup_res_b = get_starting_xi(team_b, profile_a)
    formation_b = lineup_res_b['formation']
    starting_xi_b = lineup_res_b['starting_xi']
    
    # Layout definition: single page, multi-column dashboard
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card-label">📊 Win Probability Breakdown</div>', unsafe_allow_html=True)

        # Premium horizontal probability bars
        win_pct  = probs['Win']
        draw_pct = probs['Draw']
        loss_pct = probs['Loss']

        st.markdown(f"""
        <div class="section-card">
            <div class="prob-bar-wrap">
                <div class="prob-bar-label">
                    <span>🟦 {team_a} Win</span>
                    <span style="color:#3b82f6; font-family:'Outfit',sans-serif; font-size:1.1rem;">{win_pct:.1%}</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{win_pct*100:.1f}%; background:linear-gradient(90deg,#1d4ed8,#3b82f6);"></div>
                </div>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-label">
                    <span>➖ Draw</span>
                    <span style="color:#64748b; font-family:'Outfit',sans-serif; font-size:1.1rem;">{draw_pct:.1%}</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{draw_pct*100:.1f}%; background:linear-gradient(90deg,#334155,#64748b);"></div>
                </div>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-label">
                    <span>🟥 {team_b} Win</span>
                    <span style="color:#ef4444; font-family:'Outfit',sans-serif; font-size:1.1rem;">{loss_pct:.1%}</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{loss_pct*100:.1f}%; background:linear-gradient(90deg,#991b1b,#ef4444);"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Still render a compact Plotly chart below the bars for visual depth
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{team_a} Win", "Draw", f"{team_b} Win"],
            y=[win_pct, draw_pct, loss_pct],
            marker=dict(
                color=['#3b82f6', '#475569', '#ef4444'],
                line=dict(color='rgba(0,0,0,0)', width=0)
            ),
            text=[f"{v:.1%}" for v in [win_pct, draw_pct, loss_pct]],
            textposition='outside',
            textfont=dict(family='Outfit', size=13, color='#e2e8f0'),
            width=0.5,
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            height=200, margin=dict(l=0, r=0, t=8, b=0),
            yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.15]),
            xaxis=dict(showgrid=False),
            bargap=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="card-label">🎤 Match Preview</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="commentary-box"><span class="commentary-icon">🗨️</span>{explanation["explanation_text"]}</div>',
                    unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card-label">🏆 Tournament Projection</div>', unsafe_allow_html=True)
        group_teams = [team_a, team_b, "Denmark", "Tunisia"]
        sim_res = run_tournament_simulation(team_a, tuple(group_teams), num_runs=1000)
        
        sim_rows = sim_res['table']
        stage_icons = {'Group Stage':'🏁','R16':'⚡','QF':'🔥','SF':'💥','Final':'🏆','Winner':'🥇'}
        def adv_bar(pct_str):
            pct = float(pct_str.strip('%'))/100 if isinstance(pct_str,str) else pct_str
            c = '#22c55e' if pct>0.5 else '#f59e0b' if pct>0.25 else '#ef4444'
            return f'<div style="display:flex;align-items:center;gap:8px;"><span style="font-family:\'Outfit\',sans-serif;font-weight:700;font-size:1rem;color:{c};min-width:44px;">{pct:.0%}</span><div style="flex:1;height:6px;background:#1e293b;border-radius:99px;overflow:hidden;"><div style="width:{pct*100:.0f}%;height:100%;background:{c};border-radius:99px;"></div></div></div>'
        tourn_rows_html = ''.join(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);"><td style="padding:12px 14px;font-size:1rem;font-weight:700;color:#e2e8f0;font-family:\'Outfit\',sans-serif;white-space:nowrap;">{stage_icons.get(r["Stage"],"")} {r["Stage"]}</td><td style="padding:12px 14px;">{adv_bar(r["Advancement Probability"])}</td><td style="padding:12px 14px;font-size:0.9rem;color:#94a3b8;">{r["Most Common Opponent"]}</td><td style="padding:12px 14px;font-size:0.9rem;font-weight:600;color:#cbd5e1;">{r["Opponent Frequency"]:.0%}</td></tr>'
            for r in sim_rows
        )
        st.markdown(f'<div style="background:#0f172a;border:1px solid rgba(99,179,237,0.12);border-radius:12px;overflow:hidden;"><table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#1e293b;"><th style="padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Stage</th><th style="padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Advance %</th><th style="padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Top Opponent</th><th style="padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Meet %</th></tr></thead><tbody>{tourn_rows_html}</tbody></table></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);'
            f'border-radius:8px;padding:12px 16px;margin-top:12px;font-family:Inter,sans-serif;font-size:0.9rem;">'
            f'🚨 <b style="color:#f59e0b;">Toughest Round for {team_a}:</b> '
            f'<span style="color:#e2e8f0;font-weight:600;">{sim_res["hardest_stage"]}</span> '
            f'<span style="color:#64748b;">(chance drops {sim_res["hardest_stage_drop"]:.1%})</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Show Side-by-Side Starting XIs
    st.markdown("---")
    st.markdown('<div class="card-label">📋 Recommended Match Starting XIs</div>', unsafe_allow_html=True)
    col_l1, col_l2 = st.columns(2)
    
    with col_l1:
        st.subheader(f"⚽ {team_a} XI ({formation_a})")
        mapped_a = map_players_to_formation(starting_xi_a, formation_a)
        st.pyplot(draw_pitch_portrait(mapped_a, dot_color='#60a5fa'))
        
        st.markdown('<div class="card-label" style="margin-top:20px;">📋 Why these players were chosen</div>', unsafe_allow_html=True)
        pos_colors = {'GK':'#f59e0b','CB':'#3b82f6','LB':'#3b82f6','RB':'#3b82f6','LWB':'#3b82f6','RWB':'#3b82f6','CDM':'#10b981','CM':'#10b981','CAM':'#10b981','LM':'#10b981','RM':'#10b981','LW':'#f43f5e','RW':'#f43f5e','ST':'#f43f5e'}
        def pos_badge(role): c=pos_colors.get(role,'#64748b'); return f'<span style="background:rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.18);color:{c};border:1px solid rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.35);padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700;font-family:\'Outfit\',sans-serif;">{role}</span>'
        def rat_bar(val, color='#60a5fa'): return f'<div style="display:flex;align-items:center;gap:6px;"><span style="font-family:\'Outfit\',sans-serif;font-weight:700;color:{color};font-size:0.95rem;min-width:28px;">{val}</span><div style="width:60px;height:5px;background:#1e293b;border-radius:99px;"><div style="width:{min(int((val-60)/40*100),100)}%;height:100%;background:{color};border-radius:99px;"></div></div></div>'
        rows_a_html = ''.join(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);"><td style="padding:10px 12px;">{pos_badge(p["role"])}</td><td style="padding:10px 12px;font-size:0.95rem;font-weight:600;color:#f1f5f9;">{p["name"]}</td><td style="padding:10px 12px;">{rat_bar(p["base_rating"],"#64748b")}</td><td style="padding:10px 12px;">{rat_bar(p["adjusted_rating"],"#60a5fa")}</td><td style="padding:10px 12px;font-size:0.78rem;color:#94a3b8;line-height:1.5;">{p["rationale"][:90]+"…" if len(p["rationale"])>90 else p["rationale"]}</td></tr>'
            for p in starting_xi_a
        )
        st.markdown(f'<div style="background:#0f172a;border:1px solid rgba(99,179,237,0.1);border-radius:12px;overflow:hidden;"><table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#1e293b;"><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Pos</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Player</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">OVR</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Tactical</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Coach Note</th></tr></thead><tbody>{rows_a_html}</tbody></table></div>', unsafe_allow_html=True)
        
    with col_l2:
        st.subheader(f"⚽ {team_b} XI ({formation_b})")
        mapped_b = map_players_to_formation(starting_xi_b, formation_b)
        # Mirror Team B so they attack downwards
        import copy
        mapped_b_mirror = copy.deepcopy(mapped_b)
        for p in mapped_b_mirror:
            p['y'] = 100 - p['y']
        st.pyplot(draw_pitch_portrait(mapped_b_mirror, dot_color='#f43f5e'))
        
        st.markdown('<div class="card-label" style="margin-top:20px;">📋 Why these players were chosen</div>', unsafe_allow_html=True)
        rows_b_html = ''.join(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);"><td style="padding:10px 12px;">{pos_badge(p["role"])}</td><td style="padding:10px 12px;font-size:0.95rem;font-weight:600;color:#f1f5f9;">{p["name"]}</td><td style="padding:10px 12px;">{rat_bar(p["base_rating"],"#64748b")}</td><td style="padding:10px 12px;">{rat_bar(p["adjusted_rating"],"#f43f5e")}</td><td style="padding:10px 12px;font-size:0.78rem;color:#94a3b8;line-height:1.5;">{p["rationale"][:90]+"…" if len(p["rationale"])>90 else p["rationale"]}</td></tr>'
            for p in starting_xi_b
        )
        st.markdown(f'<div style="background:#0f172a;border:1px solid rgba(244,63,94,0.1);border-radius:12px;overflow:hidden;"><table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#1e293b;"><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Pos</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Player</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">OVR</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Tactical</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Coach Note</th></tr></thead><tbody>{rows_b_html}</tbody></table></div>', unsafe_allow_html=True)

    # ── Row 2b: WEAK LINK DETECTOR ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Weak Link Detector")
    st.markdown(
        f"Our AI scanned **{team_b}'s** squad and cross-referenced every player in "
        f"**{team_a}'s** lineup against their most dangerous threats. "
        "Here are the three biggest mismatches you need to watch out for:"
    )

    # Map Team A players to their formation positions (needed for the vulnerability map)
    players_a_mapped = map_players_to_formation(starting_xi_a, formation_a)

    weak_links = find_weak_links(starting_xi_a, team_b, top_n=3)

    if not weak_links:
        st.info("No significant mismatches found — your lineup looks solid against this opponent!")
    else:
        highlight_colors = ['#dc2626', '#ea580c', '#ca8a04']
        highlight_map = {}
        for i, wl in enumerate(weak_links):
            highlight_map[wl['my_player']] = highlight_colors[i]

        col_wl1, col_wl2 = st.columns([1, 1.6])

        with col_wl1:
            st.markdown(f"**{team_a} — Vulnerability Map**")
            if players_a_mapped:
                fig_wl = draw_pitch_portrait(players_a_mapped, dot_color='#60a5fa', highlights=highlight_map)
                st.pyplot(fig_wl)
            st.caption("🔴 Critical risk\u00a0\u00a0🟠 Warning\u00a0\u00a0🟡 Caution")

        with col_wl2:
            for rank, wl in enumerate(weak_links, 1):
                sev_color = highlight_colors[rank - 1]
                badge_html = (
                    f'<span style="background:{sev_color};color:white;'
                    f'padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:bold;">'
                    f'{wl["severity"]}</span>'
                )
                st.markdown(
                    f'<div style="background:#1e293b;border-left:4px solid {sev_color};'
                    f'padding:16px 18px;border-radius:8px;margin-bottom:16px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
                    f'<span style="font-family:\'Outfit\',sans-serif;font-size:0.9rem;font-weight:700;color:#64748b;text-transform:uppercase;">Mismatch #{rank}</span>'
                    f'{badge_html}</div>'
                    f'<div style="margin-bottom:12px;display:flex;align-items:center;gap:12px;">'
                    f'<span style="font-size:1.4rem;font-weight:700;color:#f8fafc;">{wl["my_player"]}</span> '
                    f'<span style="font-size:1.1rem;color:#64748b;font-weight:600;">VS</span> '
                    f'<span style="font-size:1.4rem;font-weight:700;color:#f8fafc;">{wl["threat_player"]}</span>'
                    f'</div>'
                    f'<div style="color:#94a3b8;font-size:0.9rem;margin-bottom:12px;">'
                    f'<span style="display:inline-block;background:rgba(59,130,246,0.1);padding:4px 10px;border-radius:4px;border:1px solid rgba(59,130,246,0.2);margin-right:10px;">'
                    f'My {wl["my_key_stat"].capitalize()} ({wl["my_role"]}): <b style="color:#60a5fa">{wl["my_stat_val"]}</b></span>'
                    f'<span style="display:inline-block;background:rgba(239,68,68,0.1);padding:4px 10px;border-radius:4px;border:1px solid rgba(239,68,68,0.2);margin-right:10px;">'
                    f'Their {wl["threat_key_stat"].capitalize()} ({wl["threat_role"]}): <b style="color:#f43f5e">{wl["threat_stat_val"]}</b></span>'
                    f'<span style="display:inline-block;background:rgba(255,255,255,0.05);padding:4px 10px;border-radius:4px;">'
                    f'Gap: <b style="color:{sev_color}">{wl["vulnerability_score"]:+d}</b></span>'
                    f'</div>'
                    f'<div style="font-size:0.95rem;line-height:1.6;color:#cbd5e1;padding-top:12px;border-top:1px solid rgba(255,255,255,0.05);">'
                    f'{wl["description"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Row 3: FORMATION ADVISOR ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 Formation Advisor — Try a Different Lineup")
    st.markdown(
        f"Our model recommends **{formation_a}** for {team_a} against the opponent's "
        f"**{opp_style}** style. "
        "But you can try any formation below to see the tactical risks and updated win chances."
    )

    alt_formation = st.selectbox(
        f"Choose a formation for {team_a}",
        list(FORMATIONS.keys()),
        index=list(FORMATIONS.keys()).index(formation_a) if formation_a in FORMATIONS else 0,
        key="alt_formation_select"
    )

    style_key_map = {
        "High Pressing / Aggressive Press": 'high_press',
        "Aerial Threat / Long Balls":       'aerial_threat',
        "Fast Counter Attacks":             'counter_pace',
        "Possession Heavy / Tika-Taka":     'possession_heavy',
        "Balanced":                         'high_press',
    }
    style_key = style_key_map.get(opp_style, 'high_press')
    notes_for_formation = FORMATION_NOTES.get(alt_formation, {})
    note_text, win_modifier = notes_for_formation.get(style_key, ("No specific note available for this matchup.", 0))

    adjusted_win  = max(0.0, min(1.0, probs['Win']  + win_modifier / 100))
    adjusted_loss = max(0.0, min(1.0, probs['Loss'] - win_modifier / 100))

    col_adv1, col_adv2 = st.columns([1.4, 1])

    with col_adv1:
        st.markdown(
            f'<div class="advisor-box">'
            f'<b>Using {alt_formation} vs {opp_style}:</b><br><br>'
            f'{note_text}<br><br>'
            f'<b>Updated Win Probability:</b> '
            f'<span style="color:#38bdf8;font-size:1.3rem;font-weight:bold;">{adjusted_win:.1%}</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<b>Updated {team_b} Win Probability:</b> '
            f'<span style="color:#f43f5e;font-size:1.3rem;font-weight:bold;">{adjusted_loss:.1%}</span>'
            f'<br><span style="color:#94a3b8;font-size:0.85rem;">'
            f'(Base win% {probs["Win"]:.1%} adjusted by {win_modifier:+d}% for formation vs playstyle matchup)</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        alt_players = map_players_to_formation(starting_xi_a, alt_formation)
        if alt_players:
            st.markdown(f'<div class="card-label" style="margin-top:20px;">📋 {team_a} Starting XI in {alt_formation}</div>', unsafe_allow_html=True)
            alt_rows_html = ''.join(
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);"><td style="padding:9px 12px;font-size:0.85rem;color:#64748b;font-weight:700;">{p["num"]}</td><td style="padding:9px 12px;">{pos_badge(p["role"])}</td><td style="padding:9px 12px;font-size:0.95rem;font-weight:600;color:#f1f5f9;">{p["name"]}</td><td style="padding:9px 12px;">{rat_bar(p["adjusted_rating"],"#60a5fa")}</td></tr>'
                for p in alt_players
            )
            st.markdown(f'<div style="background:#0f172a;border:1px solid rgba(99,179,237,0.1);border-radius:12px;overflow:hidden;"><table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#1e293b;"><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">#</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Pos</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Player</th><th style="padding:8px 12px;text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;font-family:\'Outfit\',sans-serif;">Tactical</th></tr></thead><tbody>{alt_rows_html}</tbody></table></div>', unsafe_allow_html=True)

    with col_adv2:
        if alt_players:
            st.pyplot(draw_pitch_portrait(alt_players, dot_color='#60a5fa'))

    # ── PDF EXPORT ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label" style="text-align:center;">📄 Export Analysis</div>', unsafe_allow_html=True)
    with st.spinner("Generating PDF Report..."):
        try:
            pdf_bytes = generate_match_report(
                team_a=team_a, team_b=team_b, formation_a=formation_a, formation_b=formation_b,
                probs=probs, explanation=explanation, starting_xi_a=starting_xi_a, starting_xi_b=starting_xi_b,
                mapped_a=mapped_a, mapped_b_mirror=mapped_b_mirror, weak_links=weak_links,
                sim_res=sim_res, opp_style=opponent_profile
            )
            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
            with col_dl2:
                st.download_button(
                    label="⬇️ Download Full PDF Match Report",
                    data=pdf_bytes,
                    file_name=f"Beat_The_Table_{team_a}_vs_{team_b}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

# Footer predictions database
st.markdown("---")
col_e1, col_e2 = st.columns(2)

with col_e1:
    with st.expander("📅 Historical World Cup Predictions Database (2018–2026)"):
        pred_path = os.path.join(BASE_DIR, 'data', 'processed', 'world_cup_predictions_2018_2026.csv')
        
        # Auto-generate if missing
        if not os.path.exists(pred_path):
            with st.spinner("Generating predictions database..."):
                try:
                    from compile_predictions import compile_all_predictions
                    compile_all_predictions()
                except Exception as e:
                    st.error(f"Error compiling predictions: {e}")
                    
        if os.path.exists(pred_path):
            df_preds = pd.read_csv(pred_path)
            
            # Interactive Filters
            tourn_opt = ["All World Cups"] + sorted(df_preds['Tournament'].unique().tolist())
            selected_tourn = st.selectbox("Select Tournament", tourn_opt)
            search_team = st.text_input("Search Team Name", "")
            correct_opt = ["All Predictions", "Correct Only", "Incorrect Only"]
            selected_correct = st.selectbox("Accuracy Filter", correct_opt)
            
            # Filter logic
            df_filtered = df_preds.copy()
            if selected_tourn != "All World Cups":
                df_filtered = df_filtered[df_filtered['Tournament'] == selected_tourn]
            if search_team:
                df_filtered = df_filtered[
                    df_filtered['Team A'].str.contains(search_team, case=False) |
                    df_filtered['Team B'].str.contains(search_team, case=False)
                ]
            if selected_correct == "Correct Only":
                df_filtered = df_filtered[df_filtered['Correct?'] == "Yes"]
            elif selected_correct == "Incorrect Only":
                df_filtered = df_filtered[df_filtered['Correct?'] == "No"]
                
            # Metric
            total_m = len(df_filtered)
            if total_m > 0:
                correct_m = len(df_filtered[df_filtered['Correct?'] == "Yes"])
                acc = correct_m / total_m
                st.metric("Selection Accuracy", f"{acc:.1%}", f"{correct_m} / {total_m} Matches")
                
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)
        else:
            st.warning("Prediction database CSV not found.")

with col_e2:
    with st.expander("📈 Historical Predictions & Upset Success (World Cups 2018 & 2022)"):
        sum_path = os.path.join(BASE_DIR, 'data', 'processed', 'backtest_summary.csv')
        upset_path = os.path.join(BASE_DIR, 'data', 'processed', 'backtest_upsets.csv')
        
        if os.path.exists(sum_path) and os.path.exists(upset_path):
            st.subheader("Our Predictions vs Simple Rankings Predictor")
            df_sum = pd.read_csv(sum_path)
            df_sum.columns = ['Tournament', 'Total Matches', 'Our Prediction Accuracy', 'Rankings Predictor Accuracy']
            df_sum['Our Prediction Accuracy'] = df_sum['Our Prediction Accuracy'].apply(lambda x: f"{x:.1%}")
            df_sum['Rankings Predictor Accuracy'] = df_sum['Rankings Predictor Accuracy'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_sum, use_container_width=True, hide_index=True)
            
            st.subheader("Giant Killings & Upsets We Called Correctly")
            df_ups = pd.read_csv(upset_path)
            df_ups.columns = ['Year', 'Matchup', 'Winner', 'FIFA Ranks Comparison', 'Our Predicted Win Probability']
            st.dataframe(df_ups, use_container_width=True, hide_index=True)
        else:
            st.warning("Backtest report files missing. Please run `python src/backtest.py` to generate them.")
