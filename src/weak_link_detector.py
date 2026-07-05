"""
weak_link_detector.py
─────────────────────
Finds the most dangerous player-vs-player mismatches in a match:
given Team A's Starting XI, it identifies which of their players is most
at risk from Team B's best attackers / midfielders.

Logic:
  • For each defensive role in Team A's XI, we find the strongest opponent
    player whose position directly threatens that role (e.g. RW vs LB).
  • The vulnerability score = (opponent's attacking stat - my defensive stat).
  • The top-N highest scores are returned as "Weak Links".
"""

import os
import pandas as pd
import streamlit as st

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYERS_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'players_fifa23.csv')

# ── Position → threat matchup config ─────────────────────────────────────────
# For each role in Team A, define:
#   threat_roles : which opponent positions target this slot
#   my_stat      : the most relevant defensive/tactical stat for Team A's player
#   their_stat   : the most relevant attacking/tactical stat for Team B's threat
#   severity_label: phrase used when describing the matchup
MATCHUP_CFG = {
    'GK': {
        'threat_roles': ['ST', 'CF', 'LW', 'RW', 'CAM'],
        'my_stat':      'overall',      # GKs don't have a clean "defending" stat
        'their_stat':   'shooting',
        'desc': (
            "{threat} ({threat_role}) is a clinical finisher with a shooting rating "
            "of {their_val}. Your keeper {mine} (overall {my_val}) will be tested hard "
            "from shots inside and outside the box. One mistake could be costly."
        ),
    },
    'CB': {
        'threat_roles': ['ST', 'CF', 'LW', 'RW'],
        'my_stat':      'pace',
        'their_stat':   'pace',
        'desc': (
            "{threat} ({threat_role}, pace {their_val}) is significantly faster than "
            "{mine} ({role}, pace {my_val}). Runs in behind the defensive line — especially "
            "on the left channel — are the biggest danger for your back four."
        ),
    },
    'LB': {
        'threat_roles': ['RW', 'RM', 'ST'],
        'my_stat':      'defending',
        'their_stat':   'dribbling',
        'desc': (
            "{threat} ({threat_role}) has elite dribbling ({their_val}) and will "
            "consistently put {mine} ({role}, defending {my_val}) under pressure in 1v1s "
            "on the right flank. Expect crosses and cut-ins from this side."
        ),
    },
    'RB': {
        'threat_roles': ['LW', 'LM', 'ST'],
        'my_stat':      'defending',
        'their_stat':   'dribbling',
        'desc': (
            "{threat} ({threat_role}) with dribbling {their_val} is a constant threat "
            "down the left flank. {mine} ({role}, defending {my_val}) will struggle to "
            "contain the directness and could give away dangerous set-pieces."
        ),
    },
    'LWB': {
        'threat_roles': ['RW', 'RM', 'ST'],
        'my_stat':      'pace',
        'their_stat':   'pace',
        'desc': (
            "As a wing-back, {mine} ({role}) pushes very high. When possession is lost, "
            "{threat} ({threat_role}, pace {their_val}) can spring the counter attack into "
            "the vacated space behind — faster than {mine} (pace {my_val}) can recover."
        ),
    },
    'RWB': {
        'threat_roles': ['LW', 'LM', 'ST'],
        'my_stat':      'pace',
        'their_stat':   'pace',
        'desc': (
            "As a wing-back, {mine} ({role}) pushes very high. When possession is lost, "
            "{threat} ({threat_role}, pace {their_val}) can spring the counter attack into "
            "the vacated space behind — faster than {mine} (pace {my_val}) can recover."
        ),
    },
    'CDM': {
        'threat_roles': ['CAM', 'CM', 'LW', 'RW'],
        'my_stat':      'defending',
        'their_stat':   'dribbling',
        'desc': (
            "{threat} ({threat_role}) has world-class dribbling ({their_val}) and will "
            "probe, nutmeg, and glide past {mine} ({role}, defending {my_val}), opening "
            "direct routes to your back four. This is your most exposed defensive channel."
        ),
    },
    'CM': {
        'threat_roles': ['CAM', 'CM'],
        'my_stat':      'physic',
        'their_stat':   'passing',
        'desc': (
            "{threat} ({threat_role}) pulls the strings with passing {their_val}. "
            "Your {mine} ({role}, physicality {my_val}) will be stretched by diagonal "
            "switches and through-balls, making midfield control very hard to maintain."
        ),
    },
    'CAM': {
        'threat_roles': ['CDM', 'CM'],
        'my_stat':      'physic',
        'their_stat':   'defending',
        'desc': (
            "{threat} ({threat_role}) is a tenacious presser with defending {their_val}. "
            "{mine} ({role}, physicality {my_val}) could be closed down before even "
            "turning — this neutralises your main creative outlet in the final third."
        ),
    },
    'LW': {
        'threat_roles': ['RB', 'RWB', 'CB'],
        'my_stat':      'dribbling',
        'their_stat':   'defending',
        'desc': (
            "{threat} ({threat_role}) defends at {their_val} and will suffocate "
            "{mine} ({role}, dribbling {my_val}) in the wide channel. "
            "Your left attacking lane is effectively shut down — expect {mine} "
            "to be forced inside and lose the 1v1 battle repeatedly."
        ),
    },
    'RW': {
        'threat_roles': ['LB', 'LWB', 'CB'],
        'my_stat':      'dribbling',
        'their_stat':   'defending',
        'desc': (
            "{threat} ({threat_role}) defends at {their_val} and will suffocate "
            "{mine} ({role}, dribbling {my_val}) on the right flank. "
            "Without width from the right, your attack loses one of its main outlets."
        ),
    },
    'ST': {
        'threat_roles': ['CB'],
        'my_stat':      'shooting',
        'their_stat':   'defending',
        'desc': (
            "{mine} ({role}, shooting {my_val}) is up against an elite defender: "
            "{threat} ({threat_role}, defending {their_val}). Expect physical battles, "
            "offside traps, and aggressive pressing — service must be perfect to create chances."
        ),
    },
}

# ── Helper: load squad stats ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_squad(team_name: str) -> pd.DataFrame:
    """Return all FIFA 23 players for a national team as a DataFrame."""
    if not os.path.exists(PLAYERS_CSV):
        return pd.DataFrame()

    cols_needed = [
        'short_name', 'long_name', 'nationality_name',
        'player_positions', 'overall',
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'player_face_url'
    ]
    try:
        df = pd.read_csv(PLAYERS_CSV, usecols=lambda c: c in cols_needed)
    except Exception:
        df = pd.read_csv(PLAYERS_CSV)

    # Keep only the latest entry per player (FIFA 23 has multiple versions)
    df = df[df['nationality_name'] == team_name].copy()
    if df.empty:
        return df

    # Numeric coercion
    for col in ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(60).astype(int)

    # Deduplicate: keep the highest overall for each player name
    name_col = 'short_name' if 'short_name' in df.columns else 'long_name'
    df = df.sort_values('overall', ascending=False).drop_duplicates(subset=[name_col])
    df['_name'] = df[name_col]
    return df.reset_index(drop=True)


def _get_stat(row: pd.Series, stat: str) -> int:
    """Safely get a stat from a player row, defaulting to overall if missing."""
    if stat in row.index and pd.notna(row[stat]):
        return int(row[stat])
    if 'overall' in row.index:
        return int(row['overall'])
    return 60


def _primary_position(pos_string: str) -> str:
    """Return the first listed position (primary) from a comma-separated string."""
    if not isinstance(pos_string, str):
        return 'CM'
    return pos_string.split(',')[0].strip().upper()


# ── Core function ──────────────────────────────────────────────────────────────

def find_weak_links(team_a_xi: list, team_b_name: str, top_n: int = 3) -> list:
    """
    Identify the most dangerous player-vs-player mismatches.

    Parameters
    ----------
    team_a_xi   : list of player dicts from get_starting_xi()
                  Each dict has at least: 'name', 'role', 'base_rating'
    team_b_name : str  — the opposing national team name
    top_n       : int  — number of weak links to return (default 3)

    Returns
    -------
    list of dicts, sorted by vulnerability_score descending, e.g.:
    [
      {
        'my_player'         : 'Varane',
        'my_role'           : 'CB',
        'my_key_stat'       : 'pace',
        'my_stat_val'       : 72,
        'threat_player'     : 'L. Martínez',
        'threat_role'       : 'ST',
        'threat_key_stat'   : 'pace',
        'threat_stat_val'   : 88,
        'vulnerability_score': 16,
        'severity'          : 'Critical',    # Critical / Warning / Caution
        'description'       : '...',
      },
      ...
    ]
    """
    squad_b = _load_squad(team_b_name)

    results = []

    for player in team_a_xi:
        role = player.get('role', 'CM')
        cfg  = MATCHUP_CFG.get(role)
        if cfg is None:
            continue

        my_stat_name   = cfg['my_stat']
        their_stat_name = cfg['their_stat']
        my_stat_val    = player.get('base_rating', 75)   # fallback to base rating

        # Find the best matching threat from team_b
        if squad_b.empty:
            # Fallback: synthesise a generic threat
            best_threat      = None
            best_threat_val  = 75
            best_threat_name = f"A {cfg['threat_roles'][0]}"
            best_threat_pos  = cfg['threat_roles'][0]
            best_threat_face = 'https://cdn.sofifa.net/players/notfound_0.png'
        else:
            threat_roles = cfg['threat_roles']
            # Filter Team B squad to matching positions
            mask = squad_b['player_positions'].apply(
                lambda ps: any(
                    r in [p.strip().upper() for p in str(ps).split(',')]
                    for r in threat_roles
                )
            )
            candidates = squad_b[mask]
            if candidates.empty:
                # Broaden: just pick the highest overall outfield players
                candidates = squad_b[squad_b['player_positions'] != 'GK'].head(5)
            if candidates.empty:
                continue

            # Pick the one with the highest their_stat_name
            best_idx        = candidates[their_stat_name].idxmax() if their_stat_name in candidates.columns else candidates['overall'].idxmax()
            best_threat      = candidates.loc[best_idx]
            best_threat_val  = _get_stat(best_threat, their_stat_name)
            best_threat_name = best_threat['_name']
            best_threat_pos  = _primary_position(best_threat.get('player_positions', threat_roles[0]))
            best_threat_face = best_threat.get('player_face_url', 'https://cdn.sofifa.net/players/notfound_0.png')

        # Vulnerability score: how much does the threat outclass our player?
        vuln_score = best_threat_val - my_stat_val   # can be negative (not a weakness)

        description = cfg['desc'].format(
            mine=player['name'].split(' ')[-1] if ' ' in player['name'] else player['name'],
            role=role,
            my_val=my_stat_val,
            threat=best_threat_name,
            threat_role=best_threat_pos,
            their_val=best_threat_val,
        )

        # Severity banding
        if vuln_score >= 15:
            severity = '🔴 Critical'
        elif vuln_score >= 8:
            severity = '🟠 Warning'
        elif vuln_score >= 1:
            severity = '🟡 Caution'
        else:
            severity = '🟢 Solid'

        results.append({
            'my_player':          player['name'],
            'my_role':            role,
            'my_face':            player.get('face_url', 'https://cdn.sofifa.net/players/notfound_0.png'),
            'my_key_stat':        my_stat_name,
            'my_stat_val':        my_stat_val,
            'threat_player':      best_threat_name,
            'threat_role':        best_threat_pos,
            'threat_face':        best_threat_face,
            'threat_key_stat':    their_stat_name,
            'threat_stat_val':    best_threat_val,
            'vulnerability_score': vuln_score,
            'severity':           severity,
            'description':        description,
        })

    # Sort: highest vulnerability first, then trim
    results.sort(key=lambda r: r['vulnerability_score'], reverse=True)
    return results[:top_n]
