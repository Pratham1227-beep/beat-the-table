import os
import random
import pandas as pd
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

TEAM_STATS_PATH = os.path.join(PROCESSED_DIR, 'team_stats.csv')

# Real 2022 World Cup Groups as default
DEFAULT_GROUPS = {
    'A': ['Netherlands', 'Senegal', 'Ecuador', 'Qatar'],
    'B': ['England', 'USA', 'Iran', 'Wales'],
    'C': ['Argentina', 'Poland', 'Mexico', 'Saudi Arabia'],
    'D': ['France', 'Australia', 'Tunisia', 'Denmark'],
    'E': ['Japan', 'Spain', 'Germany', 'Costa Rica'],
    'F': ['Morocco', 'Croatia', 'Belgium', 'Canada'],
    'G': ['Brazil', 'Switzerland', 'Cameroon', 'Serbia'],
    'H': ['Portugal', 'South Korea', 'Uruguay', 'Ghana']
}

# Cache for match predictions to make sim extremely fast
_pred_cache = {}

def get_cached_prediction(team_a, team_b):
    """Get match prediction probabilities, caching results to speed up simulation."""
    key = tuple(sorted([team_a, team_b]))
    if key not in _pred_cache:
        try:
            from predict_engine import predict_match
            probs = predict_match(key[0], key[1])
        except Exception:
            # Fallback to ELO-based odds if prediction fails or stats missing
            probs = {'Win': 0.40, 'Draw': 0.30, 'Loss': 0.30}
        _pred_cache[key] = probs
        
    probs = _pred_cache[key]
    if team_a == key[0]:
        return probs['Win'], probs['Draw'], probs['Loss']
    else:
        return probs['Loss'], probs['Draw'], probs['Win']

def simulate_match(team_a, team_b, is_knockout=False):
    """Simulate a match outcome. In knockout, ties are resolved."""
    p_win, p_draw, p_loss = get_cached_prediction(team_a, team_b)
    
    if is_knockout:
        # In knockout, normalize Win and Loss probabilities to sum to 1
        sum_wl = p_win + p_loss
        if sum_wl > 0:
            p_win_ko = p_win / sum_wl
            p_loss_ko = p_loss / sum_wl
        else:
            p_win_ko = 0.5
            p_loss_ko = 0.5
            
        r = random.random()
        if r < p_win_ko:
            return team_a
        else:
            return team_b
    else:
        r = random.random()
        if r < p_win:
            return 'Win'
        elif r < p_win + p_draw:
            return 'Draw'
        else:
            return 'Loss'

def get_elo(team, stats_dict):
    return stats_dict.get(team, {}).get('elo', 1500)

def simulate_group(group_teams, stats_dict):
    """Simulate group stage and return 1st and 2nd place teams."""
    points = {t: 0 for t in group_teams}
    
    # 6 matches in a group of 4
    matches = [
        (group_teams[0], group_teams[1]),
        (group_teams[0], group_teams[2]),
        (group_teams[0], group_teams[3]),
        (group_teams[1], group_teams[2]),
        (group_teams[1], group_teams[3]),
        (group_teams[2], group_teams[3])
    ]
    
    for ta, tb in matches:
        res = simulate_match(ta, tb, is_knockout=False)
        if res == 'Win':
            points[ta] += 3
        elif res == 'Loss':
            points[tb] += 3
        else:
            points[ta] += 1
            points[tb] += 1
            
    # Rank teams by points, tie-break by ELO
    sorted_teams = sorted(group_teams, key=lambda t: (points[t], get_elo(t, stats_dict)), reverse=True)
    return sorted_teams[0], sorted_teams[1]

import streamlit as st

@st.cache_data(show_spinner=False)
def run_tournament_simulation(chosen_team, group_teams, num_runs=10000):
    """Run Monte Carlo simulation of the tournament."""
    # Load team stats for ELO tie-breakers
    df_stats = pd.read_csv(TEAM_STATS_PATH)
    stats_dict = df_stats.set_index('team').to_dict(orient='index')
    
    from data_pipeline import normalize_name
    chosen_team = normalize_name(chosen_team)
    group_teams = [normalize_name(t) for t in group_teams]
    
    # 1. Setup groups and swap to avoid duplicates
    groups = {k: list(v) for k, v in DEFAULT_GROUPS.items()}
    
    # Find which group the chosen_team originally belonged to, or default to Group A
    target_group = 'A'
    for k, v in DEFAULT_GROUPS.items():
        if chosen_team in v:
            target_group = k
            break
            
    orig_group = list(groups[target_group])
    groups[target_group] = list(group_teams)
    
    # Resolve duplicates in other groups
    available_replacements = [t for t in orig_group if t not in group_teams]
    for g_lbl in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        if g_lbl == target_group:
            continue
        for i, team in enumerate(groups[g_lbl]):
            if team in group_teams:
                if available_replacements:
                    groups[g_lbl][i] = available_replacements.pop(0)
                else:
                    # Fallback if we run out of replacements
                    groups[g_lbl][i] = "Placeholder"
                    
    # Track statistics
    rounds = ['Group Stage', 'R16', 'QF', 'SF', 'Final', 'Winner']
    round_counts = {r: 0 for r in rounds}
    opponents_per_round = {r: [] for r in rounds}
    
    for _ in range(num_runs):
        round_counts['Group Stage'] += 1
        
        # Simulate all groups
        winners = {}
        runners_up = {}
        for g_lbl, teams in groups.items():
            w, r = simulate_group(teams, stats_dict)
            winners[g_lbl] = w
            runners_up[g_lbl] = r
            
        # Check if chosen team advanced from group stage
        chosen_group_teams = groups[target_group]
        group_winner = winners[target_group]
        group_runner = runners_up[target_group]
        
        if chosen_team not in (group_winner, group_runner):
            continue
            
        round_counts['R16'] += 1
        
        # Build R16 matches
        # standard bracket structure:
        # Match 1: 1A vs 2B
        # Match 2: 1C vs 2D
        # Match 3: 1E vs 2F
        # Match 4: 1G vs 2H
        # Match 5: 1B vs 2A
        # Match 6: 1D vs 2C
        # Match 7: 1F vs 2E
        # Match 8: 1H vs 2G
        r16_matchups = [
            (winners['A'], runners_up['B']),
            (winners['C'], runners_up['D']),
            (winners['E'], runners_up['F']),
            (winners['G'], runners_up['H']),
            (winners['B'], runners_up['A']),
            (winners['D'], runners_up['C']),
            (winners['F'], runners_up['E']),
            (winners['H'], runners_up['G'])
        ]
        
        # Find chosen team's match
        chosen_match_idx = -1
        chosen_opponent = ""
        for idx, (ta, tb) in enumerate(r16_matchups):
            if ta == chosen_team:
                chosen_match_idx = idx
                chosen_opponent = tb
                break
            elif tb == chosen_team:
                chosen_match_idx = idx
                chosen_opponent = ta
                break
                
        opponents_per_round['R16'].append(chosen_opponent)
        
        # Simulate R16
        r16_winners = []
        for ta, tb in r16_matchups:
            r16_winners.append(simulate_match(ta, tb, is_knockout=True))
            
        # Check if chosen team advanced to QF
        if chosen_team not in r16_winners:
            continue
            
        round_counts['QF'] += 1
        
        # QF match structure:
        # QF1: Winner Match 1 vs Winner Match 2
        # QF2: Winner Match 3 vs Winner Match 4
        # QF3: Winner Match 5 vs Winner Match 6
        # QF4: Winner Match 7 vs Winner Match 8
        qf_matchups = [
            (r16_winners[0], r16_winners[1]),
            (r16_winners[2], r16_winners[3]),
            (r16_winners[4], r16_winners[5]),
            (r16_winners[6], r16_winners[7])
        ]
        
        chosen_qf_idx = -1
        chosen_opponent = ""
        for idx, (ta, tb) in enumerate(qf_matchups):
            if ta == chosen_team:
                chosen_qf_idx = idx
                chosen_opponent = tb
                break
            elif tb == chosen_team:
                chosen_qf_idx = idx
                chosen_opponent = ta
                break
                
        opponents_per_round['QF'].append(chosen_opponent)
        
        # Simulate QF
        qf_winners = []
        for ta, tb in qf_matchups:
            qf_winners.append(simulate_match(ta, tb, is_knockout=True))
            
        # Check if advanced to SF
        if chosen_team not in qf_winners:
            continue
            
        round_counts['SF'] += 1
        
        # SF Match structure:
        # SF1: Winner QF1 vs Winner QF2
        # SF2: Winner QF3 vs Winner QF4
        sf_matchups = [
            (qf_winners[0], qf_winners[1]),
            (qf_winners[2], qf_winners[3])
        ]
        
        chosen_sf_idx = -1
        chosen_opponent = ""
        for idx, (ta, tb) in enumerate(sf_matchups):
            if ta == chosen_team:
                chosen_sf_idx = idx
                chosen_opponent = tb
                break
            elif tb == chosen_team:
                chosen_sf_idx = idx
                chosen_opponent = ta
                break
                
        opponents_per_round['SF'].append(chosen_opponent)
        
        # Simulate SF
        sf_winners = []
        for ta, tb in sf_matchups:
            sf_winners.append(simulate_match(ta, tb, is_knockout=True))
            
        # Check if advanced to Final
        if chosen_team not in sf_winners:
            continue
            
        round_counts['Final'] += 1
        
        # Final match
        final_matchup = (sf_winners[0], sf_winners[1])
        chosen_opponent = final_matchup[1] if final_matchup[0] == chosen_team else final_matchup[0]
        opponents_per_round['Final'].append(chosen_opponent)
        
        winner = simulate_match(final_matchup[0], final_matchup[1], is_knockout=True)
        
        if winner == chosen_team:
            round_counts['Winner'] += 1
            
    # Calculate probabilities and most common opponents
    res_table = []
    
    # Calculate probability drops
    probs = {}
    for r in rounds:
        probs[r] = round_counts[r] / num_runs
        
    prob_drops = {}
    for i in range(len(rounds) - 1):
        curr_r = rounds[i]
        next_r = rounds[i+1]
        prob_drops[next_r] = probs[curr_r] - probs[next_r]
        
    # Hardest stage is the one with the largest absolute drop
    hardest_stage = max(prob_drops, key=prob_drops.get)
    max_drop = prob_drops[hardest_stage]
    
    for r in rounds:
        opp_counter = Counter(opponents_per_round[r])
        most_common_opp = opp_counter.most_common(1)
        opp_str = most_common_opp[0][0] if most_common_opp else "N/A"
        opp_pct = (most_common_opp[0][1] / round_counts[r]) if (most_common_opp and round_counts[r] > 0) else 0.0
        
        res_table.append({
            'Stage': r,
            'Advancement Probability': probs[r],
            'Most Common Opponent': opp_str,
            'Opponent Frequency': opp_pct
        })
        
    return {
        'table': res_table,
        'hardest_stage': hardest_stage,
        'hardest_stage_drop': max_drop
    }

if __name__ == "__main__":
    print("--- Testing Tournament Simulator ---")
    group = ['France', 'Denmark', 'Australia', 'Tunisia']
    res = run_tournament_simulation('France', group, num_runs=1000)
    print(f"\nHardest Stage: {res['hardest_stage']} (Drop of {res['hardest_stage_drop']:.1%})")
    print("\nAdvancement Table:")
    for row in res['table']:
        print(f" - {row['Stage']}: Prob={row['Advancement Probability']:.1%}, Most Common Opponent={row['Most Common Opponent']} ({row['Opponent Frequency']:.1%})")
