import os
import csv
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PLAYERS_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'players_fifa23.csv')

# Position mapping for role compatibility
POSITION_MAPS = {
    'GK': ['GK'],
    'CB': ['CB', 'LCB', 'RCB'],
    'LB': ['LB', 'LWB', 'LM', 'CB'],
    'RB': ['RB', 'RWB', 'RM', 'CB'],
    'LWB': ['LWB', 'LB', 'LM'],
    'RWB': ['RWB', 'RB', 'RM'],
    'CDM': ['CDM', 'CM', 'LDM', 'RDM'],
    'CAM': ['CAM', 'CM', 'LAM', 'RAM'],
    'CM': ['CM', 'CDM', 'CAM', 'LM', 'RM'],
    'LW': ['LW', 'LM', 'LW', 'LF', 'ST'],
    'RW': ['RW', 'RM', 'RW', 'RF', 'ST'],
    'ST': ['ST', 'CF', 'LS', 'RS', 'LF', 'RF']
}

import streamlit as st

@st.cache_data(show_spinner=False)
def get_squad_players(team_name):
    """Load and return players for a given team name from players_fifa23.csv."""
    from data_pipeline import normalize_name
    target_team = normalize_name(team_name)
    
    players = []
    if not os.path.exists(PLAYERS_PROCESSED):
        print(f"Warning: Player database {PLAYERS_PROCESSED} not found. Returning empty squad.")
        return players
        
    try:
        with open(PLAYERS_PROCESSED, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Extract column indices
            indices = {
                'name': header.index('short_name'),
                'positions': header.index('player_positions'),
                'overall': header.index('overall'),
                'tags': header.index('player_tags') if 'player_tags' in header else -1,
                'traits': header.index('player_traits') if 'player_traits' in header else -1,
                'pace': header.index('pace') if 'pace' in header else -1,
                'shooting': header.index('shooting') if 'shooting' in header else -1,
                'passing': header.index('passing') if 'passing' in header else -1,
                'dribbling': header.index('dribbling') if 'dribbling' in header else -1,
                'defending': header.index('defending') if 'defending' in header else -1,
                'physic': header.index('physic') if 'physic' in header else -1,
                'heading': header.index('attacking_heading_accuracy') if 'attacking_heading_accuracy' in header else -1,
                'composure': header.index('mentality_composure') if 'mentality_composure' in header else -1,
                'stamina': header.index('power_stamina') if 'power_stamina' in header else -1,
                'nationality': header.index('nationality_name'),
                'face_url': header.index('player_face_url') if 'player_face_url' in header else -1
            }
            
            for row in reader:
                # Basic check
                if len(row) <= max(indices.values()):
                    continue
                    
                nat = normalize_name(row[indices['nationality']])
                if nat != target_team:
                    continue
                    
                # Parse fields
                try:
                    overall = int(row[indices['overall']])
                    raw_positions = [pos.strip() for pos in row[indices['positions']].split(',')]
                    
                    # Safe numeric parse helpers
                    def safe_int(idx, default=60):
                        if idx == -1 or idx >= len(row) or not row[idx].strip():
                            return default
                        try:
                            return int(float(row[idx]))
                        except ValueError:
                            return default
                            
                    players.append({
                        'name': row[indices['name']],
                        'positions': raw_positions,
                        'overall': overall,
                        'tags': row[indices['tags']] if indices['tags'] != -1 else "",
                        'traits': row[indices['traits']] if indices['traits'] != -1 else "",
                        'pace': safe_int(indices['pace']),
                        'shooting': safe_int(indices['shooting']),
                        'passing': safe_int(indices['passing']),
                        'dribbling': safe_int(indices['dribbling']),
                        'defending': safe_int(indices['defending']),
                        'physic': safe_int(indices['physic']),
                        'heading': safe_int(indices['heading']),
                        'composure': safe_int(indices['composure']),
                        'stamina': safe_int(indices['stamina']),
                        'face_url': row[indices['face_url']] if indices['face_url'] != -1 else ""
                    })
                except Exception:
                    pass
    except Exception as e:
        print(f"Error loading squad: {e}")
        
    # If no players found, create a decent mock squad so the app doesn't crash
    if not players:
        print(f"Warning: No players found in database for '{team_name}'. Creating a mock squad.")
        mock_positions = ['GK', 'CB', 'CB', 'LB', 'RB', 'CDM', 'CM', 'CAM', 'LW', 'RW', 'ST', 'ST', 'CB', 'CM']
        for i, pos in enumerate(mock_positions):
            players.append({
                'name': f"Player {i+1} ({pos})",
                'positions': [pos],
                'overall': 70,
                'tags': "",
                'traits': "",
                'pace': 70,
                'shooting': 70,
                'passing': 70,
                'dribbling': 70,
                'defending': 70,
                'physic': 70,
                'heading': 70,
                'composure': 70,
                'stamina': 70,
                'face_url': 'https://cdn.sofifa.net/players/notfound_0.png'
            })
            
    return players

def apply_adjustment_rules(player, opponent_profile):
    """Apply the 6 tactical rules to a player's rating and return the adjusted rating and rationale."""
    base = player['overall']
    adj = 0
    rationales = []
    
    # Pre-calculate flags
    is_cb = any(p in POSITION_MAPS['CB'] for p in player['positions'])
    is_fb = any(p in POSITION_MAPS['LB'] or p in POSITION_MAPS['RB'] for p in player['positions'])
    is_mid = any(p in POSITION_MAPS['CM'] or p in POSITION_MAPS['CDM'] or p in POSITION_MAPS['CAM'] for p in player['positions'])
    is_att = any(p in POSITION_MAPS['ST'] or p in POSITION_MAPS['LW'] or p in POSITION_MAPS['RW'] for p in player['positions'])
    
    # Rule 1: Aerial Threat (Defense)
    if opponent_profile.get('aerial_threat'):
        if is_cb:
            has_heading = "header" in player['traits'].lower() or "aerial" in player['tags'].lower() or player['heading'] > 78
            if has_heading:
                adj += 5
                rationales.append("Rule 1 (Aerial Defense): +5 for dominant aerial capability.")
            elif player['defending'] < 72:
                adj -= 3
                rationales.append("Rule 1 (Aerial Defense): -3 due to physical/aerial vulnerabilities.")
                
    # Rule 2: High Press (Midfield)
    if opponent_profile.get('high_press'):
        if is_mid:
            is_press_resistant = player['composure'] > 78 or player['dribbling'] > 78
            if is_press_resistant:
                adj += 4
                rationales.append("Rule 2 (Press Resistance): +4 for exceptional composure and ball retention under press.")
            else:
                adj -= 2
                rationales.append("Rule 2 (Press Vulnerability): -2 due to low composure/dribbling splits.")
                
    # Rule 3: Counter Pace (Defense)
    if opponent_profile.get('counter_pace'):
        if is_cb or is_fb:
            if player['pace'] > 78:
                adj += 5
                rationales.append("Rule 3 (Recovery Pace): +5 for recovery speed to stop counter-attacks.")
            elif player['pace'] < 65:
                adj -= 4
                rationales.append("Rule 3 (Pace Deficit): -4 due to lack of foot speed to tracking back.")
                
    # Rule 4: Possession Heavy (Midfield & Attack)
    if opponent_profile.get('possession_heavy'):
        if is_mid or is_att:
            if player['passing'] > 78 or player['dribbling'] > 78:
                adj += 3
                rationales.append("Rule 4 (Technical Build): +3 for passing accuracy and close control against possession-heavy setup.")
                
    # Rule 5: Midfield Scrap / Physicality
    if opponent_profile.get('high_press') and (is_mid or is_cb):
        if player['physic'] > 80 or player['stamina'] > 80:
            adj += 3
            rationales.append("Rule 5 (Physical Stature): +3 for stamina and physical strength in mid block.")
            
    # Rule 6: Low Block Breaker (Attack)
    if opponent_profile.get('possession_heavy') and is_att:
        if player['shooting'] > 80:
            adj += 3
            rationales.append("Rule 6 (Clinical Finish): +3 for high shooting capability to break down deep blocks.")
            
    rationale = "; ".join(rationales) if rationales else "Tactically suited. Maintained base rating."
    return base + adj, rationale

def get_starting_xi(team_name, opponent_profile):
    """Select the best formation and Starting XI for the team against the opponent's style profile."""
    # 1. Determine Formation
    # Prefer defensive back-five against counter pace
    if opponent_profile.get('counter_pace'):
        formation = '5-3-2'
        roles_required = [
            ('GK', 'GK'),
            ('CB', 'CB'), ('CB', 'CB'), ('CB', 'CB'),
            ('LWB', 'LB'), ('RWB', 'RB'),
            ('CM', 'CM'), ('CM', 'CM'), ('CM', 'CM'),
            ('ST', 'ST'), ('ST', 'ST')
        ]
    # Prefer packed midfield against possession heavy
    elif opponent_profile.get('possession_heavy'):
        formation = '3-5-2'
        roles_required = [
            ('GK', 'GK'),
            ('CB', 'CB'), ('CB', 'CB'), ('CB', 'CB'),
            ('LWB', 'LB'), ('RWB', 'RB'),
            ('CM', 'CM'), ('CM', 'CM'), ('CM', 'CM'),
            ('ST', 'ST'), ('ST', 'ST')
        ]
    # Prefer double pivot against high press
    elif opponent_profile.get('high_press'):
        formation = '4-2-3-1'
        roles_required = [
            ('GK', 'GK'),
            ('CB', 'CB'), ('CB', 'CB'),
            ('LB', 'LB'), ('RB', 'RB'),
            ('CDM', 'CDM'), ('CDM', 'CDM'),
            ('CAM', 'CAM'),
            ('LW', 'LW'), ('RW', 'RW'),
            ('ST', 'ST')
        ]
    # Default formation
    else:
        formation = '4-3-3'
        roles_required = [
            ('GK', 'GK'),
            ('CB', 'CB'), ('CB', 'CB'),
            ('LB', 'LB'), ('RB', 'RB'),
            ('CM', 'CM'), ('CM', 'CM'), ('CM', 'CM'),
            ('LW', 'LW'), ('ST', 'ST'), ('RW', 'RW')
        ]
        
    # 2. Load players and calculate adjusted ratings
    players = get_squad_players(team_name)
    processed_players = []
    for p in players:
        adj_rating, rationale = apply_adjustment_rules(p, opponent_profile)
        processed_players.append({
            'player': p,
            'adjusted_rating': adj_rating,
            'rationale': rationale
        })
        
    # 3. Select XI based on role compatibility (prevent double selection)
    selected_names = set()
    starting_xi = []
    
    # Sort roles to select positions with lower pool sizes first to avoid conflicts
    # Order: GK, ST, LW, RW, CAM, CDM, CM, LWB, RWB, LB, RB, CB
    role_order = ['GK', 'ST', 'LW', 'RW', 'CAM', 'CDM', 'CM', 'LWB', 'RWB', 'LB', 'RB', 'CB']
    sorted_roles = sorted(roles_required, key=lambda x: role_order.index(x[0]))
    
    role_selections = {}
    
    for role_name, pos_key in sorted_roles:
        # Find best available player for this position
        compatible_positions = POSITION_MAPS[pos_key]
        
        candidates = []
        for p in processed_players:
            p_data = p['player']
            if p_data['name'] in selected_names:
                continue
            # check if any position matches compatible positions
            if any(pos in compatible_positions for pos in p_data['positions']):
                candidates.append(p)
                
        # If no compatible player available, take any highest rated available player
        if not candidates:
            candidates = [p for p in processed_players if p['player']['name'] not in selected_names]
            
        if candidates:
            # Sort candidates by adjusted rating
            candidates = sorted(candidates, key=lambda x: x['adjusted_rating'], reverse=True)
            chosen = candidates[0]
            selected_names.add(chosen['player']['name'])
            
            role_selections[role_name + f"_{len([r for r in role_selections if r.startswith(role_name)])}"] = {
                'role': role_name,
                'name': chosen['player']['name'],
                'base_rating': chosen['player']['overall'],
                'adjusted_rating': chosen['adjusted_rating'],
                'rationale': chosen['rationale']
            }
            
    # Format starting XI back to the original layout order for display (e.g. GK, Def, Mid, Att)
    formatted_xi = []
    for role_name, pos_key in roles_required:
        # Find the matching selection
        for key, val in role_selections.items():
            if val['role'] == role_name and val not in formatted_xi:
                formatted_xi.append(val)
                break
                
    return {
        'formation': formation,
        'starting_xi': formatted_xi
    }

if __name__ == "__main__":
    print("--- Testing Lineup Engine ---")
    opponent = {
        'high_press': True,
        'aerial_threat': True,
        'counter_pace': True,
        'possession_heavy': False
    }
    res = get_starting_xi("France", opponent)
    print(f"Chosen Formation: {res['formation']}")
    print("\nStarting XI:")
    for player in res['starting_xi']:
        print(f" - [{player['role']}] {player['name']} (OVR {player['base_rating']} -> ADJ {player['adjusted_rating']})")
        print(f"   Rationale: {player['rationale']}")
