import os
import pickle
import pandas as pd
import numpy as np
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

TEAM_STATS_PATH = os.path.join(PROCESSED_DIR, 'team_stats.csv')
RESULTS_PATH = os.path.join(RAW_DIR, 'results.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')

# Feature descriptions for natural language generation
FEATURE_DESCS = {
    'elo_diff': {
        'pos': "superior squad power rating",
        'neg': "weaker power rating compared to their opponents"
    },
    'form_diff': {
        'pos': "hot run of form in recent games",
        'neg': "recent dip in form and results"
    },
    'xg_diff': {
        'pos': "more dangerous attack that creates high-quality scoring chances",
        'neg': "lack of fire-power and creativity in the final third"
    },
    'xga_diff': {
        'pos': "tight, organized defense that limits opponent chances",
        'neg': "leaky defense that gives away too many easy goal opportunities"
    },
    'squad_rating_diff': {
        'pos': "superior individual player ratings and squad depth",
        'neg': "weaker overall player ratings across the squad"
    },
    'h2h_home': {
        'pos': "excellent head-to-head record in past meetings",
        'neg': "struggles and poor history against this specific team"
    },
    'h2h_draw': {
        'pos': "frequent draws when these two teams match up",
        'neg': "very few draws in their previous matches"
    },
    'h2h_away': {
        'pos': "poor track record when playing against this opponent",
        'neg': "great record of holding off this opponent in past matches"
    }
}

import streamlit as st

@st.cache_data(show_spinner=False)
def explain_prediction(team_a, team_b):
    """Compute SHAP values for the match prediction and return a commentary paragraph."""
    # 1. Load model and features
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train the model first.")
        
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        features = data['features']
        
    # 2. Get team stats to construct the vector
    df_stats = pd.read_csv(TEAM_STATS_PATH)
    stats_dict = df_stats.set_index('team').to_dict(orient='index')
    
    from data_pipeline import normalize_name
    team_a_norm = normalize_name(team_a)
    team_b_norm = normalize_name(team_b)
    
    if team_a_norm not in stats_dict or team_b_norm not in stats_dict:
        raise ValueError(f"One or both teams ('{team_a}', '{team_b}') not found in team stats.")
        
    stats_a = stats_dict[team_a_norm]
    stats_b = stats_dict[team_b_norm]
    
    # Read results to compute overall H2H
    df_results = pd.read_csv(RESULTS_PATH)
    df_results['home_team'] = df_results['home_team'].apply(normalize_name)
    df_results['away_team'] = df_results['away_team'].apply(normalize_name)
    
    past_meetings = df_results[
        ((df_results['home_team'] == team_a_norm) & (df_results['away_team'] == team_b_norm)) |
        ((df_results['home_team'] == team_b_norm) & (df_results['away_team'] == team_a_norm))
    ]
    
    if len(past_meetings) >= 5:
        wins_a = 0
        wins_b = 0
        draws = 0
        for _, row in past_meetings.iterrows():
            h_team = row['home_team']
            a_team = row['away_team']
            h_score = row['home_score']
            a_score = row['away_score']
            if pd.isna(h_score) or pd.isna(a_score):
                continue
            if h_score > a_score:
                if h_team == team_a_norm:
                    wins_a += 1
                else:
                    wins_b += 1
            elif h_score < a_score:
                if a_team == team_a_norm:
                    wins_a += 1
                else:
                    wins_b += 1
            else:
                draws += 1
        total = len(past_meetings)
        h2h_home = wins_a / total
        h2h_draw = draws / total
        h2h_away = wins_b / total
    else:
        h2h_home = 0.333
        h2h_draw = 0.333
        h2h_away = 0.333
        
    feat_vector = {
        'elo_diff': stats_a['elo'] - stats_b['elo'],
        'form_diff': stats_a['form'] - stats_b['form'],
        'xg_diff': stats_a['xg'] - stats_b['xg'],
        'xga_diff': stats_a['xga'] - stats_b['xga'],
        'squad_rating_diff': stats_a['squad_rating'] - stats_b['squad_rating'],
        'h2h_home': h2h_home,
        'h2h_draw': h2h_draw,
        'h2h_away': h2h_away
    }
    
    df_vector = pd.DataFrame([feat_vector])[features]
    
    # 3. Predict probability to get context
    probs = model.predict_proba(df_vector)[0]
    p_win = probs[2]
    p_draw = probs[1]
    p_loss = probs[0]
    
    # 4. Compute SHAP values for class 2 (Win for team_a)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(df_vector) # Shape: (1, 8, 3)
    sh = shap_vals[0, :, 2] # extract class 2 shap values for the 1st sample
    
    # 5. Extract top 3 features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(sh))[::-1]
    top_3 = []
    for i in range(3):
        idx = sorted_indices[i]
        feat_name = features[idx]
        val = df_vector.iloc[0, idx]
        shap_val = sh[idx]
        top_3.append((feat_name, val, shap_val))
        
    # 6. Generate natural language explanation
    # Categorize features as positive or negative for team_a
    pos_drivers = []
    neg_drivers = []
    
    for feat_name, val, shap_val in top_3:
        # Determine if it was positive or negative impact on team_a winning
        is_pos = shap_val > 0
        desc_dict = FEATURE_DESCS[feat_name]
        
        # Check actual value or difference direction to pick description
        # For differences: positive difference means team_a is better
        # For xga_diff, negative means team_a is better, but let's check value direction:
        if feat_name in ('elo_diff', 'form_diff', 'xg_diff', 'squad_rating_diff'):
            is_better = val > 0
        elif feat_name == 'xga_diff':
            is_better = val < 0  # lower xga is better
        elif feat_name == 'h2h_home':
            is_better = val > 0.35
        elif feat_name == 'h2h_away':
            is_better = val < 0.30
        else:
            is_better = val > 0.33
            
        desc = desc_dict['pos'] if is_better else desc_dict['neg']
        
        driver_info = {
            'name': feat_name,
            'desc': desc,
            'shap': shap_val,
            'is_pos': is_pos
        }
        
        if is_pos:
            pos_drivers.append(driver_info)
        else:
            neg_drivers.append(driver_info)
            
    # Assemble narrative
    sentences = []
    
    # Sentence 1: Prediction summary
    if p_win > 0.50:
        sentences.append(f"Our predictor strongly backs {team_a} to take home all 3 points, holding a solid {p_win:.1%} chance of winning.")
    elif p_win > 0.38:
        sentences.append(f"Expect a tight, physical match between {team_a} and {team_b}, with the forecast giving {team_a} a slim {p_win:.1%} advantage.")
    elif p_loss > 0.45:
        sentences.append(f"{team_a} has a real mountain to climb in this fixture, with {team_b} heavily backed at a {p_loss:.1%} chance of winning.")
    else:
        sentences.append(f"A very even, neck-and-neck match is on the cards, with a draw ({p_draw:.1%}) looking like a highly likely result.")
        
    # Sentence 2: Primary positive drivers
    if pos_drivers:
        primary = pos_drivers[0]
        sentences.append(f"{team_a}'s main source of confidence comes from their {primary['desc']}.")
        if len(pos_drivers) > 1:
            secondary = pos_drivers[1]
            sentences.append(f"This is further backed up by their {secondary['desc']}.")
    else:
        sentences.append(f"Interestingly, {team_a} doesn't have any major statistical advantages to brag about before kick-off.")
        
    # Sentence 3: Countervailing negative drivers / opponent advantages
    if neg_drivers:
        primary_neg = neg_drivers[0]
        sentences.append(f"However, {team_b} will look to exploit {team_a}'s {primary_neg['desc']}, which is the biggest obstacle holding back {team_a}'s win chances.")
        if len(neg_drivers) > 1:
            secondary_neg = neg_drivers[1]
            sentences.append(f"On top of that, {team_b} can capitalize on {team_a}'s {secondary_neg['desc']}.")
    else:
        sentences.append(f"To make matters worse for {team_b}, they have very little past history or form on their side to disrupt {team_a}'s game plan.")
        
    # Sentence 4: Closing analyst quote
    sentences.append(f"Expect these key matchups on the pitch to decide who walks away with the win.")
    
    commentary = " ".join(sentences)
    
    return {
        'top_3_features': [
            {
                'feature': f[0],
                'value': float(f[1]),
                'shap_val': float(f[2]),
                'direction': 'positive' if f[2] > 0 else 'negative'
            } for f in top_3
        ],
        'explanation_text': commentary
    }

if __name__ == "__main__":
    # Test execution
    print("--- Testing Explanation Layer ---")
    try:
        res = explain_prediction("Iran", "Qatar")
        print("\nTop 3 SHAP Features:")
        for feat in res['top_3_features']:
            print(f" - {feat['feature']}: val={feat['value']:.2f}, SHAP={feat['shap_val']:.4f} ({feat['direction']})")
        print("\nCommentary:")
        print(res['explanation_text'])
    except Exception as e:
        print(f"Error testing explain.py: {e}")
