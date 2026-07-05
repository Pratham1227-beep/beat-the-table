import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

RESULTS_PATH = os.path.join(RAW_DIR, 'results.csv')
TEAM_STATS_PATH = os.path.join(PROCESSED_DIR, 'team_stats.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')

def build_features():
    """Build the dataset for modeling from results.csv and team_stats.csv."""
    print("--- Building Prediction Features ---")
    if not os.path.exists(RESULTS_PATH) or not os.path.exists(TEAM_STATS_PATH):
        raise FileNotFoundError("Required data files results.csv or team_stats.csv are missing!")
        
    df_results = pd.read_csv(RESULTS_PATH)
    df_results['date'] = pd.to_datetime(df_results['date'])
    df_results = df_results.sort_values(by='date').reset_index(drop=True)
    
    df_stats = pd.read_csv(TEAM_STATS_PATH)
    # Convert team_stats into a dictionary for quick lookups
    stats_dict = df_stats.set_index('team').to_dict(orient='index')
    
    # Pre-map names in results
    from data_pipeline import normalize_name
    df_results['home_team'] = df_results['home_team'].apply(normalize_name)
    df_results['away_team'] = df_results['away_team'].apply(normalize_name)
    
    # Track head-to-head records chronologically
    h2h_history = {}
    
    feature_rows = []
    
    for idx, row in df_results.iterrows():
        team_a = row['home_team']
        team_b = row['away_team']
        date = row['date']
        
        # Only process if both teams have stats
        if team_a not in stats_dict or team_b not in stats_dict:
            continue
            
        stats_a = stats_dict[team_a]
        stats_b = stats_dict[team_b]
        
        # Calculate Head-to-Head win rates (prior to this match)
        key = frozenset([team_a, team_b])
        prior = h2h_history.get(key, [])
        
        if len(prior) >= 5:
            wins_a = sum(1 for h, a, w in prior if w == team_a)
            wins_b = sum(1 for h, a, w in prior if w == team_b)
            draws = sum(1 for h, a, w in prior if w is None)
            total = len(prior)
            h2h_home = wins_a / total
            h2h_draw = draws / total
            h2h_away = wins_b / total
        else:
            h2h_home = 0.333
            h2h_draw = 0.333
            h2h_away = 0.333
            
        # Update H2H history after computing features
        home_score = row['home_score']
        away_score = row['away_score']
        
        # If score is missing, skip outcomes, but this shouldn't happen for historical matches
        if pd.isna(home_score) or pd.isna(away_score):
            continue
            
        winner = None
        if home_score > away_score:
            winner = team_a
        elif home_score < away_score:
            winner = team_b
            
        if key not in h2h_history:
            h2h_history[key] = []
        h2h_history[key].append((team_a, team_b, winner))
        
        # Outcome from Home perspective: Loss=0, Draw=1, Win=2
        if home_score > away_score:
            outcome = 2
        elif home_score == away_score:
            outcome = 1
        else:
            outcome = 0
            
        feature_rows.append({
            'date': date,
            'elo_diff': stats_a['elo'] - stats_b['elo'],
            'form_diff': stats_a['form'] - stats_b['form'],
            'xg_diff': stats_a['xg'] - stats_b['xg'],
            'xga_diff': stats_a['xga'] - stats_b['xga'],
            'squad_rating_diff': stats_a['squad_rating'] - stats_b['squad_rating'],
            'h2h_home': h2h_home,
            'h2h_draw': h2h_draw,
            'h2h_away': h2h_away,
            'target': outcome,
            'home_team': team_a,
            'away_team': team_b
        })
        
    df_features = pd.DataFrame(feature_rows)
    print(f"Built dataset with {len(df_features)} rows.")
    return df_features

def train_and_evaluate():
    """Train XGBoost model and save to models/model.pkl."""
    df = build_features()
    
    # Split by date: train on matches before 2025, test on 2025 onward (2025 and 2026)
    train_df = df[df['date'] < '2025-01-01']
    test_df = df[df['date'] >= '2025-01-01']
    
    features = ['elo_diff', 'form_diff', 'xg_diff', 'xga_diff', 'squad_rating_diff', 'h2h_home', 'h2h_draw', 'h2h_away']
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows.")
    
    # Train XGBoost Multiclass Classifier
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=150,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Test Evaluation
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\n--- Model Evaluation ---")
    print(f"Test Accuracy (2022+): {accuracy:.4f}")
    print("Confusion Matrix (Loss/Draw/Win):")
    print(cm)
    
    # Save model and feature names
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'features': features}, f)
    print(f"Model saved to {MODEL_PATH}")

import streamlit as st

@st.cache_resource(show_spinner=False)
def get_cached_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train the model first.")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def get_cached_stats():
    df_stats = pd.read_csv(TEAM_STATS_PATH)
    return df_stats.set_index('team').to_dict(orient='index')

@st.cache_data(show_spinner=False)
def get_cached_results():
    df_results = pd.read_csv(RESULTS_PATH)
    from data_pipeline import normalize_name
    df_results['home_team'] = df_results['home_team'].apply(normalize_name)
    df_results['away_team'] = df_results['away_team'].apply(normalize_name)
    return df_results

def predict_match(team_a, team_b):
    """Predict W/D/L probabilities for a match between team_a and team_b.
    
    Returns a dictionary: {'Win': p_win, 'Draw': p_draw, 'Loss': p_loss}
    from team_a's perspective (Win = team_a wins).
    """
    # Load model
    data = get_cached_model()
    model = data['model']
    features = data['features']
        
    # Load stats
    stats_dict = get_cached_stats()
    
    from data_pipeline import normalize_name
    team_a_norm = normalize_name(team_a)
    team_b_norm = normalize_name(team_b)
    
    if team_a_norm not in stats_dict:
        raise ValueError(f"Team '{team_a}' not found in team stats database.")
    if team_b_norm not in stats_dict:
        raise ValueError(f"Team '{team_b}' not found in team stats database.")
        
    stats_a = stats_dict[team_a_norm]
    stats_b = stats_dict[team_b_norm]
    
    # Read results to compute overall H2H
    df_results = get_cached_results()
    
    # Filter all historical matches between these two
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
        
    # Build feature vector
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
    probs = model.predict_proba(df_vector)[0]
    
    # Class labels: 0=Loss, 1=Draw, 2=Win
    return {
        'Win': float(probs[2]),
        'Draw': float(probs[1]),
        'Loss': float(probs[0])
    }

if __name__ == "__main__":
    train_and_evaluate()
