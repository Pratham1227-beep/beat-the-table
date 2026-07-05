import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

RESULTS_PATH = os.path.join(RAW_DIR, 'results.csv')
PLAYERS_FIFA23_PROCESSED = os.path.join(PROCESSED_DIR, 'players_fifa23.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')

# FIFA Rankings as of June 2018
FIFA_RANKS_2018 = {
    'Germany': 1, 'Brazil': 2, 'Belgium': 3, 'Portugal': 4, 'Argentina': 5,
    'Switzerland': 6, 'France': 7, 'Poland': 8, 'Spain': 10, 'Peru': 11,
    'Denmark': 12, 'England': 12, 'Uruguay': 14, 'Mexico': 15, 'Colombia': 16,
    'Croatia': 20, 'Tunisia': 21, 'Iceland': 22, 'Costa Rica': 23, 'Sweden': 24,
    'Senegal': 27, 'Serbia': 34, 'Australia': 36, 'Iran': 37, 'Morocco': 41,
    'Egypt': 45, 'Nigeria': 48, 'Panama': 55, 'South Korea': 57, 'Japan': 61,
    'Saudi Arabia': 67, 'Russia': 70
}

# FIFA Rankings as of October 2022
FIFA_RANKS_2022 = {
    'Brazil': 1, 'Belgium': 2, 'Argentina': 3, 'France': 4, 'England': 5,
    'Spain': 7, 'Netherlands': 8, 'Portugal': 9, 'Denmark': 10, 'Germany': 11,
    'Croatia': 12, 'Mexico': 13, 'Uruguay': 14, 'Switzerland': 15, 'USA': 16,
    'Senegal': 18, 'Wales': 19, 'Iran': 20, 'Serbia': 21, 'Morocco': 22,
    'Japan': 24, 'Poland': 26, 'South Korea': 28, 'Tunisia': 30, 'Costa Rica': 31,
    'Australia': 38, 'Canada': 41, 'Cameroon': 43, 'Ecuador': 44, 'Qatar': 50,
    'Saudi Arabia': 51, 'Ghana': 61
}

def get_fifa_rank(team, year):
    ranks = FIFA_RANKS_2018 if year == 2018 else FIFA_RANKS_2022
    from data_pipeline import normalize_name
    team_norm = normalize_name(team)
    return ranks.get(team_norm, 80)

def compute_historical_elos_as_of(date, df_results):
    """Compute ELO ratings for all teams chronologically using matches before date."""
    elos = {}
    
    # Filter matches before date
    df_prior = df_results[df_results['date'] < date].sort_values(by='date').reset_index(drop=True)
    
    # Initialize all teams at 1500
    for team in pd.unique(df_results[['home_team', 'away_team']].values.ravel()):
        elos[team] = 1500
        
    K = 32
    for idx, row in df_prior.iterrows():
        th = row['home_team']
        ta = row['away_team']
        hs = row['home_score']
        as_ = row['away_score']
        
        if pd.isna(hs) or pd.isna(as_):
            continue
            
        eh = 1 / (10 ** ((elos[ta] - elos[th]) / 400) + 1)
        ea = 1 - eh
        
        if hs > as_:
            ah, aa = 1.0, 0.0
        elif hs < as_:
            ah, aa = 0.0, 1.0
        else:
            ah, aa = 0.5, 0.5
            
        elos[th] += K * (ah - eh)
        elos[ta] += K * (aa - ea)
        
    return elos

def get_stats_as_of(date, df_results, elos, squad_ratings):
    """Compute stats (form, goals, xG, xGA) for all teams as of a specific date."""
    from data_pipeline import normalize_name
    
    df_prior = df_results[df_results['date'] < date].sort_values(by='date')
    all_teams = pd.unique(df_results[['home_team', 'away_team']].values.ravel())
    
    stats = {}
    for team in all_teams:
        team_matches = df_prior[
            (df_prior['home_team'] == team) | (df_prior['away_team'] == team)
        ].tail(10)
        
        num_matches = len(team_matches)
        goals_for = []
        goals_against = []
        points = []
        
        for idx, row in team_matches.iterrows():
            if row['home_team'] == team:
                gf = row['home_score']
                ga = row['away_score']
            else:
                gf = row['away_score']
                ga = row['home_score']
                
            goals_for.append(gf)
            goals_against.append(ga)
            
            if gf > ga:
                points.append(3)
            elif gf == ga:
                points.append(1)
            else:
                points.append(0)
                
        if num_matches > 0:
            denom = sum(range(1, num_matches + 1))
            weighted_form = sum(p * (i + 1) for i, p in enumerate(points)) / denom
            avg_gf = np.mean(goals_for)
            avg_ga = np.mean(goals_against)
        else:
            weighted_form = 1.0
            avg_gf = 1.0
            avg_ga = 1.0
            
        stats[team] = {
            'team': team,
            'elo': elos.get(team, 1500),
            'form': weighted_form,
            'avg_gf': avg_gf,
            'avg_ga': avg_ga,
            'xg': avg_gf,   # default to goals scored as fallback
            'xga': avg_ga,  # default to goals conceded as fallback
            'squad_rating': squad_ratings.get(team, 65.0)
        }
    return stats

def run_backtest():
    """Run prediction on 2018 and 2022 World Cups using pre-tournament team stats."""
    # 1. Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train the model first.")
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        features = data['features']

    df_results = pd.read_csv(RESULTS_PATH)
    df_results['date'] = pd.to_datetime(df_results['date'])
    
    from data_pipeline import normalize_name
    df_results['home_team'] = df_results['home_team'].apply(normalize_name)
    df_results['away_team'] = df_results['away_team'].apply(normalize_name)
    
    # Load Squad Ratings
    squad_ratings = {}
    if os.path.exists(PLAYERS_FIFA23_PROCESSED):
        import csv
        player_ratings = {}
        with open(PLAYERS_FIFA23_PROCESSED, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            nat_idx = header.index('nationality_name')
            ovr_idx = header.index('overall')
            for row in reader:
                if len(row) > max(nat_idx, ovr_idx):
                    nat = normalize_name(row[nat_idx])
                    try:
                        ovr = int(row[ovr_idx])
                        if nat not in player_ratings:
                            player_ratings[nat] = []
                        player_ratings[nat].append(ovr)
                    except ValueError:
                        pass
        for nation, ratings in player_ratings.items():
            top_ratings = sorted(ratings, reverse=True)[:23]
            squad_ratings[nation] = sum(top_ratings) / len(top_ratings)

    # We evaluate 2018 and 2022 World Cups
    scenarios = [
        {'year': 2018, 'start_date': '2018-06-14'},
        {'year': 2022, 'start_date': '2022-11-20'}
    ]
    
    summary_results = []
    correct_upsets = []
    
    for sc in scenarios:
        year = sc['year']
        start_date = pd.to_datetime(sc['start_date'])
        
        # Calculate historical ELO and stats as of pre-tournament
        print(f"Pre-calculating ELO and stats as of {sc['start_date']}...")
        elos = compute_historical_elos_as_of(start_date, df_results)
        stats = get_stats_as_of(start_date, df_results, elos, squad_ratings)
        
        # Filter World Cup matches for that year
        df_wc = df_results[
            (df_results['tournament'] == 'FIFA World Cup') & 
            (df_results['date'].dt.year == year)
        ].copy()
        
        correct_model = 0
        correct_baseline = 0
        total_valid = 0
        
        # Track prior matches between teams for H2H feature calculation
        # To avoid data leakage, we compute H2H using matches prior to start_date
        h2h_history = {}
        df_prior_matches = df_results[df_results['date'] < start_date]
        for _, m in df_prior_matches.iterrows():
            ta = m['home_team']
            tb = m['away_team']
            winner = None
            hs, as_ = m['home_score'], m['away_score']
            if pd.isna(hs) or pd.isna(as_):
                continue
            if hs > as_:
                winner = ta
            elif hs < as_:
                winner = tb
            key = frozenset([ta, tb])
            if key not in h2h_history:
                h2h_history[key] = []
            h2h_history[key].append((ta, tb, winner))
            
        for idx, row in df_wc.iterrows():
            team_a = row['home_team']
            team_b = row['away_team']
            h_score = row['home_score']
            a_score = row['away_score']
            
            if pd.isna(h_score) or pd.isna(a_score):
                continue
                
            total_valid += 1
            
            # Actual Outcome: Loss=0, Draw=1, Win=2
            if h_score > a_score:
                actual = 2
            elif h_score == a_score:
                actual = 1
            else:
                actual = 0
                
            # Naive Baseline prediction: higher rank team wins
            rank_a = get_fifa_rank(team_a, year)
            rank_b = get_fifa_rank(team_b, year)
            
            if rank_a < rank_b:  # lower rank is better
                baseline_pred = 2
            elif rank_a > rank_b:
                baseline_pred = 0
            else:
                baseline_pred = 1
                
            if baseline_pred == actual:
                correct_baseline += 1
                
            # Model prediction
            stats_a = stats.get(team_a, stats.get(team_b, {'elo': 1500, 'form': 1.0, 'xg': 1.0, 'xga': 1.0, 'squad_rating': 65.0}))
            stats_b = stats.get(team_b, stats.get(team_a, {'elo': 1500, 'form': 1.0, 'xg': 1.0, 'xga': 1.0, 'squad_rating': 65.0}))
            
            # Look up H2H
            key = frozenset([team_a, team_b])
            prior = h2h_history.get(key, [])
            if len(prior) >= 5:
                wins_a = sum(1 for h, a, w in prior if w == team_a)
                wins_b = sum(1 for h, a, w in prior if w == team_b)
                draws = sum(1 for h, a, w in prior if w is None)
                tot = len(prior)
                h2h_home = wins_a / tot
                h2h_draw = draws / tot
                h2h_away = wins_b / tot
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
            probs = model.predict_proba(df_vector)[0]
            model_pred = np.argmax(probs) # 0, 1, 2
            
            if model_pred == actual:
                correct_model += 1
                
            # Upset check: if actual outcome was an upset (higher rank team lost) and model predicted it correctly but baseline missed
            is_upset = False
            if rank_a < rank_b and actual == 0:  # Team A (favored) lost
                is_upset = True
            elif rank_b < rank_a and actual == 2:  # Team B (favored) lost
                is_upset = True
                
            if is_upset and model_pred == actual and baseline_pred != actual:
                correct_upsets.append({
                    'Year': year,
                    'Match': f"{team_a} vs {team_b}",
                    'Winner': team_b if actual == 0 else team_a,
                    'FIFA Ranks': f"{team_a} (#{rank_a}) vs {team_b} (#{rank_b})",
                    'Model Prob': f"{probs[actual]:.1%}"
                })
                
        summary_results.append({
            'Tournament': f"{year} World Cup",
            'Total Matches': total_valid,
            'Model Accuracy': correct_model / total_valid,
            'Baseline Accuracy': correct_baseline / total_valid
        })
        
    df_summary = pd.DataFrame(summary_results)
    df_upsets = pd.DataFrame(correct_upsets)
    
    print("\n================ BACKTEST REPORT ================")
    print("\nSummary Accuracies:")
    print(df_summary.to_string(index=False))
    
    print("\nCorrectly Called Upsets (Baseline Missed):")
    if not df_upsets.empty:
        print(df_upsets.to_string(index=False))
    else:
        print("None")
    print("=================================================")
    
    # Save backtest results to processed data for Streamlit display
    df_summary.to_csv(os.path.join(PROCESSED_DIR, 'backtest_summary.csv'), index=False)
    df_upsets.to_csv(os.path.join(PROCESSED_DIR, 'backtest_upsets.csv'), index=False)

if __name__ == "__main__":
    run_backtest()
