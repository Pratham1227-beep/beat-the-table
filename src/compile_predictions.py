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

def compute_historical_elos_as_of(date, df_results):
    elos = {}
    df_prior = df_results[df_results['date'] < date].sort_values(by='date').reset_index(drop=True)
    
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
            'xg': avg_gf,
            'xga': avg_ga,
            'squad_rating': squad_ratings.get(team, 65.0)
        }
    return stats

def compile_all_predictions():
    # Load model
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

    tournaments = [
        {'year': 2018, 'start_date': '2018-06-14', 'name': '2018 World Cup'},
        {'year': 2022, 'start_date': '2022-11-20', 'name': '2022 World Cup'},
        {'year': 2026, 'start_date': '2026-06-11', 'name': '2026 World Cup'}
    ]
    
    all_predictions = []
    
    for tourn in tournaments:
        year = tourn['year']
        start_date = pd.to_datetime(tourn['start_date'])
        
        print(f"Processing ELO & stats for {tourn['name']}...")
        elos = compute_historical_elos_as_of(start_date, df_results)
        stats = get_stats_as_of(start_date, df_results, elos, squad_ratings)
        
        df_wc = df_results[
            (df_results['tournament'] == 'FIFA World Cup') & 
            (df_results['date'].dt.year == year)
        ].sort_values(by='date').copy()
        
        # H2H history
        h2h_history = {}
        df_prior_matches = df_results[df_results['date'] < start_date]
        for _, m in df_prior_matches.iterrows():
            ta, tb = m['home_team'], m['away_team']
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
            date = row['date'].strftime('%Y-%m-%d')
            
            if pd.isna(h_score) or pd.isna(a_score):
                continue
                
            # Actual Outcome: Loss=0, Draw=1, Win=2
            if h_score > a_score:
                actual = "Win"
            elif h_score == a_score:
                actual = "Draw"
            else:
                actual = "Loss"
                
            stats_a = stats.get(team_a, {'elo': 1500, 'form': 1.0, 'xg': 1.0, 'xga': 1.0, 'squad_rating': 65.0})
            stats_b = stats.get(team_b, {'elo': 1500, 'form': 1.0, 'xg': 1.0, 'xga': 1.0, 'squad_rating': 65.0})
            
            # H2H Look up
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
                h2h_home, h2h_draw, h2h_away = 0.333, 0.333, 0.333
                
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
            probs = model.predict_proba(df_vector)[0] # [Loss, Draw, Win]
            
            p_win = probs[2]
            p_draw = probs[1]
            p_loss = probs[0]
            
            pred_idx = np.argmax(probs)
            if pred_idx == 2:
                predicted = "Win"
            elif pred_idx == 1:
                predicted = "Draw"
            else:
                predicted = "Loss"
                
            correct = "Yes" if predicted == actual else "No"
            
            all_predictions.append({
                'Date': date,
                'Tournament': tourn['name'],
                'Team A': team_a,
                'Team B': team_b,
                'Score': f"{int(h_score)}-{int(a_score)}",
                'Actual Result': actual,
                'Team A Win Prob': f"{p_win:.1%}",
                'Draw Prob': f"{p_draw:.1%}",
                'Team B Win Prob': f"{p_loss:.1%}",
                'Predicted Outcome': predicted,
                'Correct?': correct
            })
            
    df_output = pd.DataFrame(all_predictions)
    output_path = os.path.join(PROCESSED_DIR, 'world_cup_predictions_2018_2026.csv')
    df_output.to_csv(output_path, index=False)
    print(f"\nSuccessfully compiled all predictions and saved to {output_path}")
    
    # Print accuracy per tournament
    for tourn in tournaments:
        t_df = df_output[df_output['Tournament'] == tourn['name']]
        correct_count = len(t_df[t_df['Correct?'] == "Yes"])
        total_count = len(t_df)
        acc = correct_count / total_count if total_count > 0 else 0
        print(f" - {tourn['name']}: {correct_count}/{total_count} correct ({acc:.2%})")

if __name__ == "__main__":
    compile_all_predictions()
