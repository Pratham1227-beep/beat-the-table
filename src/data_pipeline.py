import os
import shutil
import urllib.request
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Kaggle dataset paths in user cache
KAGGLE_CACHE_DIR = os.path.expanduser('~/.cache/kagglehub/datasets')
RESULTS_CACHE = os.path.join(KAGGLE_CACHE_DIR, 'martj42', 'international-football-results-from-1872-to-2017', 'versions', '131')
WORLD_CUP_CACHE = os.path.join(KAGGLE_CACHE_DIR, 'abecklas', 'fifa-world-cup', 'versions', '5')
PLAYERS_CACHE = os.path.join(KAGGLE_CACHE_DIR, 'stefanoleone992', 'fifa-23-complete-player-dataset', 'versions', '1')

# Target paths in data/raw
RESULTS_RAW = os.path.join(RAW_DIR, 'results.csv')
SHOOTOUTS_RAW = os.path.join(RAW_DIR, 'shootouts.csv')
GOALSCORERS_RAW = os.path.join(RAW_DIR, 'goalscorers.csv')
WC_MATCHES_RAW = os.path.join(RAW_DIR, 'WorldCupMatches.csv')
WC_PLAYERS_RAW = os.path.join(RAW_DIR, 'WorldCupPlayers.csv')
WC_RAW = os.path.join(RAW_DIR, 'WorldCups.csv')
PLAYERS_RAW = os.path.join(RAW_DIR, 'male_players.csv')

ELO_WORLD_RAW = os.path.join(RAW_DIR, 'elo_world.tsv')
ELO_TEAMS_RAW = os.path.join(RAW_DIR, 'elo_teams.tsv')

def copy_cache_files():
    """Copy kaggle files from the kagglehub cache to data/raw/."""
    print("--- Copying Cached Datasets ---")
    copies = [
        (os.path.join(RESULTS_CACHE, 'results.csv'), RESULTS_RAW),
        (os.path.join(RESULTS_CACHE, 'shootouts.csv'), SHOOTOUTS_RAW),
        (os.path.join(RESULTS_CACHE, 'goalscorers.csv'), GOALSCORERS_RAW),
        (os.path.join(WORLD_CUP_CACHE, 'WorldCupMatches.csv'), WC_MATCHES_RAW),
        (os.path.join(WORLD_CUP_CACHE, 'WorldCupPlayers.csv'), WC_PLAYERS_RAW),
        (os.path.join(WORLD_CUP_CACHE, 'WorldCups.csv'), WC_RAW),
        (os.path.join(PLAYERS_CACHE, 'male_players.csv'), PLAYERS_RAW),
    ]
    for src, dst in copies:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {os.path.basename(src)} -> {dst}")
        else:
            print(f"Warning: Cache file not found: {src}")

def download_elo_data():
    """Download ELO data from eloratings.net."""
    print("--- Downloading ELO Data ---")
    try:
        urllib.request.urlretrieve("https://www.eloratings.net/World.tsv", ELO_WORLD_RAW)
        print(f"Downloaded ELO rankings to {ELO_WORLD_RAW}")
    except Exception as e:
        print(f"Error downloading ELO World: {e}")
        
    try:
        urllib.request.urlretrieve("https://www.eloratings.net/en.teams.tsv", ELO_TEAMS_RAW)
        print(f"Downloaded ELO teams list to {ELO_TEAMS_RAW}")
    except Exception as e:
        print(f"Error downloading ELO Teams: {e}")

# Team name reconciliation map
NAME_MAP = {
    'United States': 'USA',
    'United States of America': 'USA',
    'US Virgin Islands': 'U.S. Virgin Islands',
    'Democratic Republic of the Congo': 'DR Congo',
    'Congo DR': 'DR Congo',
    'Congo-Kinshasa': 'DR Congo',
    'Congo-Brazzaville': 'Congo',
    'Republic of the Congo': 'Congo',
    'Ivory Coast': "Côte d'Ivoire",
    'South Korea': 'Korea Republic',
    'North Korea': 'Korea DPR',
    'Czech Republic': 'Czechia',
    'Cape Verde': 'Cabo Verde',
    'Cape Verde Islands': 'Cabo Verde',
    'North Macedonia': 'Macedonia',
    'Macedonia': 'North Macedonia',
    'Swaziland': 'Eswatini',
    'East Timor': 'Timor-Leste',
    'St. Lucia': 'Saint Lucia',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Curacao': 'Curaçao',
    'Turks and Caicos Islands': 'Turks & Caicos Islands',
    'Saint Vincent / Grenadines': 'Saint Vincent and the Grenadines',
    'Saint Kitts / Nevis': 'Saint Kitts and Nevis',
    'Antigua & Barbuda': 'Antigua and Barbuda',
    'Trinidad & Tobago': 'Trinidad and Tobago',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    'IR Iran': 'Iran',
    'Islamic Republic of Iran': 'Iran',
    'China PR': 'China',
    'Brunei Darussalam': 'Brunei',
    'Chinese Taipei': 'Taiwan',
    'Macau': 'Macao',
    'Syrian Arab Republic': 'Syria',
}

def normalize_name(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    return NAME_MAP.get(name, name)

def compute_team_stats():
    """Load raw data, clean names, calculate stats and output team_stats.csv."""
    print("--- Processing Team Stats ---")
    
    # 1. Load ELO ratings
    elo_ratings = {}
    if os.path.exists(ELO_WORLD_RAW) and os.path.exists(ELO_TEAMS_RAW):
        # Read teams list parsed line by line
        code_to_name = {}
        with open(ELO_TEAMS_RAW, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    code_to_name[parts[0]] = parts[1]
        
        # Read world ELO
        df_world = pd.read_csv(ELO_WORLD_RAW, sep='\t', header=None)
        # Column 2 has code, Column 3 has current ELO
        for idx, row in df_world.iterrows():
            code = row[2]
            rating = row[3]
            if code in code_to_name:
                name = normalize_name(code_to_name[code])
                elo_ratings[name] = int(rating)
    else:
        print("Warning: ELO files missing. Current ELO will be defaulted.")

    # 2. Load historical match results
    if not os.path.exists(RESULTS_RAW):
        print(f"Error: {RESULTS_RAW} is missing!")
        return
    
    df_results = pd.read_csv(RESULTS_RAW)
    df_results['date'] = pd.to_datetime(df_results['date'])
    df_results['home_team'] = df_results['home_team'].apply(normalize_name)
    df_results['away_team'] = df_results['away_team'].apply(normalize_name)
    print(f"Loaded {len(df_results)} matches from historical results.")

    # Get the set of all active/recent teams (teams that have played since 2010)
    recent_matches = df_results[df_results['date'] >= '2010-01-01']
    all_teams = pd.unique(recent_matches[['home_team', 'away_team']].values.ravel())
    print(f"Identified {len(all_teams)} active teams since 2010.")

    # 3. Load FIFA player ratings for Squad Ratings
    squad_ratings = {}
    if os.path.exists(PLAYERS_RAW):
        print("Loading FIFA players dataset (streaming and extracting FIFA 23)...")
        from collections import defaultdict
        player_ratings = defaultdict(list)
        try:
            import csv
            PLAYERS_FIFA23_PROCESSED = os.path.join(PROCESSED_DIR, 'players_fifa23.csv')
            with open(PLAYERS_RAW, 'r', encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                header = next(reader)
                
                with open(PLAYERS_FIFA23_PROCESSED, 'w', newline='', encoding='utf-8') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(header)
                    
                    nat_idx = header.index('nationality_name')
                    ovr_idx = header.index('overall')
                    ver_idx = header.index('fifa_version')
                    
                    for row in reader:
                        if len(row) > max(nat_idx, ovr_idx, ver_idx):
                            ver_str = row[ver_idx].strip()
                            if ver_str not in ('23', '23.0'):
                                continue
                            writer.writerow(row)
                            nat = normalize_name(row[nat_idx])
                            try:
                                ovr = int(row[ovr_idx])
                                player_ratings[nat].append(ovr)
                            except ValueError:
                                pass
            
            # Compute average of top 23 for each nation
            for nation, ratings in player_ratings.items():
                top_ratings = sorted(ratings, reverse=True)[:23]
                if top_ratings:
                    squad_ratings[nation] = sum(top_ratings) / len(top_ratings)
            print(f"Computed squad ratings for {len(squad_ratings)} nations.")
        except Exception as e:
            print(f"Error reading players dataset: {e}")
    else:
        print("Warning: FIFA players dataset missing. Squad ratings will be defaulted.")

    # 4. Handle FBRef Advanced Stats files if present
    xG_data = {}
    xGA_data = {}
    
    seasons_html_path = os.path.join(RAW_DIR, 'fbref_seasons.html')
    countries_html_path = os.path.join(RAW_DIR, 'fbref_countries.html')
    
    if os.path.exists(countries_html_path):
        print("Parsing advanced stats from local fbref_countries.html...")
        try:
            with open(countries_html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            # Look for stats tables
            tables = soup.find_all('table')
            for table in tables:
                df_html = pd.read_html(str(table))[0]
                # Try to locate columns like 'Squad', 'xG', 'xGA'
                df_html.columns = [col[1] if isinstance(col, tuple) else col for col in df_html.columns]
                squad_col = [c for c in df_html.columns if 'Squad' in str(c) or 'Team' in str(c)]
                xg_col = [c for c in df_html.columns if c == 'xG']
                xga_col = [c for c in df_html.columns if c == 'xGA']
                if squad_col and xg_col:
                    s_col = squad_col[0]
                    x_col = xg_col[0]
                    xa_col = xga_col[0] if xga_col else None
                    for _, r_html in df_html.iterrows():
                        team_name = normalize_name(str(r_html[s_col]))
                        try:
                            xG_val = float(r_html[x_col])
                            xG_data[team_name] = xG_val
                            if xa_col:
                                xGA_data[team_name] = float(r_html[xa_col])
                        except ValueError:
                            pass
            print(f"Extracted xG stats for {len(xG_data)} teams from fbref_countries.html")
        except Exception as e:
            print(f"Error parsing fbref_countries.html: {e}")

    # 5. Compute Stats per Team
    processed_rows = []
    mismatches_elo = []
    mismatches_squad = []
    
    for team in all_teams:
        # ELO
        team_elo = elo_ratings.get(team)
        if team_elo is None:
            mismatches_elo.append(team)
            team_elo = 1500  # Default ELO

        # Squad Rating
        team_squad = squad_ratings.get(team)
        if team_squad is None:
            mismatches_squad.append(team)
            team_squad = 65.0  # Default squad rating

        # Filter last 10 matches
        team_matches = df_results[
            ((df_results['home_team'] == team) | (df_results['away_team'] == team))
        ].sort_values(by='date', ascending=True)  # Chronological order
        
        # Take last 10
        last_10 = team_matches.tail(10)
        num_matches = len(last_10)
        
        goals_for = []
        goals_against = []
        points = []
        
        for idx, row in last_10.iterrows():
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
                
        # Compute Weighted Form
        # denominator is sum(1..K)
        if num_matches > 0:
            denom = sum(range(1, num_matches + 1))
            weighted_form_score = sum(p * (i + 1) for i, p in enumerate(points)) / denom
            avg_gf = np.mean(goals_for)
            avg_ga = np.mean(goals_against)
        else:
            weighted_form_score = 1.0  # neutral default form (similar to 1 point per match)
            avg_gf = 1.0
            avg_ga = 1.0

        # xG / xGA fallbacks or scraped
        team_xg = xG_data.get(team, avg_gf) # Fallback to average goals scored
        team_xga = xGA_data.get(team, avg_ga) # Fallback to average goals conceded
        
        processed_rows.append({
            'team': team,
            'elo': team_elo,
            'form': weighted_form_score,
            'avg_gf': avg_gf,
            'avg_ga': avg_ga,
            'xg': team_xg,
            'xga': team_xga,
            'squad_rating': team_squad
        })
        
    df_processed = pd.DataFrame(processed_rows)
    df_processed.to_csv(os.path.join(PROCESSED_DIR, 'team_stats.csv'), index=False)
    
    print(f"Generated team_stats.csv with {len(df_processed)} rows.")
    print(f"ELO name mismatches: {len(mismatches_elo)} / {len(all_teams)} (defaulted to 1500)")
    print(f"Squad name mismatches: {len(mismatches_squad)} / {len(all_teams)} (defaulted to 65.0)")

if __name__ == "__main__":
    copy_cache_files()
    download_elo_data()
    compute_team_stats()
