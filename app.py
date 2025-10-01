import numpy as np
import pandas as pd
import streamlit as st
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="Goalflux", page_icon="⚽", layout="wide")

# ESPN API for football standings
ESPN_BASE = "https://site.api.espn.com/apis/v2/sports"

# Available leagues with their ESPN codes
AVAILABLE_LEAGUES = {
    "English Premier League": "eng.1",
    "Spanish La Liga": "esp.1", 
    "Italian Serie A": "ita.1",
    "German Bundesliga": "ger.1",
    "French Ligue 1": "fra.1",
    "Dutch Eredivisie": "ned.1",
    "Portuguese Primeira Liga": "por.1",
    "Belgian Pro League": "bel.1",
    "Turkish Süper Lig": "tur.1",
    "Russian Premier League": "rus.1",
    "Scottish Premiership": "sco.1",
    "Austrian Bundesliga": "aut.1",
    "Swiss Super League": "sui.1"
}

# League strength factors for UEFA Champions League adjustments
LEAGUE_STRENGTH = {
    "English Premier League": 1.10,
    "Spanish La Liga": 1.08,
    "Italian Serie A": 1.05,
    "German Bundesliga": 1.06,
    "French Ligue 1": 1.02,
    "Portuguese Primeira Liga": 0.99,
    "Dutch Eredivisie": 0.98,
    "Belgian Pro League": 0.92,
    "Turkish Süper Lig": 0.91,
    "Scottish Premiership": 0.90,
    "Austrian Bundesliga": 0.92,
    "Swiss Super League": 0.90,
}

def _choose(v, *keys, default=None):
    """Safely pick the first existing key in dict v['stats'] list or v directly."""
    if isinstance(v, dict) and "stats" in v:
        bag = {s.get("name") or s.get("shortDisplayName",""): s for s in v["stats"]}
        for k in keys:
            if k in bag and "value" in bag[k]:
                return bag[k]["value"]
    if isinstance(v, dict):
        for k in keys:
            if k in v:
                return v[k]
    return default

def fetch_league_standings(league_name, season=None):
    """Fetch standings for a specific league using ESPN API."""
    try:
        if season is None:
            season = datetime.now(timezone.utc).year
            
        league_code = AVAILABLE_LEAGUES.get(league_name)
        if not league_code:
            raise ValueError(f"League '{league_name}' not supported")
            
        url = f"{ESPN_BASE}/soccer/{league_code}/standings?season={season}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        # The payload has groups -> standings -> entries (teams) with stats.
        # We'll flatten every group (some leagues have multiple groups/phases).
        rows = []
        groups = data.get("children") or data.get("standings", {}).get("groups") or [data]
        for g in groups:
            standings = g.get("standings") or g  # be tolerant to shapes
            entries = standings.get("entries") if isinstance(standings, dict) else g.get("entries")
            if not entries:
                continue
            for e in entries:
                team = e.get("team", {})
                name = team.get("displayName") or team.get("name")
                stats = e  # stats typically live in the entry
                mp = _choose(stats, "gamesPlayed", "played", "GP", default=None)
                gf = _choose(stats, "goalsFor", "pointsFor", "GF", default=None)
                ga = _choose(stats, "goalsAgainst", "pointsAgainst", "GA", default=None)
                # Extract rank from stats array - look for the rank stat specifically
                rank = None
                if e.get("stats"):
                    for stat in e["stats"]:
                        if stat.get("name") == "rank":
                            rank = stat.get("value")
                            break
                
                # Fallback to direct rank field
                if rank is None:
                    rank = e.get("rank")
                
                # If still no rank, use position in list + 1
                if rank is None:
                    rank = len(rows) + 1
                if all(v is not None for v in (name, mp, gf, ga)):
                    rows.append({"Club": name, "MP": int(mp), "GF": int(gf), "GA": int(ga)})
        
        # De-duplicate by Club
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("Could not parse ESPN standings; check league/season or payload shape.")
        df = df.groupby("Club", as_index=False).first()
        
        # Sort by goal difference (S = GF - GA) and reset index for clean numbering
        df = df.sort_values(["GF", "GA"], ascending=[False, True]).reset_index(drop=True)
        df.index = df.index + 1  # Start numbering from 1 instead of 0
        
        return df, league_name
        
    except Exception as e:
        st.error(f"Error fetching standings for {league_name}: {e}")
        return None, None

# ---------- helpers ----------
TIER_RULES = [
    ("Elite", 1.5,  9),       # S >= 1.5
    ("Strong", 0.6, 1.5),     # 0.6 <= S < 1.5
    ("Balanced", -0.5, 0.6),  # -0.5 <= S < 0.6
    ("Weak", -1.5, -0.5),     # -1.5 <= S < -0.5
    ("Relegation", -9, -1.5)  # S < -1.5
]

def tier_from_s(s):
    for name, lo, hi in TIER_RULES:
        if lo <= s < hi:
            return name
    return "Balanced"

def style_tag(o, d):
    # thresholds are simple; tweak as you like
    if o >= 1.8 and d >= 1.6:
        return "Attack-heavy, leaky"
    if o >= 1.8 and d <= 1.0:
        return "Balanced high-scoring"
    if o <= 1.0 and d <= 1.0:
        return "Low-event (both low)"
    if d <= 0.9 and o < 1.5:
        return "Defense-first"
    if o >= 1.6 and d > 1.2:
        return "Chaotic (goals both ways)"
    return "Mixed"

def _exp_goals(att_o, opp_d, league_mean_o, league_mean_d, home_boost=0.15):
    """
    att_o: team goals scored per match (O)
    opp_d: opponent goals conceded per match (D)
    league_mean_o, league_mean_d: means from your standings df
    home_boost: small bump for the home side
    """
    # How good is the attack vs average?
    atk_rel = (att_o / max(0.01, league_mean_o))

    # How leaky is the opponent vs average? (>1 = worse than avg, <1 = better)
    def_rel = (opp_d / max(0.01, league_mean_d))

    # Map to a gentle multiplier. Poor defenses (>1) increase xG, elite defenses (<1) decrease it.
    # Clamp so it doesn't go wild.
    mult = 1.0 + np.clip(0.55*(def_rel - 1.0), -0.35, 0.60)

    # Base xG comes from the attack quality scaled by league scoring level
    base = atk_rel * league_mean_o

    return max(0.2, base * mult + home_boost)

def adjust_to_ucl(O, D, muO_league, muD_league, league_name, muO_ucl, muD_ucl, alpha=1.0, beta=1.0):
    """
    Adjust team performance from domestic league to UEFA Champions League level.
    
    Args:
        O, D: Team's offensive and defensive performance in domestic league
        muO_league, muD_league: League means for offensive and defensive performance
        league_name: Name of the domestic league
        muO_ucl, muD_ucl: UEFA Champions League means for offensive and defensive performance
        alpha, beta: Adjustment parameters for attack and defense
    
    Returns:
        O_ucl, D_ucl: Adjusted offensive and defensive performance for UCL level
    """
    lsf = LEAGUE_STRENGTH.get(league_name, 1.00)
    atk_rel = O / max(1e-6, muO_league)
    def_rel = D / max(1e-6, muD_league)

    atk_ucl = atk_rel / (lsf ** alpha)
    def_ucl = def_rel * (lsf ** beta)

    O_ucl = atk_ucl * muO_ucl
    D_ucl = def_ucl * muD_ucl
    return O_ucl, D_ucl

def exp_goals_ucl(teamA, teamB, dfA, dfB, leagueA, leagueB, muO_ucl, muD_ucl, lambda_shrink=0.6, home_boost=0.12):
    """
    Calculate expected goals for UEFA Champions League match between teams from different leagues.
    
    Args:
        teamA, teamB: Team names
        dfA, dfB: DataFrames containing team performance data with O, D, muO, muD columns
        leagueA, leagueB: League names for each team
        muO_ucl, muD_ucl: UEFA Champions League means
        lambda_shrink: Shrinkage factor toward global mean
        home_boost: Home advantage boost
    
    Returns:
        xh, xa: Expected goals for home and away teams
    """
    # Adjust team performance to UCL level
    Oa_ucl, Da_ucl = adjust_to_ucl(dfA.O, dfA.D, dfA.muO, dfA.muD, leagueA, muO_ucl, muD_ucl)
    Ob_ucl, Db_ucl = adjust_to_ucl(dfB.O, dfB.D, dfB.muO, dfB.muD, leagueB, muO_ucl, muD_ucl)

    # Calculate expected goals
    xh = _exp_goals(Oa_ucl, Db_ucl, muO_ucl, muD_ucl, home_boost=home_boost)
    xa = _exp_goals(Ob_ucl, Da_ucl, muO_ucl, muD_ucl, home_boost=0.0)

    # Shrink toward global mean for stability
    xh = lambda_shrink*xh + (1-lambda_shrink)*muO_ucl
    xa = lambda_shrink*xa + (1-lambda_shrink)*muO_ucl
    return xh, xa

import math

def _poisson_pmf(lmbda, k):
    # stable Poisson PMF without scipy
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)

def _score_probs(exp_home, exp_away, cap=6):
    # returns matrix P[i][j] and handy aggregates
    P = []
    home_marg = []
    away_marg = []
    for i in range(cap+1):
        p_i = _poisson_pmf(exp_home, i)
        row = []
        for j in range(cap+1):
            p = p_i * _poisson_pmf(exp_away, j)
            row.append(p)
        P.append(row)
    # aggregates
    ph_win = sum(P[i][j] for i in range(cap+1) for j in range(cap+1) if i > j)
    p_draw = sum(P[i][i] for i in range(cap+1))
    pa_win = 1.0 - ph_win - p_draw
    p_over25 = sum(P[i][j] for i in range(cap+1) for j in range(cap+1) if i + j >= 3)
    p_btts   = sum(P[i][j] for i in range(1, cap+1) for j in range(1, cap+1))
    # top scores
    flat = [((i, j), P[i][j]) for i in range(cap+1) for j in range(cap+1)]
    flat.sort(key=lambda x: x[1], reverse=True)
    top = []
    for (i, j), pr in flat:
        s = f"{i}-{j}"
        if s not in top:
            top.append(s)
        if len(top) == 3:
            break
    return {
        "P": P,
        "ph_win": ph_win,
        "p_draw": p_draw,
        "pa_win": pa_win,
        "p_over25": p_over25,
        "p_btts": p_btts,
        "top_scores": top
    }

def predict_row(home, away, df, draw_bias=0.1):
    a = df.loc[df["Club"].str.casefold()==home.casefold()].iloc[0]
    b = df.loc[df["Club"].str.casefold()==away.casefold()].iloc[0]

    # strengths / styles for display
    s_diff = (a["S"] - b["S"])
    o_vs_d_edge = (a["O"] - b["D"]) - (b["O"] - a["D"])

    # xG (the fixed version from my last message)
    mean_o = df["O"].mean(); mean_d = df["D"].mean()
    exp_home = _exp_goals(a["O"], b["D"], mean_o, mean_d, home_boost=0.20)
    exp_away = _exp_goals(b["O"], a["D"], mean_o, mean_d, home_boost=0.00)
    gsum = exp_home + exp_away

    # --- single source of truth: Poisson grid ---
    agg = _score_probs(exp_home, exp_away, cap=6)

    # 1X2 from probabilities (adds % to be transparent)
    trio = [("Draw", agg["p_draw"]), (a["Club"]+" win", agg["ph_win"]), (b["Club"]+" win", agg["pa_win"])]
    trio.sort(key=lambda x: x[1], reverse=True)
    outcome, p_outcome = trio[0]
    # add "Lean …" if margin is small
    margin = p_outcome - trio[1][1]
    if margin < 0.05:
        outcome = "Draw" if outcome == "Draw" else f"Lean {outcome}"

    # Goals market from p(Over 2.5)
    p_over = agg["p_over25"]
    if p_over >= 0.58:
        goals = "Over 2.5"
    elif p_over <= 0.42:
        goals = "Under 2.5"
    else:
        goals = "Lean Over 2.5"

    # BTTS from p(BTTS)
    p_btts = agg["p_btts"]
    if p_btts >= 0.58:
        btts = "Yes"
    elif p_btts <= 0.42:
        btts = "No"
    else:
        btts = "Lean Yes" if p_btts > 0.5 else "Lean No"

    # Scorelines (top 3 from the same grid)
    top = agg["top_scores"]
    score_hint = f"Most likely: {top[0]}" + (f" | Also: {', '.join(top[1:])}" if len(top) > 1 else "")

    return {
        "Fixture": f"{a['Club']} vs {b['Club']}",
        "Home tier": a["Tier"],
        "Away tier": b["Tier"],
        "Style (Home)": a["Style"],
        "Style (Away)": b["Style"],
        "1X2 lean": outcome,
        "Goals": goals,
        "BTTS": btts,
        "Score range": score_hint,
        "Edges": (
            f"SΔ={s_diff:+.2f}, O_vs_DΔ={o_vs_d_edge:+.2f}, "
            f"exp_home={exp_home:.2f}, exp_away={exp_away:.2f}, "
            f"P(H)= {agg['ph_win']:.2%}, P(D)= {agg['p_draw']:.2%}, P(A)= {agg['pa_win']:.2%}, "
            f"P(Over2.5)= {p_over:.2%}, P(BTTS)= {p_btts:.2%}"
        ),
    }

def predict_row_ucl(home_team, away_team, df_home, df_away, league_home, league_away, draw_bias=0.1):
    """
    Predict UEFA Champions League match between teams from different leagues.
    
    Args:
        home_team, away_team: Team names
        df_home, df_away: DataFrames containing team performance data
        league_home, league_away: League names for each team
        draw_bias: Bias toward draw predictions
    
    Returns:
        Dictionary with prediction results
    """
    # Get team data
    a = df_home.loc[df_home["Club"].str.casefold() == home_team.casefold()].iloc[0]
    b = df_away.loc[df_away["Club"].str.casefold() == away_team.casefold()].iloc[0]

    # UEFA Champions League means (typical values for elite competition)
    muO_ucl = 1.4  # Expected goals per match in UCL
    muD_ucl = 1.4  # Expected goals conceded per match in UCL
    
    # Calculate expected goals using UCL adjustment
    exp_home, exp_away = exp_goals_ucl(home_team, away_team, a, b, league_home, league_away, muO_ucl, muD_ucl)
    
    # Calculate score probabilities
    agg = _score_probs(exp_home, exp_away, cap=6)
    
    # Adjust for draw bias
    agg["p_draw"] = min(0.35, agg["p_draw"] + draw_bias)
    agg["ph_win"] = agg["ph_win"] * (1 - draw_bias/2)
    agg["pa_win"] = agg["pa_win"] * (1 - draw_bias/2)
    
    # Normalize probabilities
    total = agg["ph_win"] + agg["p_draw"] + agg["pa_win"]
    agg["ph_win"] /= total
    agg["p_draw"] /= total
    agg["pa_win"] /= total

    # Determine outcomes
    trio = [("Draw", agg["p_draw"]), (f"{a['Club']} win", agg["ph_win"]), (f"{b['Club']} win", agg["pa_win"])]
    trio.sort(key=lambda x: x[1], reverse=True)
    outcome, p_outcome = trio[0]
    
    margin = p_outcome - trio[1][1]
    if margin < 0.05:
        outcome = "Draw" if outcome == "Draw" else f"Lean {outcome}"

    # Goals market
    p_over = agg["p_over25"]
    if p_over >= 0.58:
        goals = "Over 2.5"
    elif p_over <= 0.42:
        goals = "Under 2.5"
    else:
        goals = "Lean Over 2.5"

    # BTTS
    p_btts = agg["p_btts"]
    if p_btts >= 0.58:
        btts = "Yes"
    elif p_btts <= 0.42:
        btts = "No"
    else:
        btts = "Lean Yes" if p_btts > 0.5 else "Lean No"

    # Scorelines
    top = agg["top_scores"]
    score_hint = f"Most likely: {top[0]}" + (f" | Also: {', '.join(top[1:])}" if len(top) > 1 else "")
    
    # League strength info
    lsf_home = LEAGUE_STRENGTH.get(league_home, 1.00)
    lsf_away = LEAGUE_STRENGTH.get(league_away, 1.00)
    league_strength_info = f"League strength: {league_home} ({lsf_home:.2f}) vs {league_away} ({lsf_away:.2f})"

    return {
        "Fixture": f"{a['Club']} vs {b['Club']}",
        "League (Home)": league_home,
        "League (Away)": league_away,
        "Home tier": a["Tier"],
        "Away tier": b["Tier"],
        "Style (Home)": a["Style"],
        "Style (Away)": b["Style"],
        "1X2 lean": outcome,
        "Goals": goals,
        "BTTS": btts,
        "Score range": score_hint,
        "UCL xG": f"Home: {exp_home:.2f}, Away: {exp_away:.2f}",
        "Probabilities": f"P(H)= {agg['ph_win']:.2%}, P(D)= {agg['p_draw']:.2%}, P(A)= {agg['pa_win']:.2%}",
        "League Strength": league_strength_info,
        "Edges": (
            f"P(Over2.5)= {p_over:.2%}, P(BTTS)= {p_btts:.2%}"
        ),
    }

def compute_table(raw):
    req_cols = {"Club","MP","GF","GA"}
    if not req_cols.issubset(set(raw.columns)):
        raise ValueError(f"CSV needs columns: {', '.join(sorted(req_cols))}")
    df = raw.copy()
    df["O"] = (df["GF"] / df["MP"]).round(3)
    df["D"] = (df["GA"] / df["MP"]).round(3)
    df["S"] = (df["O"] - df["D"]).round(3)

    # Note: Removed Off Rank and Def Rank columns for cleaner display
    df["Tier"] = df["S"].apply(tier_from_s)
    df["Style"] = df.apply(lambda r: style_tag(r["O"], r["D"]), axis=1)
    # optional: League Rank if provided; otherwise derive from Pts or leave blank
    return df

# ---------- UI ----------
st.title("Goalflux")
st.markdown("**Intelligent Football Matchup Predictor**")

with st.expander("1) Fetch League Standings", expanded=True):
    st.markdown("**Choose a league to automatically fetch current standings:**")
    
    # Create league selection from predefined list
    selected_league = st.selectbox("Select League:", list(AVAILABLE_LEAGUES.keys()))
    
    # Season selection
    current_year = datetime.now(timezone.utc).year
    season = st.number_input("Season:", min_value=2020, max_value=current_year+1, value=current_year)
    
    # Fetch standings button
    if st.button("Fetch Standings", type="primary"):
        with st.spinner(f"Fetching {selected_league} standings..."):
            df, league_name = fetch_league_standings(selected_league, season)
            
        if df is not None:
            df = compute_table(df)
            # Store in session state to persist across reruns
            st.session_state.standings_df = df
            st.session_state.league_name = league_name
            st.success(f"{league_name} standings loaded successfully!")
        else:
            st.error("Failed to fetch standings. Please try again.")
    
    # Display standings if loaded (only once)
    if 'standings_df' in st.session_state:
        st.info("**Teams are ordered by goals scored, then goals against**")
        st.dataframe(st.session_state.standings_df.sort_values("S", ascending=False), width='stretch')
    

with st.expander("2) Enter fixtures", expanded=True):
    st.markdown("**Choose teams to create fixtures:**")
    
    # Check if we have standings data in session state
    if 'standings_df' in st.session_state and st.session_state.standings_df is not None:
        # Get team names from the standings data
        team_names = sorted(st.session_state.standings_df['Club'].tolist())
        
        # Create two columns for team selection
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            home_team = st.selectbox("Home Team:", [""] + team_names, key="home_team")
        
        with col2:
            away_team = st.selectbox("Away Team:", [""] + team_names, key="away_team")
        
        with col3:
            if st.button("Add Fixture", type="primary"):
                if home_team and away_team and home_team != away_team:
                    # Add to session state
                    if 'fixtures_list' not in st.session_state:
                        st.session_state.fixtures_list = []
                    st.session_state.fixtures_list.append(f"{home_team} vs {away_team}")
                    st.success(f"Added: {home_team} vs {away_team}")
                elif home_team == away_team:
                    st.error("Home and away teams cannot be the same!")
                else:
                    st.error("Please select both teams!")
        
        # Display current fixtures
        if 'fixtures_list' in st.session_state and st.session_state.fixtures_list:
            st.markdown("**Current Fixtures:**")
            for i, fixture in enumerate(st.session_state.fixtures_list):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {fixture}")
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.fixtures_list.pop(i)
                        st.rerun()
            
            # Clear all button
            if st.button("Clear All Fixtures"):
                st.session_state.fixtures_list = []
                st.rerun()
        
        # Convert to text for processing
        if 'fixtures_list' in st.session_state and st.session_state.fixtures_list:
            fixtures_text = "\n".join(st.session_state.fixtures_list)
        else:
            fixtures_text = ""
    else:
        st.warning("⚠️ Please fetch standings data first to use team selectors.")
        fixtures_text = ""

fixtures = []
for line in fixtures_text.splitlines():
    if " vs " in line:
        h, a = [x.strip() for x in line.split(" vs ", 1)]
        if h and a:
            fixtures.append((h, a))

# UEFA Champions League Section
with st.expander("🏆 UEFA Champions League Cross-League Predictions", expanded=False):
    st.markdown("**Compare teams from different leagues using league strength adjustments**")
    
    # Check if we have any standings data
    available_leagues = []
    if 'standings_df' in st.session_state:
        available_leagues.append((st.session_state.league_name, st.session_state.standings_df))
    
    # Add any previously loaded UCL leagues
    for key, value in st.session_state.items():
        if key.startswith('ucl_league_'):
            league_name = key.replace('ucl_league_', '')
            if league_name not in [league[0] for league in available_leagues]:
                available_leagues.append((league_name, value))
    
    # Allow users to load multiple leagues for UCL comparisons
    st.markdown("**Load additional leagues for cross-league comparisons:**")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        additional_league = st.selectbox("Load additional league:", [""] + [name for name in AVAILABLE_LEAGUES.keys() 
                                                                          if name not in [league[0] for league in available_leagues]])
    
    with col2:
        if st.button("Load League", type="secondary"):
            if additional_league:
                with st.spinner(f"Loading {additional_league}..."):
                    df, league_name = fetch_league_standings(additional_league)
                    if df is not None:
                        df = compute_table(df)
                        available_leagues.append((league_name, df))
                        st.session_state[f'ucl_league_{league_name}'] = df
                        st.success(f"{league_name} loaded for UCL comparisons!")
                        st.rerun()
                    else:
                        st.error("Failed to load league data.")
    
    # Display loaded leagues
    if len(available_leagues) >= 2:
        st.markdown("**Available leagues for UCL predictions:**")
        league_names = [league[0] for league in available_leagues]
        st.info(f"Loaded: {', '.join(league_names)}")
        
        # UCL Fixture Creation
        st.markdown("**Create UEFA Champions League Fixtures:**")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            home_league = st.selectbox("Home Team League:", league_names, key="ucl_home_league")
        
        with col2:
            if home_league:
                home_league_df = next(league[1] for league in available_leagues if league[0] == home_league)
                home_teams = sorted(home_league_df['Club'].tolist())
                home_team = st.selectbox("Home Team:", [""] + home_teams, key="ucl_home_team")
        
        with col3:
            away_league = st.selectbox("Away Team League:", [name for name in league_names if name != home_league], key="ucl_away_league")
        
        with col4:
            if away_league:
                away_league_df = next(league[1] for league in available_leagues if league[0] == away_league)
                away_teams = sorted(away_league_df['Club'].tolist())
                away_team = st.selectbox("Away Team:", [""] + away_teams, key="ucl_away_team")
        
        if st.button("Add UCL Fixture", type="primary"):
            if home_team and away_team and home_league and away_league:
                if 'ucl_fixtures_list' not in st.session_state:
                    st.session_state.ucl_fixtures_list = []
                fixture_info = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_league': home_league,
                    'away_league': away_league
                }
                st.session_state.ucl_fixtures_list.append(fixture_info)
                st.success(f"Added UCL fixture: {home_team} ({home_league}) vs {away_team} ({away_league})")
            else:
                st.error("Please select both teams and leagues!")
        
        # Display UCL fixtures
        if 'ucl_fixtures_list' in st.session_state and st.session_state.ucl_fixtures_list:
            st.markdown("**UCL Fixtures:**")
            for i, fixture in enumerate(st.session_state.ucl_fixtures_list):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {fixture['home_team']} ({fixture['home_league']}) vs {fixture['away_team']} ({fixture['away_league']})")
                with col2:
                    if st.button("Remove", key=f"ucl_remove_{i}"):
                        st.session_state.ucl_fixtures_list.pop(i)
                        st.rerun()
            
            # Clear all UCL fixtures
            if st.button("Clear All UCL Fixtures"):
                st.session_state.ucl_fixtures_list = []
                st.rerun()
    
    elif len(available_leagues) == 1:
        st.info("Load at least 2 different leagues to create UEFA Champions League predictions.")
    
    else:
        st.warning("Please load at least 2 different leagues to use UEFA Champions League predictions.")

st.markdown("---")
st.subheader("Predictions")

# Check if we have standings data
if 'standings_df' not in st.session_state:
    st.warning("⚠️ Please fetch standings data first before making predictions.")
    st.stop()

rows = []
missing = []
for h, a in fixtures:
    try:
        rows.append(predict_row(h, a, st.session_state.standings_df))
    except Exception:
        missing.append(f"{h} vs {a}")

pred_df = pd.DataFrame(rows)
if not pred_df.empty:
    st.dataframe(pred_df, width='stretch')
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions (CSV)", csv, "predictions.csv", "text/csv")
else:
    st.info("No predictions yet — check your fixtures or team names.")

if missing:
    st.warning("Not found teams: " + ", ".join(missing))

# UEFA Champions League Predictions Section
if 'ucl_fixtures_list' in st.session_state and st.session_state.ucl_fixtures_list:
    st.markdown("---")
    st.subheader("🏆 UEFA Champions League Predictions")
    
    ucl_rows = []
    ucl_missing = []
    
    # Get all loaded leagues
    all_leagues = {}
    if 'standings_df' in st.session_state:
        all_leagues[st.session_state.league_name] = st.session_state.standings_df
    
    # Add any additional UCL leagues
    for key, value in st.session_state.items():
        if key.startswith('ucl_league_'):
            league_name = key.replace('ucl_league_', '')
            all_leagues[league_name] = value
    
    for fixture in st.session_state.ucl_fixtures_list:
        try:
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            home_league = fixture['home_league']
            away_league = fixture['away_league']
            
            df_home = all_leagues[home_league]
            df_away = all_leagues[away_league]
            
            # Check if teams exist in their respective leagues
            home_teams = df_home['Club'].str.casefold().tolist()
            away_teams = df_away['Club'].str.casefold().tolist()
            
            if home_team.casefold() not in home_teams:
                raise ValueError(f"Team '{home_team}' not found in {home_league}. Available teams: {', '.join(df_home['Club'].tolist()[:5])}...")
            
            if away_team.casefold() not in away_teams:
                raise ValueError(f"Team '{away_team}' not found in {away_league}. Available teams: {', '.join(df_away['Club'].tolist()[:5])}...")
            
            prediction = predict_row_ucl(home_team, away_team, df_home, df_away, home_league, away_league)
            ucl_rows.append(prediction)
            
        except Exception as e:
            error_msg = f"{fixture['home_team']} ({fixture['home_league']}) vs {fixture['away_team']} ({fixture['away_league']}) - Error: {str(e)}"
            ucl_missing.append(error_msg)
    
    if ucl_rows:
        ucl_pred_df = pd.DataFrame(ucl_rows)
        st.dataframe(ucl_pred_df, width='stretch')
        
        # Download button for UCL predictions
        ucl_csv = ucl_pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download UCL predictions (CSV)", ucl_csv, "ucl_predictions.csv", "text/csv")
        
        # League strength comparison
        st.markdown("**League Strength Factors:**")
        strength_df = pd.DataFrame([
            {"League": league, "Strength Factor": factor}
            for league, factor in LEAGUE_STRENGTH.items()
        ])
        strength_df = strength_df.sort_values("Strength Factor", ascending=False)
        st.dataframe(strength_df, width='stretch')
        
    if ucl_missing:
        st.warning("UCL fixtures with issues:")
        for missing in ucl_missing:
            st.error(missing)
        
        # Show available teams for debugging
        st.markdown("**Available teams by league:**")
        for league_name, df in all_leagues.items():
            with st.expander(f"{league_name} ({len(df)} teams)"):
                teams_list = df['Club'].tolist()
                # Show in columns for better readability
                cols = st.columns(3)
                for i, team in enumerate(teams_list):
                    with cols[i % 3]:
                        st.write(f"• {team}")