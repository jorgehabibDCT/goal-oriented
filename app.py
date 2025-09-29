import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Matchup Predictor", page_icon="⚽", layout="wide")

# Football Standings API - https://github.com/azharimm/football-standings-api
STANDINGS_API_BASE = "https://api-football-standings.azharimm.site"

def fetch_available_leagues():
    """Fetch all available leagues from the standings API."""
    try:
        url = f"{STANDINGS_API_BASE}/leagues"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if data.get("status") and "data" in data:
            return data["data"]
        return []
    except Exception as e:
        st.error(f"Error fetching leagues: {e}")
        return []

def fetch_league_standings(league_id, season=None):
    """Fetch standings for a specific league."""
    try:
        if season is None:
            season = datetime.utcnow().year
            
        url = f"{STANDINGS_API_BASE}/leagues/{league_id}/standings"
        params = {"season": season, "sort": "asc"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if not data.get("status") or "data" not in data:
            raise ValueError("Invalid API response")
            
        league_data = data["data"]
        standings = league_data.get("standings", [])
        
        if not standings:
            raise ValueError("No standings data available")
            
        # Convert to our expected format
        rows = []
        for team in standings:
            name = team.get("name", "")
            stats = team.get("stats", {})
            
            # Extract stats with fallbacks
            mp = stats.get("gamesPlayed", stats.get("played", 0))
            gf = stats.get("goalsFor", stats.get("pointsFor", 0))
            ga = stats.get("goalsAgainst", stats.get("pointsAgainst", 0))
            rank = stats.get("rank", team.get("rank", 0))
            
            if name and mp is not None and gf is not None and ga is not None:
                rows.append({
                    "Club": name,
                    "MP": int(mp),
                    "GF": int(gf), 
                    "GA": int(ga),
                    "Rank": int(rank) if rank else 0
                })
        
        if not rows:
            raise ValueError("No valid team data found")
            
        df = pd.DataFrame(rows)
        return df, league_data.get("name", "Unknown League")
        
    except Exception as e:
        st.error(f"Error fetching standings for {league_id}: {e}")
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
        return "⚡ Attack-heavy, leaky"
    if o >= 1.8 and d <= 1.0:
        return "🔥 Balanced high-scoring"
    if o <= 1.0 and d <= 1.0:
        return "❄️ Low-event (both low)"
    if d <= 0.9 and o < 1.5:
        return "🛡️ Defense-first"
    if o >= 1.6 and d > 1.2:
        return "⚔️ Chaotic (goals both ways)"
    return "Mixed"

def predict_row(home, away, df, draw_bias=0.1):
    a = df.loc[df["Club"].str.lower()==home.lower()].iloc[0]
    b = df.loc[df["Club"].str.lower()==away.lower()].iloc[0]

    # strengths
    s_diff = (a["S"] - b["S"])
    o_vs_d_edge = (a["O"] - b["D"]) - (b["O"] - a["D"])  # how each attack matches the other's defense

    # 1X2 lean
    if s_diff > 0.25 + draw_bias:
        outcome = f"{a['Club']} win"
    elif s_diff < -0.25 - draw_bias:
        outcome = f"{b['Club']} win"
    else:
        outcome = "Draw"

    # goals market
    # expected goals proxy using simple matchup mean
    exp_home = max(0.1, (a["O"] + max(0.2, 2.0 - b["D"])) / 2.8)
    exp_away = max(0.1, (b["O"] + max(0.2, 2.0 - a["D"])) / 2.8)
    gsum = exp_home + exp_away

    if gsum >= 2.6:
        goals = "Over 2.5"
    elif gsum <= 2.2:
        goals = "Under 2.5"
    else:
        goals = "Lean Over 2.5"

    # BTTS heuristic
    btts = "Yes" if (a["O"] > 1.1 and b["O"] > 0.9) or (a["D"] > 1.3 or b["D"] > 1.3) else "No/Lean No"

    # score ranges (very rough)
    def rng(x):
        if x < 0.6: return [0,1]
        if x < 1.2: return [1,2]
        if x < 1.8: return [1,3]
        return [2,4]

    hr = rng(exp_home)
    ar = rng(exp_away)
    score_hint = f"{hr[0]}–{ar[0]} .. {hr[1]}–{ar[1]}"

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
        "Edges": f"SΔ={s_diff:+.2f}, O_vs_DΔ={o_vs_d_edge:+.2f}"
    }

def compute_table(raw):
    req_cols = {"Club","MP","GF","GA"}
    if not req_cols.issubset(set(raw.columns)):
        raise ValueError(f"CSV needs columns: {', '.join(sorted(req_cols))}")
    df = raw.copy()
    df["O"] = (df["GF"] / df["MP"]).round(3)
    df["D"] = (df["GA"] / df["MP"]).round(3)
    df["S"] = (df["O"] - df["D"]).round(3)

    # ranks (lower is better)
    df["Off Rank"] = df["O"].rank(ascending=False, method="min").astype(int)
    df["Def Rank"] = df["D"].rank(ascending=True,  method="min").astype(int)  # lower GA better
    df["Tier"] = df["S"].apply(tier_from_s)
    df["Style"] = df.apply(lambda r: style_tag(r["O"], r["D"]), axis=1)
    # optional: League Rank if provided; otherwise derive from Pts or leave blank
    return df

# ---------- UI ----------
st.title("⚽ Tier-based Matchup Predictor")

with st.expander("1) Select League and Season", expanded=True):
    st.markdown("**Choose a league to automatically fetch current standings:**")
    
    # Fetch available leagues
    with st.spinner("Loading available leagues..."):
        leagues = fetch_available_leagues()
    
    if leagues:
        # Create league selection
        league_options = {f"{league['name']} ({league['abbr']})": league['id'] for league in leagues}
        selected_league_name = st.selectbox("Select League:", list(league_options.keys()))
        selected_league_id = league_options[selected_league_name]
        
        # Season selection
        current_year = datetime.utcnow().year
        season = st.number_input("Season:", min_value=2020, max_value=current_year+1, value=current_year)
        
        # Fetch standings button
        if st.button("Fetch Standings", type="primary"):
            with st.spinner(f"Fetching {selected_league_name} standings..."):
                df, league_name = fetch_league_standings(selected_league_id, season)
                
            if df is not None:
                df = compute_table(df)
                st.success(f"✅ {league_name} standings loaded successfully!")
                st.dataframe(df.sort_values("S", ascending=False), width='stretch')
            else:
                st.error("Failed to fetch standings. Please try again.")
    else:
        st.error("Unable to load leagues. Please check your internet connection.")
        st.info("You can still use the manual CSV upload option below.")
        
        # Fallback to CSV upload
        st.markdown("**Manual CSV Upload (Fallback):**")
        st.markdown("**Required columns:** `Club, MP, GF, GA`")
        sample = """Club,MP,GF,GA,Rank
Real Madrid,5,10,2,1
Barcelona,5,16,3,2
Villarreal,5,10,4,3
Espanyol,5,8,7,4
Elche,5,7,4,5
Real Betis,6,9,7,6
Athletic Club,5,6,6,7
Getafe,5,6,7,8
Sevilla,5,9,8,9
Alavés,5,5,5,10
Valencia,5,6,8,11
Atlético Madrid,5,6,5,12
Osasuna,5,4,4,13
Rayo Vallecano,5,5,6,14
Celta Vigo,6,5,7,15
Levante,5,9,9,16
Oviedo,5,1,8,17
Real Sociedad,5,5,9,18
Mallorca,5,5,10,19
Girona,5,2,15,20
"""
        c1, c2 = st.columns([2,1])
        with c1:
            text = st.text_area("Paste CSV here", sample, height=250)
        with c2:
            up = st.file_uploader("...or upload CSV", type=["csv"])
        if up is not None:
            raw_df = pd.read_csv(up)
        else:
            raw_df = pd.read_csv(io.StringIO(text))

        df = compute_table(raw_df)
        st.success("Standings processed.")
        st.dataframe(df.sort_values("S", ascending=False), width='stretch')

with st.expander("2) Enter fixtures", expanded=True):
    st.markdown("One per line as `Home vs Away`")
    fx_sample = """Espanyol vs Valencia
Athletic Club vs Girona
Sevilla vs Villarreal
Levante vs Real Madrid
Getafe vs Alavés
Atlético Madrid vs Rayo Vallecano
Real Sociedad vs Mallorca
Osasuna vs Elche
Oviedo vs Barcelona"""
    fixtures_text = st.text_area("Fixtures", fx_sample, height=220)

fixtures = []
for line in fixtures_text.splitlines():
    if " vs " in line:
        h, a = [x.strip() for x in line.split(" vs ", 1)]
        if h and a:
            fixtures.append((h, a))

st.markdown("---")
st.subheader("Predictions")

# Check if we have standings data
if 'df' not in locals():
    st.warning("⚠️ Please fetch standings data first before making predictions.")
    st.stop()

rows = []
missing = []
for h, a in fixtures:
    try:
        rows.append(predict_row(h, a, df))
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