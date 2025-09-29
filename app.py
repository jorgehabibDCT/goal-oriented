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
    "Russian Premier League": "rus.1"
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
st.title("⚽ Goalflux")
st.markdown("**Intelligent Football Matchup Predictor** - Analyze team tiers, predict outcomes, and discover betting insights")

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
            st.success(f"✅ {league_name} standings loaded successfully!")
        else:
            st.error("Failed to fetch standings. Please try again.")
    
    # Display standings if loaded (only once)
    if 'standings_df' in st.session_state:
        st.success(f"✅ {st.session_state.league_name} standings loaded!")
        st.info("ℹ️ **Teams are ordered by goals scored, then goals against**")
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