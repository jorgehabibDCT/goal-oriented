# Goalflux ⚽

**Intelligent Football Matchup Predictor**

Goalflux is a powerful Streamlit application that analyzes football team performance, predicts match outcomes, and provides betting insights using tier-based analysis and real-time standings data.

## Features

- **Live Standings Integration**: Automatically fetch current standings from ESPN API
- **Tier-Based Analysis**: Classify teams as Elite, Strong, Balanced, Weak, or Relegation
- **Match Predictions**: Get 1X2 predictions, goals analysis, and BTTS insights
- **Team Style Analysis**: Identify playing styles (Attack-heavy, Defense-first, etc.)
- **Dynamic Team Selection**: Easy fixture creation with team dropdowns
- **Export Results**: Download predictions as CSV

## Supported Leagues

- English Premier League
- Spanish La Liga
- Italian Serie A
- German Bundesliga
- French Ligue 1
- Dutch Eredivisie
- Portuguese Primeira Liga
- Belgian Pro League
- Turkish Süper Lig
- Russian Premier League

## Quick Start

1. **Select a League**: Choose from the dropdown menu
2. **Fetch Standings**: Click to load current season data
3. **Create Fixtures**: Use team selectors to add matchups
4. **Get Predictions**: View tier-based analysis and predictions
5. **Export Results**: Download your analysis as CSV

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Technology Stack

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Requests**: API integration
- **ESPN API**: Live football data
- **NumPy**: Numerical computations

---

*Goalflux - Where data meets football intelligence* ⚽