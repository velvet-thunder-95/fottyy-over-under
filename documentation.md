# Fottyy: Football Match Prediction System



## Introduction

Fottyy is a sophisticated football match prediction system built using Streamlit, XGBoost, and various data APIs. The application provides match outcome predictions, betting odds analysis, and performance tracking for football matches across numerous leagues worldwide.

The system combines machine learning predictions with market odds and team performance metrics to generate comprehensive match analyses. It includes features for tracking prediction history, analyzing prediction performance by league and confidence level, and calculating expected value (EV) for potential bets.

## System Architecture

Fottyy follows a modular architecture with the following key components:

1. **Data Collection Layer**: Integrates with football-data.api and Transfermarkt API to gather match data, team statistics, and market values.

2. **Prediction Engine**: Uses an XGBoost model to predict match outcomes based on comprehensive feature engineering.

3. **Odds Generation System**: Converts probabilities to betting odds and calculates expected value.

4. **Persistence Layer**: Uses Supabase for storing prediction history and results.

5. **User Interface**: Streamlit-based interface with multiple pages for match prediction, history tracking, and performance analytics.

The application is designed to be extensible, with clear separation between data collection, prediction logic, and presentation layers.

## Machine Learning Model

### Model Type and Implementation

Fottyy uses an **XGBoost (eXtreme Gradient Boosting)** model for match outcome prediction. XGBoost is an ensemble learning algorithm that uses gradient boosted decision trees, known for its performance and accuracy in classification tasks.

The model is loaded from a saved file using joblib:


The model is designed to predict three possible outcomes for a football match:
- HOME win
- AWAY win

The model implementation includes handling for both scikit-learn's XGBoostClassifier and the native XGBoost Booster:


### Feature Engineering

The model uses an extensive set of features that can be categorized into several groups:

#### Team Performance Features
- Win rates (`home_win_rate`, `away_win_rate`)
- Points per game (PPG) metrics
- Form points (`home_form_points`, `away_form_points`)
- Momentum indicators (`home_momentum`, `away_momentum`)

#### Match Statistics Features
- Expected goals (xG) metrics (`home_xg`, `away_xg`)
- Shot statistics (`home_shots`, `away_shots`, `home_shots_on_target`, `away_shots_on_target`)
- Possession percentages (`home_possession`, `away_possession`)
- Corner statistics (`home_corners`, `away_corners`)
- Foul statistics (`home_fouls`, `away_fouls`)

#### Derived Comparative Features
- Win rate difference (`win_rate_difference`)
- Possession difference (`possession_difference`)
- xG difference (`xg_difference`)
- Shot difference (`shot_difference`)
- Momentum difference (`momentum_difference`)
- Form difference (`form_difference`)

#### Odds-based Features
- Bookmaker odds (`odds_home_win`, `odds_draw`, `odds_away_win`)
- Implied probabilities from odds (`implied_home_prob`, `implied_draw_prob`, `implied_away_prob`)
- Odds ratio (`odds_ratio`)
- Sum of implied probabilities (`implied_prob_sum`)

#### Contextual Features
- Season (`season`)
- Competition ID (`competition_id`)
- Match counts (`home_total_matches`, `away_total_matches`)

The feature creation process is handled by the `create_match_features_from_api` function, which:

1. Extracts raw data from the API response
2. Applies data cleaning and validation
3. Handles missing values with sensible defaults
4. Calculates derived features
5. Ensures all features are in the correct format and order

The function includes extensive error handling to ensure robustness, with fallback values for missing or invalid data.

#### Feature Order and Importance

The model requires features to be provided in a specific order, which is maintained through the `feature_order` list:

```python
feature_order = [
    'win_rate_difference', 'possession_difference', 'xg_difference',
    'shot_difference', 'momentum_difference', 'implied_prob_sum',
    'form_difference', 'odds_ratio', 'total_momentum',
    'home_win_rate', 'away_win_rate', 'home_possession', 'away_possession',
    'home_xg', 'away_xg', 'home_shots', 'away_shots', 'home_momentum',
    'away_momentum', 'implied_home_prob', 'implied_draw_prob', 'implied_away_prob',
    'home_form_points', 'away_form_points', 'odds_home_win', 'odds_draw',
    'odds_away_win', 'season', 'competition_id', 'home_total_matches',
    'away_total_matches', 'home_shots_on_target', 'away_shots_on_target',
    'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
    'shot_accuracy_home', 'shot_accuracy_away', 'home_win_rate_ratio'
]
```

The order of features in this list reflects their importance in the model, with difference-based features (comparing home vs. away team) generally having higher importance.

### Training Process

While the training code is not directly included in the application (as it uses a pre-trained model), the feature engineering and model structure indicate a sophisticated training process that likely involved:

1. **Data Collection**: Gathering historical match data with outcomes
2. **Feature Engineering**: Creating the extensive feature set described above
3. **Feature Scaling**: Standardizing numerical features
4. **Model Training**: Using XGBoost with hyperparameter tuning
5. **Cross-Validation**: Ensuring model generalization
6. **Model Evaluation**: Assessing performance on test data

The model appears to be trained on a multi-class classification task (HOME/DRAW/AWAY outcomes) with probability outputs.

### Prediction Workflow

The prediction workflow follows these steps:

1. **Feature Creation**: Match data is transformed into model features
2. **Model Prediction**: Features are passed to the XGBoost model
3. **Outcome Determination**: The highest probability outcome is selected as the prediction
4. **Confidence Calculation**: Confidence is calculated based on probability margins

The prediction process is implemented in the `get_match_prediction` function:

```python
def get_match_prediction(match_data):
    # Create features DataFrame
    features_df = create_match_features_from_api(match_data)
    
    # Convert DataFrame to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(features_df)
    
    # Make prediction using model
    probabilities = predictor.predict(dmatrix)
    
    # Process and normalize probabilities
    # ...
    
    # Adjust probabilities based on odds and team strengths
    home_prob, draw_prob, away_prob = adjust_probabilities(
        home_prob, draw_prob, away_prob, match_data
    )
    
    return home_prob, draw_prob, away_prob
```

### Probability Adjustments

The model's raw probabilities are adjusted using market odds and team strength indicators through the `adjust_probabilities` function:

```python
def adjust_probabilities(home_prob, draw_prob, away_prob, match_data):
    # Weights for different factors
    model_weight = 0.5  # Model predictions
    odds_weight = 0.3   # Market odds
    form_weight = 0.2   # Team form
    
    # Calculate final probabilities
    final_home_prob = (home_prob * model_weight + 
                    odds_home_prob * odds_weight + 
                    form_home_prob * form_weight)
    
    # Similar calculations for draw and away probabilities
    
    # Normalize final probabilities
    # ...
    
    return final_home_prob, final_draw_prob, final_away_prob
```

This approach combines:
- Model predictions (50% weight)
- Market odds (30% weight)
- Team form indicators (20% weight)

The adjustment process ensures that the final probabilities incorporate both the machine learning model's predictions and market knowledge embedded in the odds.



The confidence levels are categorized as:
- **High**: ≥ 70% confidence
- **Medium**: 50-70% confidence
- **Low**: < 50% confidence

## Additional Prediction Models

### Over/Under 2.5 Goals Model

The system includes a statistical model for predicting the probability of over 2.5 total goals in a match using the Poisson distribution:

```python
def calculate_over25_probability(home_xg, away_xg):
    # Calculate probability of 3 or more goals
    for i in range(max_goals):
        for j in range(max_goals):
            if i + j > 2:  # Over 2.5 goals
                p1 = poisson.pmf(i, home_xg)
                p2 = poisson.pmf(j, away_xg)
                total_prob += p1 * p2
    return total_prob
```

This model:
1. Takes expected goals (xG) for both teams as input
2. Uses the Poisson distribution to model goal scoring
3. Calculates the probability of the total goals exceeding 2.5
4. Includes error handling for invalid xG values

### Both Teams To Score (BTTS) Model

Similarly, the system predicts the probability of both teams scoring at least one goal:

```python
def calculate_btts_probability(home_xg, away_xg):
    # Probability of home team scoring at least 1
    home_scoring_prob = 1 - poisson.pmf(0, home_xg)
    
    # Probability of away team scoring at least 1
    away_scoring_prob = 1 - poisson.pmf(0, away_xg)
    
    # Probability of both teams scoring
    btts_prob = home_scoring_prob * away_scoring_prob
    return btts_prob
```

This model:
1. Calculates the probability of each team scoring at least one goal
2. Multiplies these probabilities to get the BTTS probability
3. Handles error cases with reasonable defaults





### Expected Value Calculation

The system calculates Expected Value (EV) for potential bets:

```python
def calculate_ev(predicted_prob, odds):
    # Convert probability to decimal (0-1)
    prob_decimal = predicted_prob / 100
    
    # Calculate breakeven odds
    breakeven_odds = 1 / prob_decimal
    
    # Calculate EV percentage
    ev_percentage = (odds / breakeven_odds - 1) * 100
    
    return round(ev_percentage, 2)
```

This EV calculation:
1. Takes the predicted probability and market odds as input
2. Calculates the breakeven odds (fair odds without margin)
3. Compares market odds to breakeven odds to determine EV
4. Returns the EV as a percentage

The system uses color coding to visually represent EV values:
- Dark Green: PLUS EV over 25%
- Light Green: PLUS EV 15-25%
- Very Light Green: PLUS EV 5-15%
- Yellow: Breakeven -5% to 5%
- Light Red: MINUS EV -5% to -15%
- Dark Red: MINUS EV below -15%

## Data Sources and APIs

### Football-Data API

The application uses the Football-Data API as its primary source of match data:

The API provides:
- Upcoming match schedules
- Team statistics
- Match results
- League standings
- Historical performance data

The system supports a comprehensive list of leagues, defined in the `LEAGUE_IDS` dictionary, which maps league names to their API IDs. This includes leagues from:
- Major European leagues (Premier League, La Liga, Serie A, etc.)
- Secondary European leagues
- South American leagues
- Asian leagues
- African leagues
- North American leagues

The `get_matches` function retrieves matches for a specific date:

```python
def get_matches(date_str):
    """Get matches for a specific date"""
    try:
        url = f"{BASE_URL}/matches-by-date/{date_str}?key={API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if data.get('data'):
            return data['data']
        else:
            print(f"No matches found for {date_str}")
            return []
    except Exception as e:
        print(f"Error fetching matches: {str(e)}")
        return []
```

### Transfermarkt API

The application integrates with the Transfermarkt API to obtain team market values:


This API provides:
- Team market values
- Player valuations
- Team statistics
- Transfer information

### Data Caching

The application implements caching mechanisms to reduce API calls and improve performance:

```python
@st.cache_resource
def load_model():
    # Model loading code...
```

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_values(home_team, away_team):
    # Market value retrieval code...
```

This caching strategy:
1. Reduces API call frequency
2. Improves application responsiveness
3. Minimizes the risk of hitting API rate limits

## Streamlit Application

### Main Application (app.py)

The main application file (`app.py`) serves as the entry point for the Streamlit application and contains:

1. **Application Setup**:
   - Page configuration
   - Custom CSS styling
   - Authentication system

2. **Match Prediction Interface**:
   - Date selection
   - League filtering
   - Match display with team logos
   - Prediction visualization

3. **Match Analysis Components**:
   - Probability bars for match outcomes
   - Team market value comparison
   - Expected goals (xG) display
   - Odds comparison

4. **Prediction Processing**:
   - Feature creation
   - Model prediction
   - Confidence calculation
   - Result storage

The main application flow is controlled by the `show_main_app` function, which:
1. Displays the date selector
2. Fetches matches for the selected date
3. Processes and displays predictions for each match
4. Provides navigation to other application sections

### Prediction History (history.py)

The `history.py` file implements the prediction history tracking system:

1. **Database Integration**:
   - Supabase connection setup
   - Prediction storage
   - Result updating

2. **History Display**:
   - Tabular presentation of predictions
   - Filtering by date, league, and confidence
   - Styling based on prediction outcome

3. **Performance Metrics**:
   - Success rate calculation
   - Profit/loss tracking
   - ROI computation

The `PredictionHistory` class manages the prediction data:

```python
class PredictionHistory:
    def __init__(self):
        """Initialize the Supabase database connection."""
        self.db = SupabaseDB()

    def add_prediction(self, prediction_data):
        """Add a new prediction to the database"""
        # Implementation...

    def update_prediction_result(self, prediction_id, actual_outcome, profit_loss, home_score=None, away_score=None):
        """Update prediction with actual result and profit/loss"""
        # Implementation...

    def calculate_statistics(self, confidence_levels=None, leagues=None, start_date=None, end_date=None):
        """Calculate prediction statistics with optional confidence level and league filters"""
        # Implementation...
```

The history page UI is implemented in the `show_history_page` function, which:
1. Displays filters for date range, leagues, and confidence levels
2. Shows summary statistics for filtered predictions
3. Presents a styled table of prediction history
4. Provides savable filter presets

### Analytics Dashboard (graph_page.py)

The `graph_page.py` file implements an analytics dashboard for analyzing prediction performance:

1. **Performance by League**:
   - Success rate by league
   - Profit/loss by league
   - ROI by league

2. **Performance by Confidence Level**:
   - High confidence performance
   - Medium confidence performance
   - Low confidence performance

3. **Combined Analysis**:
   - Cross-tabulation of league and confidence
   - Summary statistics for different confidence bands

The analytics are presented in a tabular format with color coding to highlight performance:
- Green for profitable outcomes
- Red for losing outcomes
- Color gradients for success rates

### User Interface Design

The application features a sophisticated UI design with:

1. **Custom CSS Styling**:
   ```css
   .stApp {
       background-color: #f0f2f6;
   }
   
   .match-card {
       background: white;
       padding: 1.5rem;
       border-radius: 12px;
       box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
   }
   ```

2. **Responsive Layout**:
   - Adapts to different screen sizes
   - Uses Streamlit columns for horizontal layouts
   - Maintains readability on mobile devices

3. **Visual Elements**:
   - Team logos
   - Color-coded probability bars
   - Status indicators for predictions
   - Progress bars for probabilities

4. **Interactive Components**:
   - Date selectors
   - League filters
   - Confidence level filters
   - Navigation buttons

## Database Integration

### Supabase Implementation

The application uses Supabase as its database backend:


The Supabase integration provides:
1. Secure data storage
2. Real-time updates
3. SQL querying capabilities
4. User authentication

### Data Schema

The primary data schema includes:

1. **Predictions Table**:
   - `id`: Unique identifier
   - `date`: Match date
   - `league`: League name
   - `home_team`: Home team name
   - `away_team`: Away team name
   - `predicted_outcome`: Predicted result (HOME/DRAW/AWAY)
   - `actual_outcome`: Actual result (HOME/DRAW/AWAY)
   - `home_odds`: Decimal odds for home win
   - `draw_odds`: Decimal odds for draw
   - `away_odds`: Decimal odds for away win
   - `confidence`: Prediction confidence percentage
   - `bet_amount`: Fixed bet amount (1.0)
   - `profit_loss`: Calculated profit or loss
   - `status`: Prediction status (Pending/Completed)
   - `match_id`: External match identifier
   - `home_score`: Home team goals (for completed matches)
   - `away_score`: Away team goals (for completed matches)

## Performance Tracking System

### Profit/Loss Calculation

The system calculates profit and loss for each prediction:

```python
# Calculate profit/loss with fixed bet amount
if actual_outcome == pred['predicted_outcome']:
    if actual_outcome == "HOME":
        profit = float(pred['home_odds']) - 1
    elif actual_outcome == "AWAY":
        profit = float(pred['away_odds']) - 1
    else:
        profit = float(pred['draw_odds']) - 1
else:
    profit = -1
```

This calculation:
1. Uses a fixed $1 bet amount for consistency
2. For winning bets: P/L = (odds * bet_amount) - bet_amount
3. For losing bets: P/L = -bet_amount

### ROI and Success Rate Metrics

The system calculates key performance metrics:

```python
# Calculate success rate
success_rate = (correct_predictions / completed_count * 100) if completed_count > 0 else 0.0

# Calculate ROI using completed bets only (each bet is £1)
roi = (total_profit / completed_count * 100) if completed_count > 0 else 0.0
```

These metrics provide:
1. Success rate: Percentage of correct predictions
2. ROI: Return on investment as a percentage of bet amount
3. Total profit: Cumulative profit/loss across all bets

### Confidence Level Analysis

The system analyzes performance by confidence level:

```python
def get_confidence_level(confidence):
    """Convert confidence value to display text"""
    try:
        confidence = float(confidence)
        if confidence >= 70:
            return "High"
        elif confidence >= 50:
            return "Medium"
        else:
            return "Low"
    except (ValueError, TypeError):
        return "Unknown"
```

This allows for:
1. Filtering predictions by confidence level
2. Analyzing performance across different confidence bands
3. Identifying which confidence levels yield the best ROI

The analytics dashboard provides detailed breakdowns of performance by confidence level, helping users understand which types of predictions are most profitable.


### Dependencies

The application relies on several key libraries:

- **Streamlit**: Web application framework
- **XGBoost**: Machine learning model
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning utilities
- **Requests**: API communication
- **SciPy**: Statistical functions
- **Supabase**: Database integration

