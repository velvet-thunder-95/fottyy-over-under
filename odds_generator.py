import numpy as np
from datetime import datetime, timedelta
import logging
from football_api import get_matches, get_match_result
from transfermarkt_api import TransfermarktAPI
from sklearn.preprocessing import MinMaxScaler
import json
import os
from scipy.stats import poisson

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OddsGenerator:
    def __init__(self, transfermarkt_api):
        """Initialize the OddsGenerator with necessary components"""
        self.transfermarkt_api = transfermarkt_api
        # Define reasonable odds ranges for each outcome
        self.base_odds_range = {
            'home': {'min': 1.05, 'max': 15.0},
            'draw': {'min': 2.0, 'max': 8.0},
            'away': {'min': 1.05, 'max': 15.0},
            'over25': {'min': 1.2, 'max': 3.5},
            'under25': {'min': 1.2, 'max': 3.5}
        }
        
        # Default probabilities when no data available
        self.default_probs = {
            'home': 0.40,
            'draw': 0.25,
            'away': 0.35
        }
        
    def _get_team_market_values(self, home_team, away_team):
        """Get market values for both teams"""
        try:
            # Clean team names first
            home_team_clean = self.transfermarkt_api.clean_team_name(home_team)
            away_team_clean = self.transfermarkt_api.clean_team_name(away_team)
            
            logger.info(f"Cleaned team names - Home: {home_team} -> {home_team_clean}, Away: {away_team} -> {away_team_clean}")
            
            # Get market values using cleaned names
            values = self.transfermarkt_api.get_multiple_teams_market_value([home_team_clean, away_team_clean])
            
            if not values:
                logger.warning(f"Could not get market values for teams: {home_team_clean}, {away_team_clean}")
                return None, None
                
            # Values come directly as integers from get_multiple_teams_market_value
            home_value = values.get(home_team_clean, 0)
            away_value = values.get(away_team_clean, 0)
            
            if home_value <= 0 or away_value <= 0:
                logger.warning(f"Invalid market values: home={home_value}, away={away_value}")
                return None, None
                
            logger.info(f"Market values - {home_team_clean}: €{home_value:,}, {away_team_clean}: €{away_value:,}")
            return home_value, away_value
            
        except Exception as e:
            logger.error(f"Error getting market values: {str(e)}")
            return None, None

    def _calculate_base_probabilities(self, home_market_value, away_market_value):
        """Calculate base probabilities from market values"""
        try:
            total_value = home_market_value + away_market_value
            if total_value == 0:
                return self.default_probs.copy()
            
            # Use log scale to handle large market value differences
            home_log = np.log(home_market_value + 1)  # Add 1 to avoid log(0)
            away_log = np.log(away_market_value + 1)
            total_log = home_log + away_log
            
            # Calculate raw probabilities based on log market values
            home_prob = float(home_log / total_log)  # Convert to Python float
            away_prob = float(away_log / total_log)  # Convert to Python float
            
            # Adjust for home advantage (increase home prob by 10%)
            home_advantage = 0.10
            home_prob *= (1 + home_advantage)
            away_prob *= (1 - home_advantage)
            
            # Calculate draw probability - higher when log values are closer
            strength_diff = abs(home_log - away_log) / total_log
            draw_prob = float(0.25 * (1 - strength_diff))  # Convert to Python float
            
            # Ensure minimum draw probability
            draw_prob = max(draw_prob, 0.15)
            
            # Normalize probabilities
            total = home_prob + away_prob + draw_prob
            probabilities = {
                'home': float(round(home_prob / total, 4)),  # Convert to Python float
                'draw': float(round(draw_prob / total, 4)),  # Convert to Python float
                'away': float(round(away_prob / total, 4))   # Convert to Python float
            }
            
            logger.info(f"Calculated probabilities from market values: {probabilities}")
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating base probabilities: {str(e)}")
            return self.default_probs.copy()

    def _get_team_form(self, team_name, recent_matches):
        """Calculate team form based on recent matches"""
        form_score = 0
        matches_counted = 0
        
        for match in recent_matches:
            if matches_counted >= 5:  # Only consider last 5 matches
                break
                
            if match['home_team'] == team_name:
                if match['home_score'] > match['away_score']:
                    form_score += 3
                elif match['home_score'] == match['away_score']:
                    form_score += 1
            elif match['away_team'] == team_name:
                if match['away_score'] > match['home_score']:
                    form_score += 3
                elif match['home_score'] == match['away_score']:
                    form_score += 1
                    
            matches_counted += 1
            
        return form_score / (matches_counted * 3) if matches_counted > 0 else 0.5

    def _calculate_over_under_probabilities(self, home_xg, away_xg):
        """Calculate over/under probabilities using xG"""
        try:
            total_xg = home_xg + away_xg
            
            # Base probability from Poisson distribution
            over25_prob = 1 - poisson.cdf(2, total_xg)
            
            # Adjust based on xG distribution
            if total_xg > 3.0:
                over25_prob = min(0.85, over25_prob + 0.1)
            elif total_xg < 2.0:
                over25_prob = max(0.15, over25_prob - 0.1)
            
            return {
                'over25': over25_prob,
                'under25': 1 - over25_prob
            }
        except Exception as e:
            logger.error(f"Error calculating over/under probabilities: {str(e)}")
            return {'over25': 0.5, 'under25': 0.5}

    def _calculate_ev(self, probability, odds):
        """Calculate expected value for a bet
        EV = (Probability * (Odds-1)) - (1-Probability)
        """
        try:
            if not (0 <= probability <= 1) or odds <= 1:
                return -100  # Clearly unfavorable bet
            
            ev = (probability * (odds - 1)) - (1 - probability)
            return round(ev * 100, 2)  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Error calculating EV: {str(e)}")
            return -100

    def _probabilities_to_odds(self, probabilities, margin=0.1):
        """Convert probabilities to odds with a specified margin"""
        try:
            # Validate probabilities
            if not probabilities or not all(0 <= p <= 1 for p in probabilities.values()):
                logger.warning("Invalid probabilities, using defaults")
                return {k: v['min'] for k, v in self.base_odds_range.items()}

            # Normalize probabilities
            total = sum(probabilities.values())
            if not (0.99 <= total <= 1.01):
                logger.warning(f"Probability total ({total}) not close to 1, normalizing")
                probabilities = {k: v/total for k, v in probabilities.items()}
            
            # Convert to odds with margin
            margin_per_outcome = margin / len(probabilities)
            odds = {}
            
            for outcome, prob in probabilities.items():
                # Add margin and convert to odds
                prob_with_margin = prob * (1 - margin_per_outcome)
                
                if prob_with_margin <= 0:
                    odds[outcome] = self.base_odds_range[outcome]['max']
                else:
                    raw_odds = 1 / prob_with_margin
                    # Ensure odds are within reasonable ranges
                    min_odd = self.base_odds_range[outcome]['min']
                    max_odd = self.base_odds_range[outcome]['max']
                    odds[outcome] = round(max(min(raw_odds, max_odd), min_odd), 2)
            
            return odds
            
        except Exception as e:
            logger.error(f"Error converting probabilities to odds: {str(e)}")
            return {k: v['min'] for k, v in self.base_odds_range.items()}

    def _calculate_probabilities_from_stats(self, match_data):
        """Calculate probabilities using FootyStats data when market values aren't available"""
        try:
            # Get team stats
            home_ppg = float(match_data.get('pre_match_teamA_overall_ppg', 0))
            away_ppg = float(match_data.get('pre_match_teamB_overall_ppg', 0))
            home_xg = float(match_data.get('team_a_xg_prematch', 0))
            away_xg = float(match_data.get('team_b_xg_prematch', 0))
            
            if home_ppg <= 0 or away_ppg <= 0:
                logger.warning("Invalid PPG values")
                return None
                
            # Calculate base probabilities from PPG
            total_ppg = home_ppg + away_ppg
            home_prob = home_ppg / total_ppg
            away_prob = away_ppg / total_ppg
            
            # Adjust for home advantage
            home_advantage = 0.15
            home_prob *= (1 + home_advantage)
            away_prob *= (1 - home_advantage)
            
            # Use xG to influence probabilities if available
            if home_xg > 0 and away_xg > 0:
                total_xg = home_xg + away_xg
                xg_home_prob = home_xg / total_xg
                xg_away_prob = away_xg / total_xg
                
                # Blend PPG and xG probabilities (60% PPG, 40% xG)
                home_prob = (0.6 * home_prob) + (0.4 * xg_home_prob)
                away_prob = (0.6 * away_prob) + (0.4 * xg_away_prob)
            
            # Calculate draw probability based on potential values
            o25_potential = float(match_data.get('o25_potential', 50)) / 100
            u25_potential = float(match_data.get('u25_potential', 50)) / 100
            btts_potential = float(match_data.get('btts_potential', 50)) / 100
            
            # Higher draw probability if game is expected to be low scoring
            draw_factor = (u25_potential + (1 - btts_potential)) / 2
            draw_prob = 0.25 * (1 + draw_factor)
            
            # Normalize probabilities
            total = home_prob + away_prob + draw_prob
            return {
                'home': round(home_prob / total, 4),
                'draw': round(draw_prob / total, 4),
                'away': round(away_prob / total, 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating probabilities from stats: {str(e)}")
            return None

    def generate_odds(self, match_data):
        """Generate odds for a match"""
        try:
            # Extract team names
            home_team = match_data.get('home_name', '')
            away_team = match_data.get('away_name', '')
            
            if not home_team or not away_team:
                logger.error("Missing team names in match data")
                return None
            
            # Try market values first
            home_value, away_value = self._get_team_market_values(home_team, away_team)
            
            # Calculate probabilities
            if home_value and away_value:
                probabilities = self._calculate_base_probabilities(home_value, away_value)
            else:
                # Fall back to stats-based probabilities
                probabilities = self._calculate_probabilities_from_stats(match_data)
                
            if not probabilities:
                logger.warning("Using default probabilities as fallback")
                probabilities = self.default_probs.copy()
            
            # Validate and normalize probabilities
            total_prob = sum(probabilities.values())
            if not (0.99 <= total_prob <= 1.01):
                logger.warning(f"Invalid probability total ({total_prob}), normalizing")
                total = sum(probabilities.values())
                if total > 0:
                    probabilities = {k: float(v/total) for k, v in probabilities.items()}
                else:
                    logger.warning("Zero total probability, using defaults")
                    probabilities = self.default_probs.copy()
            
            # Additional validation to ensure no negative or >1 probabilities
            for outcome, prob in probabilities.items():
                if not (0 <= prob <= 1):
                    logger.warning(f"Invalid {outcome} probability: {prob}, using defaults")
                    probabilities = self.default_probs.copy()
                    break
            
            # Convert probabilities to odds with margin
            odds = self._probabilities_to_odds(probabilities)
            
            # Calculate over/under probabilities using xG
            home_xg = float(match_data.get('team_a_xg_prematch', 1.5))
            away_xg = float(match_data.get('team_b_xg_prematch', 1.0))
            
            over_under_probs = self._calculate_over_under_probabilities(home_xg, away_xg)
            over_under_odds = self._probabilities_to_odds(over_under_probs, margin=0.08)
            
            # Update match_data with probabilities and generated odds
            match_data['home_prob'] = float(probabilities['home'])
            match_data['draw_prob'] = float(probabilities['draw'])
            match_data['away_prob'] = float(probabilities['away'])
            
            # Store generated odds with correct keys
            match_data['generated_odds_1'] = float(odds['home'])
            match_data['generated_odds_x'] = float(odds['draw'])
            match_data['generated_odds_2'] = float(odds['away'])
            match_data['generated_odds_over25'] = float(over_under_odds['over25'])
            match_data['generated_odds_under25'] = float(over_under_odds['under25'])
            
            # Combine all odds
            odds.update(over_under_odds)
            
            return odds
            
        except Exception as e:
            logger.error(f"Error generating odds: {str(e)}")
            return None

    def get_odds(self, match_data, footystats_odds=None):
        """Get odds for a match, using FootyStats if available, otherwise generate them"""
        try:
            if footystats_odds:
                # Check if we have valid FootyStats odds for all markets
                has_match_odds = all([
                    footystats_odds.get('home_odds', 0),
                    footystats_odds.get('draw_odds', 0),
                    footystats_odds.get('away_odds', 0)
                ])
                
                has_over_under_odds = all([
                    footystats_odds.get('over25_odds', 0),
                    footystats_odds.get('under25_odds', 0)
                ])
                
                if has_match_odds and has_over_under_odds:
                    # Store FootyStats odds in match_data
                    match_data['odds_ft_1'] = float(footystats_odds['home_odds'])
                    match_data['odds_ft_x'] = float(footystats_odds['draw_odds'])
                    match_data['odds_ft_2'] = float(footystats_odds['away_odds'])
                    match_data['odds_ft_over25'] = float(footystats_odds['over25_odds'])
                    match_data['odds_ft_under25'] = float(footystats_odds['under25_odds'])
                    return footystats_odds
                
                # Generate odds for missing markets
                generated_odds = self.generate_odds(match_data)
                if not generated_odds:
                    logger.error("Failed to generate odds")
                    return None
                
                # Use FootyStats odds where available, generated odds where not
                odds = {
                    'home_odds': footystats_odds.get('home_odds') or generated_odds['home'],
                    'draw_odds': footystats_odds.get('draw_odds') or generated_odds['draw'],
                    'away_odds': footystats_odds.get('away_odds') or generated_odds['away'],
                    'over25_odds': footystats_odds.get('over25_odds') or generated_odds['over25'],
                    'under25_odds': footystats_odds.get('under25_odds') or generated_odds['under25']
                }
                
                # Ensure all odds are valid floats
                odds = {k: float(v) if v else float(self.base_odds_range[k.split('_')[0]]['min']) 
                       for k, v in odds.items()}
                
                return odds
            
            # No FootyStats odds, generate all odds
            generated_odds = self.generate_odds(match_data)
            if not generated_odds:
                logger.error("Failed to generate odds")
                return None
                
            return {
                'home_odds': float(generated_odds['home']),
                'draw_odds': float(generated_odds['draw']),
                'away_odds': float(generated_odds['away']),
                'over25_odds': float(generated_odds['over25']),
                'under25_odds': float(generated_odds['under25'])
            }
            
        except Exception as e:
            logger.error(f"Error in get_odds: {str(e)}")
            return None

    def _get_default_response(self):
        """Get default response when errors occur"""
        return {
            'probabilities': self.default_probs,
            'odds': {k: v['min'] for k, v in self.base_odds_range.items()},
            'evs': {k: 0 for k in self.default_probs.keys()},
            'goals': {'over25': 0.5, 'under25': 0.5}
        }
