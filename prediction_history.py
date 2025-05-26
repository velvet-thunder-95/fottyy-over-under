import pandas as pd
from datetime import datetime, timedelta
from supabase_db import SupabaseDB

class PredictionHistory:
    def __init__(self):
        """Initialize the Supabase database connection."""
        self.db = SupabaseDB()

    def init_database(self):
        """Initialize the Supabase database"""
        # No need to create tables as they are managed in Supabase dashboard
        self.db.init_database()

    def add_prediction(self, prediction_data):
        """Add a new prediction to the database"""
        return self.db.add_prediction(prediction_data)

    def get_predictions(self, start_date=None, end_date=None, confidence_levels=None, leagues=None, status=None):
        """Get predictions with optional filters
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            confidence_levels (list, optional): List of confidence levels to filter by
            leagues (list, optional): List of leagues to filter by
            status (str, optional): Status to filter by (e.g., 'Pending', 'Completed')
            
        Returns:
            pd.DataFrame: Filtered predictions
        """
        # Build filters dictionary
        filters = {}
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
            
        # Get base predictions
        predictions = self.db.get_predictions(start_date=start_date, end_date=end_date)
        
        # Apply additional filters
        if not predictions.empty:
            if status:
                predictions = predictions[predictions['status'] == status]
                
            if confidence_levels:
                predictions = predictions[predictions['confidence'].isin(confidence_levels)]
                
            if leagues:
                predictions = predictions[predictions['league'].isin(leagues)]
        
        return predictions

    def update_prediction(self, prediction_id, update_data):
        """Update an existing prediction"""
        return self.db.update_prediction(prediction_id, update_data)

    def delete_prediction(self, prediction_id):
        """Delete a prediction from the database"""
        return self.db.delete_prediction(prediction_id)

    def get_prediction_by_id(self, prediction_id):
        """Get a single prediction by ID"""
        return self.db.get_prediction_by_id(prediction_id)

    def get_all_leagues(self):
        """Get all unique leagues from predictions"""
        return self.db.get_all_leagues()

    def get_all_teams(self):
        """Get all unique teams from predictions"""
        return self.db.get_all_teams()

    def get_predictions_by_date_range(self, start_date, end_date):
        """Get predictions within a date range"""
        return self.db.get_predictions_by_date_range(start_date, end_date)

    def get_predictions_by_league(self, league):
        """Get predictions for a specific league"""
        return self.db.get_predictions_by_league(league)

    def update_match_results_all(self):
        """Update match results for pending predictions only"""
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        from match_analyzer import MatchAnalyzer
        analyzer = MatchAnalyzer("633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49")
        
        # Get only pending predictions that have a match_id
        result = self.db.supabase.table('predictions') \
            .select('*') \
            .eq('status', 'Pending') \
            .not_.eq('match_id', '') \
            .execute()
            
        pending_predictions = result.data if hasattr(result, 'data') else []
        
        if not pending_predictions:
            logger.info("No pending predictions to update")
            return
            
        updated_count = 0
        
        for pred in pending_predictions:
            try:
                match_id = pred.get('match_id')
                home_team = pred.get('home_team')
                away_team = pred.get('away_team')
                
                if not match_id or not home_team or not away_team:
                    continue
                    
                # Get match result using match_id
                result = analyzer.analyze_match_result(match_id)
                
                if not result:
                    logger.debug(f"No result found for match {match_id}")
                    continue
                    
                # Check if the match has completed
                api_status = result.get('status')
                
                if api_status == 'Completed':
                    # Determine the actual outcome
                    home_score = result.get('home_score')
                    away_score = result.get('away_score')
                    actual_outcome = None
                    
                    if home_score > away_score:
                        actual_outcome = '1'
                    elif away_score > home_score:
                        actual_outcome = '2'
                    else:
                        actual_outcome = 'X'
                    
                    # Update the result
                    self.update_prediction(pred.get('id'), {
                        'status': 'Completed',
                        'home_score': home_score,
                        'away_score': away_score,
                        'actual_outcome': actual_outcome
                    })
                    logger.info(f"Updated {home_team} vs {away_team} - Match completed with result")
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing prediction {pred.get('id')}: {str(e)}")
        
        logger.info(f"Updated results for {updated_count} matches")
        return updated_count
        
    def calculate_statistics(self, confidence_levels=None, leagues=None, start_date=None, end_date=None):
        """Calculate prediction statistics with optional confidence level and league filters"""
        try:
            # Get all predictions first using our paginated get_predictions method
            predictions = self.get_predictions(
                start_date=start_date,
                end_date=end_date,
                confidence_levels=confidence_levels,
                leagues=leagues
            )
            
            if predictions is None or predictions.empty:
                return [0, 0, 0.0, 0.0, 0.0], 0
            
            # Calculate statistics
            completed_predictions = predictions[predictions['status'] == 'Completed']
            pending_predictions = predictions[predictions['status'] == 'Pending']
            
            total_predictions = len(predictions)
            completed_count = len(completed_predictions)
            pending_count = len(pending_predictions)
            
            if completed_count == 0:
                return [total_predictions, 0, 0.0, 0.0, 0.0], pending_count
            
            # Calculate correct predictions
            correct_predictions = len(
                completed_predictions[
                    completed_predictions['predicted_outcome'] == 
                    completed_predictions['actual_outcome']
                ]
            )
            
            # Calculate success rate
            success_rate = (correct_predictions / completed_count * 100) if completed_count > 0 else 0.0
            
            # Calculate total profit/loss and ROI
            total_profit = completed_predictions['profit_loss'].sum()
            
            # Calculate ROI using completed bets only (each bet is £1)
            roi = (total_profit / completed_count * 100) if completed_count > 0 else 0.0
            
            # Debug info
            import logging
            logging.info("Statistics calculation:")
            logging.info(f"Total predictions: {total_predictions}")
            logging.info(f"Completed predictions: {completed_count}")
            logging.info(f"Pending predictions: {pending_count}")
            logging.info(f"Correct predictions: {correct_predictions}")
            logging.info(f"Success rate: {success_rate:.2f}%")
            logging.info(f"Total profit: £{total_profit:.2f}")
            logging.info(f"ROI: {roi:.2f}%")
            logging.info(f"Date range: {predictions['date'].min()} to {predictions['date'].max()}")
            
            return [total_predictions, correct_predictions, success_rate, total_profit, roi], pending_count
            
        except Exception as e:
            import logging
            logging.error(f"Error calculating statistics: {str(e)}")
            return [0, 0, 0.0, 0.0, 0.0], 0
            
    def get_predictions_by_team(self, team):
        """Get predictions for a specific team"""
        return self.db.get_predictions_by_team(team)

    def get_predictions_by_confidence(self, min_confidence=None, max_confidence=None):
        """Get predictions within a confidence range"""
        return self.db.get_predictions_by_confidence(min_confidence, max_confidence)

    def get_predictions_by_result(self, result):
        """Get predictions with a specific result"""
        return self.db.get_predictions_by_result(result)

    def get_predictions_by_profit_loss(self, min_profit=None, max_profit=None):
        """Get predictions within a profit/loss range"""
        return self.db.get_predictions_by_profit_loss(min_profit, max_profit)

    def get_predictions_by_season(self, season):
        """Get predictions for a specific season"""
        return self.db.get_predictions_by_season(season)

    def get_predictions_by_competition(self, competition):
        """Get predictions for a specific competition"""
        return self.db.get_predictions_by_competition(competition)

    def get_predictions_by_prediction(self, prediction):
        """Get predictions with a specific prediction"""
        return self.db.get_predictions_by_prediction(prediction)

    def get_predictions_by_odds_range(self, min_odds=None, max_odds=None):
        """Get predictions within an odds range"""
        return self.db.get_predictions_by_odds_range(min_odds, max_odds)

    def get_predictions_by_stake(self, min_stake=None, max_stake=None):
        """Get predictions within a stake range"""
        return self.db.get_predictions_by_stake(min_stake, max_stake)

    def get_predictions_by_bookmaker(self, bookmaker):
        """Get predictions for a specific bookmaker"""
        return self.db.get_predictions_by_bookmaker(bookmaker)

    def get_predictions_by_strategy(self, strategy):
        """Get predictions using a specific strategy"""
        return self.db.get_predictions_by_strategy(strategy)

    def get_predictions_by_team_and_league(self, team, league):
        """Get predictions for a specific team in a specific league"""
        return self.db.get_predictions_by_team_and_league(team, league)

    def get_predictions_by_league_and_season(self, league, season):
        """Get predictions for a specific league and season"""
        return self.db.get_predictions_by_league_and_season(league, season)

    def get_predictions_by_date_and_league(self, date, league):
        """Get predictions for a specific date and league"""
        return self.db.get_predictions_by_date_and_league(date, league)

    def get_predictions_by_date_range_and_league(self, start_date, end_date, league):
        """Get predictions within a date range for a specific league"""
        return self.db.get_predictions_by_date_range_and_league(start_date, end_date, league)

    def get_predictions_by_date_range_and_team(self, start_date, end_date, team):
        """Get predictions within a date range for a specific team"""
        return self.db.get_predictions_by_date_range_and_team(start_date, end_date, team)

    def get_predictions_by_date_range_and_league_and_team(self, start_date, end_date, league, team):
        """Get predictions within a date range for a specific league and team"""
        return self.db.get_predictions_by_date_range_and_league_and_team(start_date, end_date, league, team)

    def get_predictions_by_date_range_and_league_and_season(self, start_date, end_date, league, season):
        """Get predictions within a date range for a specific league and season"""
        return self.db.get_predictions_by_date_range_and_league_and_season(start_date, end_date, league, season)

    def get_predictions_by_date_range_and_team_and_season(self, start_date, end_date, team, season):
        """Get predictions within a date range for a specific team and season"""
        return self.db.get_predictions_by_date_range_and_team_and_season(start_date, end_date, team, season)

    def get_predictions_by_date_range_and_league_and_team_and_season(self, start_date, end_date, league, team, season):
        """Get predictions within a date range for a specific league, team, and season"""
        return self.db.get_predictions_by_date_range_and_league_and_team_and_season(start_date, end_date, league, team, season)
