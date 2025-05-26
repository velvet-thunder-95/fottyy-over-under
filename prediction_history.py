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

    def get_predictions(self, filters=None):
        """Get predictions with optional filters"""
        return self.db.get_predictions(filters)

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
