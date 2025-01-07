import sqlite3
import pandas as pd

def init_database(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create predictions table with all necessary fields
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_outcome TEXT,
            actual_outcome TEXT,
            home_odds REAL,
            draw_odds REAL,
            away_odds REAL,
            confidence REAL,
            bet_amount REAL,
            profit_loss REAL,
            prediction_type TEXT,
            status TEXT,
            match_date TEXT,
            match_id TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def read_predictions(db_path):
    print(f"\nReading from database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Get list of tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", [table[0] for table in tables])
    
    # Read all data from predictions table
    query = "SELECT * FROM predictions"
    try:
        df = pd.read_sql_query(query, conn)
        
        # Display the data
        if len(df) > 0:
            print("\nTotal predictions:", len(df))
            print("\nPredictions data:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df.to_string())
        else:
            print("\nNo predictions found in the database.")
    except Exception as e:
        print(f"Error reading predictions: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    db_path = 'predictions.db'
    init_database(db_path)
    read_predictions(db_path)
