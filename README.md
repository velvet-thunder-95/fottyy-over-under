# Football Match Predictor

A Streamlit-based web application for predicting football match outcomes and tracking betting history.

## Features

- Real-time match data from football-data API
- Match prediction with betting functionality
- Historical prediction tracking
- Match analysis and statistics
- User authentication
- Clean and intuitive UI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/football-match-predictor.git
cd football-match-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Copy the example environment file and configure your secrets:
Configure your local secrets by editing `.streamlit/secrets.toml`:

```toml
# Football API Configuration
[football_api]
api_key = "your_football_api_key_here"

# Supabase Configuration
[supabase]
url = "https://your-project.supabase.co"
key = "your_supabase_anon_key_here"

# Azure PostgreSQL Configuration
[azure_postgres]
host = "your-azure-host.postgres.database.azure.com"
user = "your_username"
password = "your_password"
database = "your_database_name"
port = 5432

# Application Authentication
[auth]
username_1 = "your_username_1"
password_1 = "your_password_1"
username_2 = "your_username_2"
password_2 = "your_password_2"
```

**Note:** The `.streamlit/secrets.toml` file is gitignored and should never be committed to the repository. Automation scripts will use the same `auth` credentials for consistency.

5. Verify your setup:
```bash
python setup_check.py
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Log in and start making predictions!

## Project Structure

- `app.py`: Main application file with Streamlit UI
- `football_api.py`: Football data API integration
- `history.py`: Prediction history management
- `match_analyzer.py`: Match analysis functionality
- `read_db.py`: Database operations
- `session_state.py`: User session management

## Dependencies

- Python 3.8+
- Streamlit
- Pandas
- SQLite3
- Requests

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
