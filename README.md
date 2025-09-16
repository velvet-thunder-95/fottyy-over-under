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
```bash
cp .env.example .env
```

Edit `.env` and add your actual credentials:
```env
# Football API Configuration
FOOTBALL_API_KEY=your_football_api_key_here

# Supabase Configuration  
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here

# Azure PostgreSQL Configuration
AZURE_PG_HOST=your-azure-host.postgres.database.azure.com
AZURE_PG_USER=your_username
AZURE_PG_PASSWORD=your_password
AZURE_PG_DATABASE=your_database_name
AZURE_PG_PORT=5432

# Application Authentication
APP_USERNAME_1=your_username_1
APP_PASSWORD_1=your_password_1
APP_USERNAME_2=your_username_2
APP_PASSWORD_2=your_password_2
```

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
