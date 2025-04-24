import json
import os
from datetime import datetime

def load_saved_filters():
    """Load saved filters from JSON file"""
    file_path = 'saved_filters.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def save_filters_to_file(filters):
    """Save filters to JSON file"""
    file_path = 'saved_filters.json'
    with open(file_path, 'w') as f:
        json.dump(filters, f, indent=4)

def add_filter(name, leagues, confidence_levels):
    """Add a new filter and save to file"""
    filters = load_saved_filters()
    new_filter = {
        "name": name,
        "leagues": leagues,
        "confidence": confidence_levels,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    filters.append(new_filter)
    save_filters_to_file(filters)
    return filters

def delete_filter(index):
    """Delete a filter by index and save to file"""
    filters = load_saved_filters()
    if 0 <= index < len(filters):
        filters.pop(index)
        save_filters_to_file(filters)
    return filters
