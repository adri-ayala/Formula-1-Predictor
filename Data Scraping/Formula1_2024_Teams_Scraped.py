#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import time
import pandas as pd
from bs4 import BeautifulSoup


# In[2]:


def fetch_team_data(team_url):
    data = requests.get(team_url)
    time.sleep(12)  
    races = pd.read_html(data.text)  
    return races[0]  


# In[3]:



# Function to get race results for all teams
def fetch_all_teams_race_results():
    standings_url = "https://www.formula1.com/en/results/2024/team"
    data = requests.get(standings_url)
    time.sleep(12)  # Throttle requests to avoid hitting rate limits
    
    # Parse the main page to get the links to individual teams
    soup = BeautifulSoup(data.text, 'html.parser')
    standings_table = soup.select('table.f1-table')[0]

    # Extract all team URLs
    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if 'team/' in l]  # Filter only team-related links
    team_urls = [f"https://formula1.com/en/results/2024/{l}" for l in links]

    all_team_data = []

    # Loop through all team URLs to fetch their race results
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ").title()
        
        # Debugging: Print the team being processed
        print(f"Fetching race results for {team_name} from {team_url}")
        
        # Fetch the race results for the current team
        team_race_results = fetch_team_data(team_url)
        
        # Add the team name as a column to the dataframe
        team_race_results['team_name'] = team_name
        
        # Append the team's race results to the all_team_data list
        all_team_data.append(team_race_results)

    # Combine data from all teams into one dataframe
    full_race_data = pd.concat(all_team_data, ignore_index=True)
    
    # Clean up column names (optional)
    full_race_data.columns = [col.lower() for col in full_race_data.columns]
    
    # Sort by 'date' to display races chronologically
    full_race_data['date'] = pd.to_datetime(full_race_data['date'], errors='coerce')  # Convert to datetime
    full_race_data = full_race_data.sort_values(by='date').reset_index(drop=True)
    
    # Format the columns to be more readable
    full_race_data['date'] = full_race_data['date'].dt.strftime('%d %b %Y')  # Format date nicely
    full_race_data['pts'] = full_race_data['pts'].astype(int)  # Ensure points are integers
    
    # Reorder the columns for better readability (team_name, grand prix, date, points)
    full_race_data = full_race_data[['team_name', 'grand prix', 'date', 'pts']]
    
    # Save to CSV
    full_race_data.to_csv("races.csv", index=False)
    print("CSV file saved: races.csv")
    
    # Return the DataFrame to print it as a nice table
    return full_race_data


# In[4]:


# Execute the function to fetch and save all race results, and store the result in a variable
full_race_data = fetch_all_teams_race_results()

# Now, display the data in a clean table format
import IPython.display as display
display.display(full_race_data)  # This will render the DataFrame as a nice table in Jupyter

# If you're running this in a regular Python script, you can print it
print(full_race_data.head())  # Display the first few rows of the DataFrame

