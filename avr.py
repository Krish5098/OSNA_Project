import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the webpage
url = "https://www.elections.alaska.gov/research/statistics/#aeqstats"

# Send a GET request to the webpage
try:
    response = requests.get(url, verify=False)  # Bypass SSL verification
    response.raise_for_status()  # Raise an error for bad responses
except requests.exceptions.RequestException as e:
    print(f"Error fetching the URL: {e}")
    exit()

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find the relevant table
table = soup.find('table')

# Check if the table was found
if table is None:
    print("No table found on the webpage.")
    exit()

# Initialize lists to hold the data
data = []

# Iterate over each row in the table
for row in table.find_all('tr'):
    cols = row.find_all(['td', 'th'])  # Include header cells as well
    # Extract text from each column and strip whitespace
    cols = [ele.text.strip() for ele in cols]
    if cols:  # Check if the row is not empty
        data.append(cols)

# Display the data extracted for debugging
print("Extracted Data:")
for d in data:
    print(d)

# Check lengths of extracted rows for debugging
for index, row in enumerate(data):
    print(f"Row {index}: {len(row)} columns -> {row}")

# Create a DataFrame from the data if enough columns exist
if len(data) > 1 and len(data[0]) == 14:  # Adjust according to the expected number of columns
    columns = ["PFD Year", "Valid Applications", "Potential New Voters",
               "Existing Voters1", "Ambiguous Voters", "Total",
               "Potential New Voters", "Existing Voters2", "Total",
               "New Voters", "Existing Voters3", "New Voters Who Voted4",
               "Print and Mail Costs", "Total Annual Cost"]

    df = pd.DataFrame(data[1:], columns=columns)  # Skip the header row if present
else:
    print(f"Expected 14 columns but found {len(data[0]) if data else 0}.")
    exit()

# Save the DataFrame to a CSV file
csv_file_path = 'alaska_voter_registration_data.csv'
df.to_csv(csv_file_path, index=False)

print(f"Data has been scraped and saved to '{csv_file_path}'.")
