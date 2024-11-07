import requests
from bs4 import BeautifulSoup
import csv
import os

url = 'https://www.elections.alaska.gov/election-polls/'

response = requests.get(url, verify=False)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    polling_data = []
    table = soup.find('table')
    rows = table.find_all('tr')

    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) >= 5:
            house_district = cells[0].get_text(strip=True)
            precinct_number = cells[1].get_text(strip=True)
            precinct_name = cells[2].get_text(strip=True)
            polling_place_name = cells[3].get_text(strip=True)
            address = cells[4].get_text(strip=True)
            polling_data.append({
                'House District': house_district,
                'Precinct Number': precinct_number,
                'Precinct Name': precinct_name,
                'Polling Place Name': polling_place_name,
                'Address': address
            })

    csv_file = r'C:\Users\mithi\electionprediction\polling_data.csv'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['House District', 'Precinct Number', 'Precinct Name', 'Polling Place Name', 'Address'])
        writer.writeheader()
        for data in polling_data:
            writer.writerow(data)

    print(f"Polling data saved to {csv_file}")
else:
    print(f"Failed to retrieve data: {response.status_code}")
