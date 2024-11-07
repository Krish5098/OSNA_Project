import pdfplumber
import pandas as pd
import re
import matplotlib.pyplot as plt

def extract_raw_data(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        raw_data = []
        for page in pdf.pages:
            # Extract text from the page
            text = page.extract_text()
            if text:
                # Split the text into lines and add to raw data
                lines = text.split('\n')
                raw_data.extend(lines)
    return raw_data
def tabulate_data(raw_data):
    tabulated_data = []
    current_house = None

    for line in raw_data:
        # Check for house number
        if "HOUSE:" in line:
            current_house = line.split(":")[1].strip()  # Extract house number
            continue
        match = re.search(
            r'(?P<method>INPERSON|MAIL|ONLINE|FAX|FWAB|EARLY|QUESTIONED)\s+(?P<accept_full>\d+)\s+(?P<accept_partial>\d+)\s+(?P<reject>\d+)\s+(?P<apps_rcvd>\d+)\s+(?P<ballot_sent>\d+)\s+(?P<ballot_rcvd>\d+)',
            line)

        if match and current_house and current_house != "99":  # Exclude house 99
            data_entry = {
                'ACCEPT_FULL': int(match.group('accept_full')),
                'ACCEPT_PARTIAL': int(match.group('accept_partial')),
                'REJECT': int(match.group('reject')),
                'APPS_RCVD': int(match.group('apps_rcvd')),
                'BALLOT_SENT': int(match.group('ballot_sent')),
                'BALLOT_RCVD': int(match.group('ballot_rcvd')),
                'HOUSE': current_house,  # Include the house number
                'VOTING_METHOD': match.group('method')  # Store the voting method for reference
            }
            tabulated_data.append(data_entry)

    return tabulated_data

def clean_data(df):
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['HOUSE'] = df['HOUSE'].astype(str)

    print("\nCleaned DataFrame summary:")
    print(df.info())

    return df

# Main execution
pdf_path = r'C:\Users\mithi\electionprediction\Combined Ballot Count Report_9.2.2022.pdf'  # Change to your actual PDF file path
raw_data = extract_raw_data(pdf_path)
print("Raw data extracted from PDF:")
for line in raw_data:
    print(line)

tabulated_data = tabulate_data(raw_data)
df = pd.DataFrame(tabulated_data)

df = clean_data(df)

print("\nTabulated and Cleaned DataFrame representation:")
print(df)

# Save the cleaned DataFrame to a CSV file
output_csv_path = r'C:\Users\mithi\electionprediction\tabulated_data.csv'  # Specify your desired output CSV file path
df.to_csv(output_csv_path, index=False)

print(f"\nData has been saved to {output_csv_path}")

# Calculate Total Accepted Votes (Full + Partial) and sum by House
df['TOTAL_ACCEPTED'] = df['ACCEPT_FULL'].astype(int) + df['ACCEPT_PARTIAL'].astype(int)

# Group by HOUSE and sum the total accepted votes
total_accepted_by_house = df.groupby('HOUSE')['TOTAL_ACCEPTED'].sum().reset_index()

# Print the total accepted votes by house for verification
print("\nTotal Accepted Votes by House:")
print(total_accepted_by_house)

# Plotting Total Accepted Votes by House
plt.figure(figsize=(12, 6))
plt.bar(total_accepted_by_house['HOUSE'], total_accepted_by_house['TOTAL_ACCEPTED'], color='purple')
plt.title('Total Accepted Votes by House')
plt.xlabel('House')
plt.ylabel('Total Accepted Votes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plot_data = df.groupby('VOTING_METHOD')[['ACCEPT_FULL', 'REJECT']].sum().reset_index()

bar_width = 0.35
x = range(len(plot_data['VOTING_METHOD']))

plt.bar(x, plot_data['ACCEPT_FULL'], width=bar_width, color='green', label='Accepted', align='center')
plt.bar([p + bar_width for p in x], plot_data['REJECT'], width=bar_width, color='red', label='Rejected', align='center')

plt.title('Accepted and Rejected Ballots by Voting Method')
plt.xlabel('Voting Method')
plt.ylabel('Number of Ballots')
plt.xticks([p + bar_width / 2 for p in x], plot_data['VOTING_METHOD'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
