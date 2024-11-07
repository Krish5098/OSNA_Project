import pdfplumber
import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract raw data from the first page of the PDF
def extract_raw_data(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        raw_data = []
        # Only extract from the first page
        if len(pdf.pages) > 0:
            page = pdf.pages[0]
            text = page.extract_text()
            if text:
                # Split the text into lines and add to raw data
                lines = text.split('\n')
                raw_data.extend(lines)
    return raw_data

# Function to tabulate the extracted data
def tabulate_data(raw_data):
    tabulated_data = []

    # Process lines that contain voting data
    for line in raw_data:
        # Use regex to match the expected line format
        match = re.search(
            r'^(?P<candidate>.+?)\s+(?P<party>[A-Z]{3})\s+(?P<votes>[\d,]+)\s+(?P<percentage>[\d.]+%)$',
            line.strip())

        if match:
            data_entry = {
                'CANDIDATE': match.group('candidate').strip(),
                'PARTY': match.group('party'),
                'VOTES': int(match.group('votes').replace(',', '')),  # Remove commas for integer conversion
                'PERCENTAGE': match.group('percentage')
            }
            tabulated_data.append(data_entry)

    return tabulated_data

# Main execution
pdf_path = r'C:\Users\mithi\electionprediction\ElectionSummaryReportRPT.pdf'  # Change to your actual PDF file path
raw_data = extract_raw_data(pdf_path)

# Print raw data for verification
print("Raw data extracted from the first page of PDF:")
for line in raw_data:
    print(line)

# Clean and tabulate the extracted data
tabulated_data = tabulate_data(raw_data)

# Print the tabulated data
print("\nTabulated Data:")
for entry in tabulated_data:
    print(entry)

# Save the tabulated data to a CSV file
output_csv_path = r'C:\Users\mithi\electionprediction\complete_data.csv'  # Specify your desired output CSV file path
df = pd.DataFrame(tabulated_data)
df.to_csv(output_csv_path, index=False)

print(f"\nData has been saved to {output_csv_path}")

# Convert percentage to numeric values for plotting
df['PERCENTAGE'] = df['PERCENTAGE'].str.replace('%', '').astype(float)

# Plotting the candidates and their vote percentages
plt.figure(figsize=(12, 6))
plt.bar(df['CANDIDATE'], df['PERCENTAGE'], color='skyblue')
plt.title('Vote Percentage by Candidate')
plt.xlabel('Candidates')
plt.ylabel('Vote Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
