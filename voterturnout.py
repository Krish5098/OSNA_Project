import pdfplumber
import pandas as pd
import re
import matplotlib.pyplot as plt


def extract_voter_data(pdf_path):
    """
    Extracts text from all pages in a PDF file.
    """
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    return all_text


def format_text_data(text_data):
    """
    Formats text data to replace specific phrases with underscores.
    """
    text_data = text_data.replace("UNKNOWN GENDER", "UNKNOWN_GENDER")
    text_data = text_data.replace("UNKNOWN DOB", "UNKNOWN_DOB")
    text_data = re.sub(r"(\d+)\s+THRU\s+(\d+)", r"\1_THRU_\2", text_data)
    text_data = re.sub(r"(\d+)\s+(ABOVE|BELOW)", r"\1_\2".lower(), text_data)

    return text_data


def save_raw_data_to_csv(text_data, output_csv_path):
    """
    Saves the formatted text data to a CSV file, splitting each line into separate columns based on whitespace.
    Keeps specific phrases together in single cells.
    """
    text_data = format_text_data(text_data)
    lines = text_data.splitlines()

    data = []

    for line in lines:
 
        row = re.findall(r'(\d+_THRU_\d+|[\w]+_above|[\w]+_below|UNKNOWN_GENDER|\S+)', line)
        data.append(row)  

    df = pd.DataFrame(data) 
    df.to_csv(output_csv_path, index=False, header=False, encoding='utf-8')
    print(f"\nFormatted data saved to {output_csv_path}")


def separate_age_gender(input_csv_path, output_csv_path):
    """
    Reads the input CSV file, separates age and gender into different columns,
    appends them to the remaining data, and saves the output.
    """
    df = pd.read_csv(input_csv_path, header=None) 
    separated_data = []
    current_age_group = None 

    for index, row in df.iterrows():
        line = row[0]
        age_gender_matches = re.findall(r'(\d+_THRU_\d+|\d+|UNKNOWN_GENDER|MALE|FEMALE|UNKNOWN|VOTED)', line)

        if age_gender_matches:
            for match in age_gender_matches:
                if re.match(r'\d+_THRU_\d+', match) or re.match(r'\d+', match):
                    current_age_group = match
                elif match in ['MALE', 'FEMALE', 'UNKNOWN', 'VOTED']:
                    separated_data.append([current_age_group if current_age_group else '', match] + list(row[1:]))

        else:
            continue

    columns = ['AGE_GROUP', 'GENDER'] + [f'COLUMN_{i + 3}' for i in range(len(row) - 1)]

    new_df = pd.DataFrame(separated_data, columns=columns)
    new_df.to_csv(output_csv_path, index=False)
    print(f"\nSeparated age and gender data appended with remaining data saved to {output_csv_path}")


def plot_voter_data(csv_path):
    """
    Reads the CSV file and plots the number of voters by age group and gender.
    """
    df = pd.read_csv(csv_path)
    df_voted = df[df['GENDER'].isin(['MALE', 'FEMALE'])]  # Consider only Male and Female
    counts = df_voted.groupby(['AGE_GROUP', 'GENDER']).size().unstack(fill_value=0)

    # Plotting
    counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['blue', 'orange'])

    plt.title('Number of Voters by Age Group and Gender')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Voters')
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.show()


def plot_party_age_data(csv_path):

    df = pd.read_csv(csv_path)
    party_columns = df.columns[2:]
    df_parties = df[party_columns]
    counts_party = df_parties.groupby(df['AGE_GROUP']).sum() 

    counts_party.plot(kind='bar', stacked=True, figsize=(10, 6))

    plt.title('Number of Voters by Age Group and Party Affiliation')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Voters')
    plt.xticks(rotation=45) 
    plt.legend(title='Party Affiliation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.show()


# Main execution
pdf_path = r'C:\Users\mithi\electionprediction\VoterHistoryByAgeReport.pdf'  
output_csv_path = r'C:\Users\mithi\electionprediction\EntireFormattedVoterData.csv'

text_data = extract_voter_data(pdf_path)
if text_data:
    print("Raw Data:")
    print(text_data)

    save_raw_data_to_csv(text_data, output_csv_path)
    separated_output_path = r'C:\Users\mithi\electionprediction\SeparatedVoterData.csv'
    separate_age_gender(output_csv_path, separated_output_path)

    plot_voter_data(separated_output_path)

    plot_party_age_data(separated_output_path)
else:
    print("No text found in the PDF.")
