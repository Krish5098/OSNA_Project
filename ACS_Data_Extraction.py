# -*- coding: utf-8 -*-
"""CensusData1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jBfpQdoURHqi33tV4aWLAZanHb7lUwln

# Extract Decennial Census Data
"""

!pip3 install requests

import requests
import pandas as pd

def base_url_func(year = "2000", dataset = "dec/sf1"):
    # Census API host name
    HOST = "https://api.census.gov/data"
    base_url = "/".join([HOST, year, dataset])

    return base_url

base_url = base_url_func()
print(f"Census Gov Data Base URL: {base_url}")

# Specify Census variables and other predicates
get_vars = ["NAME", "P001001"]
predicates = {}
predicates["get"] = ",".join(get_vars)
predicates["for"] = "county:*"
predicates["in"] = "state:02"

# Execute the request, examine text of response object
r = requests.get(base_url, params=predicates)

"""##### https://census.missouri.edu/geocodes/

![image.png](attachment:image.png)
"""

r.text

population_df = pd.DataFrame(data = r.json()[1:], columns = ["NAME", "POPULATION", "STATE_CODE","COUNTY"])

population_df

"""# Exploration 2020 DataSet"""

def base_url_func(year = "2010", dataset = "dec/sf1"):
    # Census API host name
    HOST = "https://api.census.gov/data"
    base_url = "/".join([HOST, year, dataset])

    return base_url

base_url = base_url_func(year = "2020", dataset = "dec/ddhca")
print(f"Census Gov Data Base URL: {base_url}")

# Specify Census variables and other predicates
get_vars = ["NAME","POPGROUP","T01001_001N","T02001_002N","T02001_007N"]
predicates = {}
predicates["get"] = ",".join(get_vars)
predicates["for"] = "county:*"
predicates["in"] = "state:02"
predicates["key"] = "1e4b8c1715da9e2f6f845ff970bcec86e22efd86"

# Execute the request, examine text of response object
r = requests.get(base_url, params=predicates)
print(base_url)
print(f"Status Code:{r}")

r.json()

data = pd.DataFrame(data = r.json()[1:], columns =
                    ['NAME', 'POPGROUP', 'Total Population','Male Population','Female Population','state', 'county'])

data

base_url = base_url_func(year = "2020", dataset = "dec/ddhcb")
print(f"Census Gov Data Base URL: {base_url}")

predicates = {
    "get":"NAME,POPGROUP,T03001_001N,T03002_002N,T03002_003N",
    "ucgid": "pseudo(0400000US02$0500000)",
    "key": "1e4b8c1715da9e2f6f845ff970bcec86e22efd86"
}
# Execute the request
r = requests.get(base_url, params=predicates)

# Output results
print(f"Status Code: {r.status_code}")
print("Response Text:", r.text)

data = pd.DataFrame(data = r.json()[1:], columns =
                    ['NAME', 'POPGROUP','Household type total','family household type','non-family household type','ucgid'])
data = data.drop(columns=['ucgid'])
data

"""Exploring 2023 dataset

2023 acs data: Population by sex
"""

base_url = base_url_func(year = "2023", dataset = "acs/acsse")
print(f"Census Gov Data Base URL: {base_url}")

# Define parameters
predicates = {
    "get": "GEO_ID,NAME,K200101_001E,K200101_002E,K200101_003E,K200201_001E,K200201_002E,K200201_003E,K200201_004E,K200201_005E,K200201_006E,K200201_007E,K200201_008E,K201501_001E,K201501_002E,K201501_003E,K201501_004E,K201501_005E,K201501_006E,K201501_007E,K201501_008E,K201901_001E,K201901_002E,K201901_003E,K201901_004E,K201901_005E,K201901_006E,K201901_007E,K201901_008E",
    "ucgid": "pseudo(0400000US02$0500000)",
    "key": "1e4b8c1715da9e2f6f845ff970bcec86e22efd86"
}
# Execute the request
r = requests.get(base_url, params=predicates)

# Output results
print(f"Status Code: {r.status_code}")
print("Response Text:", r.text)

data = pd.DataFrame(data = r.json()[1:], columns =
                    ['GEO_ID',
  'NAME',
  'Population total Estimate',
  'Population Male estimate',
  'Population Female estimate','Race total estimate','white alone estimate','Black or African American alone estimate', 'American Indian and alaska native alone', 'Asian alone estimate','Native Hawaiian and Other Pacific Islander alone estimate', 'Some other race alone estimate', 'Two or more races estimate','education estimate','less than 9th grade','9th to 12th no diploma','high school graduate','some college no degree', 'Associate degree','bachelors degree','graduate or professional degree',
   'total household income estimate','less than $20,000 estimate','$20,000 to $39,000 estimate','$40,000 to $59,000 estimate','$60,000 to $99,999 estimate','$100,000 to $149000 estimate','$150000 to $199,999 estimate','$200,000 or more estimate',
  'ucgid'])
data = data.drop(columns=['GEO_ID','ucgid'])

data

data['Population by Sex total Estimate'] = pd.to_numeric(data['Population by Sex total Estimate'], errors='coerce')

# Drop any rows with NaN in the 'Population by Sex total Estimate' column (optional)
data = data.dropna(subset=['Population by Sex total Estimate'])

# Plot the histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['Population by Sex total Estimate'], bins=30, kde=True)  # kde=True adds a density estimate
plt.title('Population Distribution (Histogram)')
plt.xlabel('Total Population')
plt.ylabel('Frequency')
plt.show()

# Convert columns to numeric if needed
data['Population Male estimate'] = pd.to_numeric(data['Population Male estimate'], errors='coerce')
data['Population Female estimate'] = pd.to_numeric(data['Population Female estimate'], errors='coerce')

# Sum male and female populations across all counties
total_male = data['Population Male estimate'].sum()
total_female = data['Population Female estimate'].sum()

# Bar chart
plt.figure(figsize=(6, 4))
plt.bar(['Male', 'Female'], [total_male, total_female], color=['blue', 'pink'])
plt.title('Total Male and Female Population')
plt.ylabel('Population')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the DataFrame 'data' is already defined as per your structure

# Convert population columns to numeric, if necessary
data['Population Male estimate'] = pd.to_numeric(data['Population Male estimate'], errors='coerce')
data['Population Female estimate'] = pd.to_numeric(data['Population Female estimate'], errors='coerce')

# Drop any rows with NaN in the population columns
data = data.dropna(subset=['Population Male estimate', 'Population Female estimate'])

# Melt the DataFrame for easier plotting
data_melted = data.melt(id_vars=['NAME'],
                         value_vars=['Population Male estimate', 'Population Female estimate'],
                         var_name='Gender',
                         value_name='Population')

# Set the figure size
plt.figure(figsize=(12, 8))

# Create the bar plot
sns.barplot(x='NAME', y='Population', hue='Gender', data=data_melted)

# Customize the plot
plt.title('Male and Female Population by County')
plt.xlabel('County Name')
plt.ylabel('Population')
plt.xticks(rotation=90)  # Rotate county names for better visibility
plt.legend(title='Gender')

# Show the plot
plt.tight_layout()
plt.show()

# Define the columns related to education attainment
education_columns = [
    'less than 9th grade',
    '9th to 12th no diploma',
    'high school graduate',
    'some college no degree',
    'Associate degree',
    'bachelors degree',
    'graduate or professional degree'
]

# Convert education attainment columns to numeric, if necessary
for col in education_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Sum the counts for each education level
education_summary = data[education_columns].sum().reset_index()
education_summary.columns = ['Degree', 'Count']

# Set the figure size
plt.figure(figsize=(12, 6))

# Create the bar plot
sns.barplot(x='Degree', y='Count', data=education_summary, palette='viridis')

# Customize the plot
plt.title('Education Attainment by Degree')
plt.xlabel('Degree')
plt.ylabel('Count of People')
plt.xticks(rotation=45)  # Rotate degree names for better visibility

# Show the plot
plt.tight_layout()
plt.show()

data_2023 = data

"""# **ACS 5 year estimate**"""

base_url = base_url_func(year = "2022", dataset = "acs/acs5/profile")
print(f"Census Gov Data Base URL: {base_url}")

# Define parameters
predicates = {
    "get": "GEO_ID,NAME,DP02_0059E,DP02_0060E,DP02_0061E,DP02_0062E,DP02_0063E,DP02_0064E,DP02_0065E,DP02_0066E,DP02_0067E,DP02_0068E,DP03_0051E,DP03_0052E,DP03_0053E,DP03_0054E,DP03_0055E,DP03_0056E,DP03_0057E,DP03_0058E,DP03_0059E,DP03_0060E,DP03_0061E",
    "for": "COUNTY:*",
    "in":"state:02",
    "ucgid": "pseudo(0400000US02$0500000)",
    "key": "1e4b8c1715da9e2f6f845ff970bcec86e22efd86"
}
# Execute the request
r = requests.get(base_url, params=predicates)

# Output results
print(f"Status Code: {r.status_code}")
print("Response Text:", r.text)

base_url = base_url_func(year = "2022", dataset = "acs/acs5/subject")
print(f"Census Gov Data Base URL: {base_url}")

# Define parameters
predicates = {
    "get": "GEO_ID,NAME,S0101_C01_001E,S0101_C03_001E,S0101_C05_001E",
    "for": "COUNTY:*",
    "in":"state:02",
    "ucgid": "pseudo(0400000US02$0500000)",
    "key": "1e4b8c1715da9e2f6f845ff970bcec86e22efd86"
}
# Execute the request
r1 = requests.get(base_url, params=predicates)

# Output results
print(f"Status Code: {r.status_code}")
print("Response Text:", r.text)

data1 = pd.DataFrame(data = r1.json()[1:], columns =
                    ['GEO_ID','NAME','Population total Estimate',
  'Population Male estimate',
  'Population Female estimate',
  'ucgid'])
data1 = data1.drop(columns=['GEO_ID','ucgid'])
data1

"""Population by Sex total Estimate',
  'Population Male estimate',
  'Population Female estimate','Race total estimate','white alone estimate','Black or African American alone estimate', 'American Indian and alaska native alone', 'Asian alone estimate','Native Hawaiian and Other Pacific Islander alone estimate', 'Some other race alone estimate', 'Two or more races estimate',"""

data = pd.DataFrame(data = r.json()[1:], columns =
                    ['GEO_ID',
  'NAME','education estimate','less than 9th grade','9th to 12th no diploma','high school graduate','some college no degree', 'Associate degree','bachelors degree','graduate or professional degree',
  'high school graduate or higher','bachelors degree or higher','total household income estimate','less than $10000 income estimate','$10,000 to $14999 estimate','$15,000 to $24999 estimate','$25000 to $34999 estimate','$35,000 to $49,999 estimate','$50,000 to $74999 estimate','$75000 to $99,999 estimate','$100,000 to $ 149,999 estimate','$150,000 to $200,000 estimate','$200,000 or higher estimate','ucgid'])
data = data.drop(columns=['GEO_ID','ucgid'])
data

#data = pd.concat([data1, data], axis=1, ignore_index=True)
# Stack rows
# Drop duplicated columns after concatenation
data_2022 = pd.concat([data1, data], axis=1)
data_2022 = data_2022.loc[:, ~data_2022.columns.duplicated()]

data_2022

data.to_csv('ACS_2022_5year_estimate.csv')

import pandas as pd
import matplotlib.pyplot as plt

# Assuming data_2022 and data_2023 are your DataFrames containing the relevant population data
# Make sure to rename your population columns if they differ

# Example population columns names
# For the sake of this example, let's say:
# data_2022 contains 'Population total Estimate'
# data_2023 contains 'Population by Sex total Estimate'

# Merging the two DataFrames on the 'NAME' column
merged_data = pd.merge(data_2022, data_2023, on='NAME', suffixes=('_2022', '_2023'))

# Calculating the population difference
merged_data['Population Difference'] = merged_data['Population total Estimate'] - merged_data['Population total Estimate']

# Plotting the results
plt.figure(figsize=(14, 6))
merged_data.plot(kind='bar', x='NAME', y='Population Difference', color='skyblue')
plt.title('Population Difference Between 2022 and 2023 by County')
plt.xlabel('County')
plt.ylabel('Population Difference')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

