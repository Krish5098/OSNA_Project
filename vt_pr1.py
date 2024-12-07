import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from ballot_total_acceptence import calculate_predicted_acceptance_rate,calculate_acceptance_rate,calculate_total_accepted

# Load data
data_train1 = pd.read_csv(r"C:\Users\mithi\electionprediction\data\voter_turnout2016P.csv")
data_train2 = pd.read_csv(r"C:\Users\mithi\electionprediction\data\voter_turnout2018P.csv")
data_test = pd.read_csv(r"C:\Users\mithi\electionprediction\data\voter_turnout2020P.csv")

# Combine training data
data_train = pd.concat([data_train1, data_train2], ignore_index=True)

# Define party-related columns (votes for each party)
party_columns = ['Av', 'Cv', 'Dv', 'Ev', 'Gv', 'Lv', 'Nv', 'Ov', 'Pv', 'Rv', 'Uv']

# Define the target variable: total voter turnout
data_train['Total_voted'] = data_train[party_columns].sum(axis=1)

# Separate features and target variable for training
X_train = data_train[party_columns]  # Use party-related columns as features
y_train = data_train['Total_voted']

# Use the same features in the test data
X_test = data_test[party_columns]

# Align columns between train and test data to handle any potential mismatches
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=np.nan)

# Create a pipeline with imputation, scaling, and Gradient Boosting Regressor
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),  # Fill missing values with mean
    StandardScaler(),  # Scale features
    GradientBoostingRegressor()  # Gradient Boosting for predictions
)

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Predict voter turnout for the test dataset
data_test['predicted_voter_turnout'] = pipeline.predict(X_test)

# Identify the most preferred party for each age group in the test data
# Find the party with the maximum votes for each row
data_test['predicted_most_preferred_party'] = data_test[party_columns].idxmax(axis=1)

# Save the updated test data with predictions to a CSV file
output_path = r"C:\Users\mithi\electionprediction\predicted_voter_turnout.csv"
data_test.to_csv(output_path, index=False)

# Load the updated test data
data_test = pd.read_csv(output_path)

# Calculate overall most preferred party
overall_most_preferred_party = data_test[party_columns].sum().idxmax()

# Calculate predicted voter turnout percentage
predicted_total_turnout = data_test['predicted_voter_turnout'].sum()
actual_total_turnout = data_test[party_columns].sum().sum()
total_population = data_test['TOTAL'].sum()

predicted_turnout_percentage = (predicted_total_turnout / total_population) * 100
actual_turnout_percentage = (actual_total_turnout / total_population) * 100

# Display results
print(f"Overall Most Preferred Party: {overall_most_preferred_party}")
print(f"Predicted Voter Turnout Percentage: {predicted_turnout_percentage:.2f}%")
print(f"Actual Voter Turnout Percentage: {actual_turnout_percentage:.2f}%")


# Plot voter turnout percentage by age group
data_test['Total_Votes'] = data_test[party_columns].sum(axis=1)
data_test['Voter_Turnout_Percentage'] = (data_test['Total_Votes'] / data_test['TOTAL']) * 100
age_turnout_percentage = data_test.groupby('AGE_GROUP')['Voter_Turnout_Percentage'].mean()

plt.figure(figsize=(12, 8))
age_turnout_percentage.plot(kind='bar', color='skyblue')
plt.title('Voter Turnout Percentage by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Voter Turnout Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ROC Curve and AUC
threshold = 100
data_test['predicted_turnout_binary'] = (data_test['predicted_voter_turnout'] > threshold).astype(int)
data_test['actual_turnout_binary'] = (data_test[party_columns].sum(axis=1) > threshold).astype(int)

y_true = data_test['actual_turnout_binary']
y_pred = data_test['predicted_turnout_binary']

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Voter Turnout Prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np

# File paths
file_paths = [
    r"C:\Users\mithi\electionprediction\csv_outputs\18GenCombineFilingReport2018G_processed.csv",
    r"C:\Users\mithi\electionprediction\csv_outputs\2016GenCombineFilingReport2016G_processed.csv",
    r"C:\Users\mithi\electionprediction\csv_outputs\CombinedBallotCountReport_Server2020G_processed.csv"
]

# Step 1: Load Data
data_2018 = pd.read_csv(file_paths[0])
data_2016 = pd.read_csv(file_paths[1])
data_2020 = pd.read_csv(file_paths[2])

# Step 2: Inspect Data
print("2018 Data Columns:", data_2018.columns)
print("2016 Data Columns:", data_2016.columns)
print("2020 Data Columns:", data_2020.columns)

# Step 3: Clean Data
# Remove rows with missing values and drop duplicates
def clean_data(data):
    data = data.dropna()
    data = data.drop_duplicates()
    return data

data_2018_clean = clean_data(data_2018)
data_2016_clean = clean_data(data_2016)
data_2020_clean = clean_data(data_2020)

# Step 4: Feature Selection (Ensure we're using relevant columns for Total Accepted Votes)
# Assuming columns similar to 'ACCEPTED FULL', 'ACCEPTED PARTIAL', and 'REJECT' are available in each dataset
columns_to_use = ['ACCEPT_FULL', 'ACCEPT_PARTIAL', 'REJECT']

# Calculate Total Accepted Votes by adding Full and Partial Accepted Votes
def calculate_total_accepted(data, columns=columns_to_use):
    data['TOTAL_ACCEPTED'] = data['ACCEPT_FULL'] + data['ACCEPT_PARTIAL']
    return data

# Apply calculation on each dataset
data_2018_clean = calculate_total_accepted(data_2018_clean)
data_2016_clean = calculate_total_accepted(data_2016_clean)
data_2020_clean = calculate_total_accepted(data_2020_clean)

# Step 5: Combine Data (if needed, combining by year or other groupings)
combined_data = pd.concat([data_2018_clean, data_2016_clean, data_2020_clean], axis=0)

# Step 6: Model Development (Optional)
# We can create a simple model if there's a need to predict Total Accepted Votes from other features
from sklearn.linear_model import LinearRegression

# Feature Engineering - Let's assume we have 'HOUSE' as a feature and 'TOTAL_ACCEPTED' as the target
X = combined_data[['HOUSE']]  # Features (add more if needed)
y = combined_data['TOTAL_ACCEPTED']  # Target variable

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluation
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 8: Save processed data with Total Accepted Votes
combined_data.to_csv(r'C:\Users\mithi\electionprediction\csv_outputs\combined_data_with_total_accepted.csv', index=False)

print("\nProcessed data with Total Accepted Votes saved.")
# Calculate Total Acceptance Rate
# Total Acceptance Rate = (Total Accepted Votes / (Total Accepted + Total Rejected)) * 100
def calculate_acceptance_rate(data):
    total_accepted = data['TOTAL_ACCEPTED'].sum()
    total_rejected = data['REJECT'].sum()
    total_votes = total_accepted + total_rejected
    acceptance_rate = (total_accepted / total_votes) * 100 if total_votes > 0 else 0
    return acceptance_rate

# Calculate acceptance rate for the combined data
total_acceptance_rate = calculate_acceptance_rate(combined_data)
print(f"Total Acceptance Rate (All Data Combined): {total_acceptance_rate:.2f}%")

# Calculate Predicted Total Acceptance Rate for all houses combined
# Sum predicted accepted votes and calculate the predicted acceptance rate
def calculate_predicted_acceptance_rate(X_test, y_pred, data):
    # Map predictions back to original data
    predicted_total_accepted = pd.DataFrame({'HOUSE': X_test['HOUSE'], 'PREDICTED_ACCEPTED': y_pred})
    predicted_total = predicted_total_accepted.groupby('HOUSE')['PREDICTED_ACCEPTED'].sum().sum()
    actual_rejected = data['REJECT'].sum()
    predicted_acceptance_rate = (predicted_total / (predicted_total + actual_rejected)) * 100 if (predicted_total + actual_rejected) > 0 else 0
    return predicted_acceptance_rate

# Calculate the predicted acceptance rate for combined data
predicted_acceptance_rate = calculate_predicted_acceptance_rate(X_test, y_pred, combined_data)

new_voter_turnout_actual = (actual_total_turnout * (total_acceptance_rate / 100)) / total_population * 100

new_voter_turnout_predicted = (predicted_total_turnout * (predicted_acceptance_rate / 100)) / total_population * 100

print(f"Predicted Acceptance Rate: {predicted_acceptance_rate:.2f}%")
print(f"New Voter Turnout Prediction (with acceptance rate): {new_voter_turnout_predicted:.2f}%")
print(f"New Voter Turnout: {new_voter_turnout_actual:.2f}%")