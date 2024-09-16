import pandas as pd

# Load the CSV file
predictions_df = pd.read_csv('test_predictions_self_cnn.csv')

# Clean the 'id' column by removing 'tensor(...)' and converting to integers
predictions_df['id'] = predictions_df['id'].apply(lambda x: int(x.strip('tensor()')))

# Save the cleaned DataFrame back to CSV
predictions_df.to_csv('test_predictions_cleaned_self_cnn.csv', index=False)

print("Cleaned predictions saved to test_predictions_cleaned.csv")
