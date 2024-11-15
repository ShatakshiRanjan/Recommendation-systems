import zipfile
import pandas as pd

# Correct path to the zip file (use raw string to handle backslashes in Windows)
zip_file_path = r'ml-latest-small.zip'  # Ensure this path points to the actual .zip file

# Open the .zip file
with zipfile.ZipFile(zip_file_path, 'r') as z:
    # List all files in the zip archive
    print("Files in the zip archive:")
    print(z.namelist())  # Check the file names inside the .zip archive
    
    # Read specific files into pandas DataFrames
    links_df = pd.read_csv(z.open('links.csv'))
    movies_df = pd.read_csv(z.open('movies.csv'))
    ratings_df = pd.read_csv(z.open('ratings.csv'))
    tags_df = pd.read_csv(z.open('tags.csv'))

# Display the first few rows of one DataFrame (optional)
print("Links:")
print(links_df.head())

