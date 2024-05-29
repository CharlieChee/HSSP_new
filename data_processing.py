# import pandas as pd
# from io import StringIO
#
# # Load the text content from the uploaded file
# file_path = 'attack_results.txt'
#
# # Read the file content
# with open(file_path, 'r') as file:
#     content = file.read()

# Since the file content is structured with colons and commas, we'll parse it accordingly.
# First, we split the text into lines
# lines = content.strip().split('\n')
#
# # Then we parse each line into a dictionary
# data = []
# for line in lines:
#     # Split the line by commas first
#     parts = line.split(', ')
#     # Create a dictionary for each part splitting by the first colon
#     entry = {part.split(': ', 1)[0]: part.split(': ', 1)[1] for part in parts}
#     data.append(entry)
#
# # Convert the list of dictionaries to a DataFrame
# df = pd.DataFrame(data)
#
# # Saving the DataFrame to an Excel file
# excel_path = 'attack_results.xlsx'
# df.to_excel(excel_path, index=False)


import pandas as pd
from PIL import Image
import pytesseract

file_name = f"attack_results_round_Rand_100"

file_path = f'./comparisonData/data_new/{file_name}.txt'
file_path = '/Users/jichanglong/Desktop/hssp_new/comparisonData/data_m_times/docu.txt'

excel_path = f'./comparisonData/data_new/excel/{file_name}.xlsx'

excel_path = '/Users/jichanglong/Desktop/hssp_new/comparisonData/data_m_times/docu2.xlsx'
# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Since the file content is structured with colons and commas, we'll parse it accordingly.
# First, we split the text into lines
lines = content.strip().split('\n')

# Then we parse each line into a dictionary
data = []
for line in lines:
    # Skip empty lines
    if not line.strip():
        continue

    # Split the line by commas first
    parts = line.split(', ')

    # Create a dictionary for each part, splitting by the first colon
    entry = {}
    for part in parts:
        key, value = part.split(': ', 1)

        # If 'Error' is not '0', set 'Time', 'NFound', and 'Coef' as None
        if key == 'Error' and value != '0':
            entry['Time'] = None
            entry['NFound'] = None
            entry['Coef'] = None

        entry[key] = value if key not in ['Time', 'NFound', 'Coef'] else entry.get(key, value)

    data.append(entry)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Saving the DataFrame to an Excel file

df.to_excel(excel_path, index=False)

