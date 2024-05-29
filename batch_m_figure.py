import pandas as pd


excel_file_path = '/Users/jichanglong/Desktop/hssp_new/comparisonData/data_m_times/docu.xlsx'


df_second_sheet = pd.read_excel(excel_file_path)


ave_time = df_second_sheet.iloc[:, 3].tolist()

print(ave_time)