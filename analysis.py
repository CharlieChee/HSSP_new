import pandas as pd



# df = pd.read_excel('/Users/jichanglong/Desktop/hssp_new/comparisonData/data_new/excel/raw_data_all.xlsx')
#
# ns = 'NS Attack'
# multi = 'Multi Attack'
# stat = 'Statistical Attack'
#
#
# # filtered_df = df[
# #     (df['Attack Type'] == multi) &
# #     (df['R_method'] == 'total_random') &
# #     (df['sampling_number'] == 100)
# # ]
# # output1 = filtered_df['NFound'].mean()
# # output2 = filtered_df['Coef'].mean()
# # print(output1,output2)
#
# filtered_df = df[
#     (df['Attack Type'] == multi
#      ) &
#     (df['R_method'] == 'total_random') &
#     (df['sampling_number'] == 400) &
#     (df['Error'] == 0)
# ]
# output = filtered_df['Time'].mean()
# print(output)



import pandas as pd


excel_file_path = '/Users/jichanglong/Desktop/hssp_new/comparisonData/data_new/excel/raw_data_all.xlsx'


df_second_sheet = pd.read_excel(excel_file_path, sheet_name=1)


ave_time = df_second_sheet.iloc[:, 3].tolist()

ave_NFound = df_second_sheet.iloc[:, 4].tolist()

ave_Coef = df_second_sheet.iloc[:, 5].tolist()
import matplotlib.pyplot as plt
x_coordinates = [100, 200, 300, 400]



time_G_ns = ave_time[:4]
time_G_multi = ave_time[4:8]
time_G_stat = ave_time[8:12]

time_R_sub_ns = ave_time[12:16]
time_R_sub_multi = ave_time[16:20]
time_R_sub_stat = ave_time[20:24]

time_R_total_ns = ave_time[24:28]
time_R_total_multi = ave_time[28:32]
time_R_total_stat = ave_time[32:36]


#
# # Plotting the lines
# plt.figure(figsize=(10, 8))
#
# # Each plot call plots one of the lines with the given x and y values
# plt.plot(x_coordinates, time_G_ns, label='time_G_ns')
# plt.plot(x_coordinates, time_G_multi, label='time_G_multi')
# plt.plot(x_coordinates, time_G_stat, label='time_G_stat')
# plt.plot(x_coordinates, time_R_sub_ns, label='time_R_sub_ns')
# plt.plot(x_coordinates, time_R_sub_multi, label='time_R_sub_multi')
# plt.plot(x_coordinates, time_R_sub_stat, label='time_R_sub_stat')
# plt.plot(x_coordinates, time_R_total_ns, label='time_R_total_ns')
# plt.plot(x_coordinates, time_R_total_multi, label='time_R_total_multi')
# plt.plot(x_coordinates, time_R_total_stat, label='time_R_total_stat')
#
# # Adding titles and labels
# # plt.title('Line Plot of Times')
# plt.xlabel('Sampling number')
# plt.ylabel('Average Time')
#
# # Adding a legend to distinguish the lines
# plt.legend()
#
# # Display the plot
# plt.show()



NFound_G_ns = ave_NFound[:4]
NFound_G_multi = ave_NFound[4:8]
NFound_G_stat = ave_NFound[8:12]

NFound_R_sub_ns = ave_NFound[12:16]
NFound_R_sub_multi = ave_NFound[16:20]
NFound_R_sub_stat = ave_NFound[20:24]

NFound_R_total_ns = ave_NFound[24:28]
NFound_R_total_multi = ave_NFound[28:32]
NFound_R_total_stat = ave_NFound[32:36]

#
# # Plotting the lines
# plt.figure(figsize=(10, 8))
#
# # Each plot call plots one of the lines with the given x and y values
# plt.plot(x_coordinates, NFound_G_ns, label='NFound_G_ns')
# plt.plot(x_coordinates, NFound_G_multi, label='NFound_G_multi')
# plt.plot(x_coordinates, NFound_G_stat, label='NFound_G_stat')
# plt.plot(x_coordinates, NFound_R_sub_ns, label='NFound_R_sub_ns')
# plt.plot(x_coordinates, NFound_R_sub_multi, label='NFound_R_sub_multi')
# plt.plot(x_coordinates, NFound_R_sub_stat, label='NFound_R_sub_stat')
# plt.plot(x_coordinates, NFound_R_total_ns, label='NFound_R_total_ns')
# plt.plot(x_coordinates, NFound_R_total_multi, label='NFound_R_total_multi')
# plt.plot(x_coordinates, NFound_R_total_stat, label='NFound_R_total_stat')
#
# # Adding titles and labels
# plt.title('Success rate of NFound')
# plt.xlabel('Sampling number')
# plt.ylabel('Success rate')
#
# # Adding a legend to distinguish the lines
# plt.legend()
#
# # Display the plot
# plt.show()


Coef_G_ns = ave_Coef[:4]
Coef_G_multi = ave_Coef[4:8]
Coef_G_stat = ave_Coef[8:12]

Coef_R_sub_ns = ave_Coef[12:16]
Coef_R_sub_multi = ave_Coef[16:20]
Coef_R_sub_stat = ave_Coef[20:24]

Coef_R_total_ns = ave_Coef[24:28]
Coef_R_total_multi = ave_Coef[28:32]
Coef_R_total_stat = ave_Coef[32:36]


# Plotting the lines
plt.figure(figsize=(10, 8))

# Each plot call plots one of the lines with the given x and y values
plt.plot(x_coordinates, Coef_G_ns, label='Coef_G_ns')
plt.plot(x_coordinates, Coef_G_multi, label='Coef_G_multi')
plt.plot(x_coordinates, Coef_G_stat, label='Coef_G_stat')
plt.plot(x_coordinates, Coef_R_sub_ns, label='Coef_R_sub_ns')
plt.plot(x_coordinates, Coef_R_sub_multi, label='Coef_R_sub_multi')
plt.plot(x_coordinates, Coef_R_sub_stat, label='Coef_R_sub_stat')
plt.plot(x_coordinates, Coef_R_total_ns, label='Coef_R_total_ns')
plt.plot(x_coordinates, Coef_R_total_multi, label='Coef_R_total_multi')
plt.plot(x_coordinates, Coef_R_total_stat, label='Coef_R_total_stat')

# Adding titles and labels
plt.title('Success rate of Coef')
plt.xlabel('Sampling number')
plt.ylabel('Success rate')

# Adding a legend to distinguish the lines
plt.legend()

# Display the plot
plt.show()


