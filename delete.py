
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