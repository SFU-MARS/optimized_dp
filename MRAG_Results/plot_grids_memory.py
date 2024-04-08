import numpy as numpy
import matplotlib.pyplot as plt

# plotting data
grids_7D = [17**7, 18**7, 18**3*19**4, 19**7, 19**4*20**3, 19**3*20**4, 20**7, 20**4*21**3, 20**3*21**4, 21**7, 21**6*22]  # , 21**6*23
memory_7D = [8.34, 12.38, 15.33, 18.02, 20.98, 22.08, 25.75, 29.77, 31.25, 36.18, 37.87] # unit: GB , 39.58
time_7D = [661.43, 977.06, 1233.98, 1450.93, 1678.00, 1801.53, 2088.49, 2413.23, 2569.39, 2972.76, 3066.04] # unit: s , 4353.03

grids_8D = [9**8, 9**4*10**4, 10**8, 11**8, 12**8, 12**4*13**4, 13**8, 13**6*14**2, 13**4*14**4, 14**8, 14**6*15**2, 14**4*15**4]  # unit: 1
grids_8D_short = [12**8, 12**4*13**4, 13**8, 13**6*14**2, 13**4*14**4, 13**2*14**6, 14**8, 14**6*15**2, 14**4*15**4]  # unit: 1

memory_8D = [0.99, 1.45, 2.14, 4.42, 8.74, 11.98, 16.45, 19.06, 22.08, 29.65, 34.02, 39.03] # unit: GB
memory_8D_short = [8.74, 11.98, 16.45, 19.06, 22.08, 25.59, 29.65, 34.02, 39.03] # unit: GB

time_8D = [53.44, 82.32, 126.14, 270.27, 534.50, 887.33, 1516.57, 1715.52, 1988.96, 2782.50, 3127.69, 3722.27] # unit: s
time_8D_short = [534.50, 887.33, 1516.57, 1715.52, 1988.96, 2345.13, 2782.50, 3127.69, 3722.27] # unit: s



# Plotting memory usage
plt.figure(figsize=(10, 5))
plt.scatter(grids_7D, memory_7D, label='7D', marker='o', color='b')
# plt.scatter(grids_8D, memory_8D, label='8D', marker='o', color='r')
plt.scatter(grids_8D_short, memory_8D_short, label='8D', marker='o', color='r')
plt.xlabel('Grids')
plt.ylabel('Memory (GB)')
plt.title('Memory Usage')
plt.legend()
plt.grid(True)
plt.savefig('MRAG_Results/CalculationMemory_7D8D.png')
plt.show()


# plt.figure(figsize=(10, 5))
# plt.scatter(grids_7D, memory_7D, label='7D', marker='o')
# plt.scatter(grids_8D, memory_8D, label='8D', marker='o')
# # Connecting points with dashed lines
# for i in range(len(grids_7D)):
#     plt.plot([grids_7D[i], grids_8D[i]], [memory_7D[i], memory_8D[i]], 'k--')

# plt.xlabel('Grids')
# plt.ylabel('Memory (GB)')
# plt.title('Memory Usage')
# plt.legend()
# plt.grid(True)
# plt.savefig('MRAG_Results/CalculationMemory_7D8D.png')
# plt.show()


# Plotting time consumption
plt.figure(figsize=(10, 5))
plt.plot(grids_7D, time_7D, marker='o', label='7D')
# plt.plot(grids_8D, time_8D, marker='o', label='8D')
plt.plot(grids_8D_short, time_8D_short, marker='o', label='8D')
plt.xlabel('Grids')
plt.ylabel('Time (s)')
plt.title('Time Consumption')
plt.legend()
plt.grid(True)
plt.savefig('MRAG_Results/CalculationTime_7D8D.png')
plt.show()
