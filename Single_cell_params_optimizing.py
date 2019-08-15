import json
import matplotlib.pyplot as plt

with open('output_files/Data_0.045.json') as json_file:
    data = json.load(json_file)


bins = {i:0 for i in range(7)}


for spike in data['simData']['spkt']:
    bins[int((spike-200) /2000)]+=1

print(bins)
rs_current = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
plt.scatter(rs_current,bins.values())
plt.plot(rs_current,bins.values())

plt.show()
