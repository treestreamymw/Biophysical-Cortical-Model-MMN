import json
import matplotlib.pyplot as plt

with open('output_files/Data_0.045.json') as json_file:
    data = json.load(json_file)



bins = {i:0 for i in range(5)}


for spike in data['simData']['spkt']:
    bins[int((spike-200) /2000)]+=1

print(bins)

plt.scatter(bins.keys(),bins.values())
plt.plot(bins.keys(),bins.values())

plt.show()
