import json
import matplotlib.pyplot as plt

with open('output_files/single_cell_1.json') as json_file:
    data = json.load(json_file)

print(data.keys())
print(data['simData'].keys())
print(data['simData']['spkt'])
