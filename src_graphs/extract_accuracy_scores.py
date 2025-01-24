# pylint: skip-file

from pathlib import Path
import re
import json

filepath = "metrics/accuracy/GCJ2-4_py"

data = {}

directory = Path(filepath)
counter = 0
for file_path in directory.rglob('*'):
   if file_path.is_file():
        filename = str(file_path).split('/')[-1].split('_')[2:]
        del filename[2]

        embedding_model = filename[0]
        k = int(filename[1][1:])
        database = filename[2][:-4]

        counter += 1
        with open(file_path, 'r', encoding='iso8859-1') as f: 
            file_text = f.read()

        # Extract MAP and Precision using regex
        map_match = re.search(r"Mean Average Precision \(MAP\)\s*=\s*([\d.]+)", file_text)
        precision_match = re.search(r"Precision\s*=\s*([\d.]+)", file_text)

        # Convert to float and round to 4 decimal places
        map_value = round(float(map_match.group(1)), 4) if map_match else None
        precision_value = round(float(precision_match.group(1)), 4) if precision_match else None

        #print("MAP:", map_value)
        #print("Precision:", precision_value)

        if database not in data.keys():
            data[database] = {}

        if embedding_model not in data[database]:
            data[database][embedding_model] = {}
        
        data[database][embedding_model][int(k)] = map_value

for db_name, embeddings in data.items():
    for embedding_name, values in embeddings.items():
        # Sort the innermost dictionary by values in ascending order
        sorted_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
        data[db_name][embedding_name] = sorted_values

for db_name, embedding_model in data.items():
    for embedding, values in embedding_model.items():
        print(f"[{db_name}][{embedding}]")
        string = ""
        res = []
        for k, measurement in values.items():
            string += f" {measurement} &"
            res.append(measurement)
        print(string)

        data[db_name][embedding] = res

print(json.dumps(data, indent=4))
        