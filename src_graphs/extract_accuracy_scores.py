# pylint: skip-file

from pathlib import Path
import re
import json

def get_accuracy_scores(filepath):
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

                if database not in data.keys():
                    data[database] = {}

                if embedding_model not in data[database]:
                    data[database][embedding_model] = {}
                
                data[database][embedding_model][int(k)] = [precision_value, map_value]


    for db_name, embeddings in data.items():
        for embedding_name, values in embeddings.items():
            # Sort the innermost dictionary by values in ascending order
            sorted_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
            data[db_name][embedding_name] = sorted_values

    for db_name, embedding_model in data.items():
        for embedding, values in embedding_model.items():
            precision_values = []
            map_values = []
            for k, measurement in values.items():
                precision_values.append(measurement[0])
                map_values.append(measurement[1])

            data[db_name][embedding] = {
                "Precision": precision_values,
                "MAP": map_values
            }

    print(json.dumps(data, indent=4))
    return data

def get_average_helper():
    filepath = "metrics/accuracy/POJ"
    results_POJ = get_accuracy_scores(filepath)

    filepath = "metrics/accuracy/GCJ2-4_cpp"
    results_cpp = get_accuracy_scores(filepath)

    filepath = "metrics/accuracy/GCJ2-4_java"
    results_java = get_accuracy_scores(filepath)

    filepath = "metrics/accuracy/GCJ2-4_py"
    results_py = get_accuracy_scores(filepath)

    average = {}
    for db in ["chroma", "milvus", "weaviate", "qdrant"]:
        for model in ["voyage-code-3", "ordis-jina-embeddings-v2-base-code", "llama3.2", "deepseek-r1:1.5B"]:
                def calc_average(metric):
                    precision_poj = results_POJ[db][model][metric]
                    precision_cpp = results_cpp[db][model][metric]
                    precision_java = results_java[db][model][metric]
                    precision_py = results_py[db][model][metric]

                    res = [0, 0, 0 ,0]
                    for i in range(4):
                        res[i] = precision_poj[i] + precision_cpp[i] + precision_java[i] + precision_py[i]
                        tmp = round(res[i] / 4, 4)
                        res[i] = tmp

                    return res

                if db not in average:
                     average[db] = {}

                if model not in average[db]:
                     average[db][model] = {}

                average[db][model]["Precision"] = calc_average("Precision")
                average[db][model]["MAP"] = calc_average("MAP")

    return average

def get_average():
    tmp = get_average_helper()
    average = {}

    for model in ["voyage-code-3", "ordis-jina-embeddings-v2-base-code", "llama3.2", "deepseek-r1:1.5B"]:
        def temp(metric, tmp):
            chroma = tmp["chroma"][model][metric]
            milvus = tmp["chroma"][model][metric]
            weaviate = tmp["chroma"][model][metric]
            qdrant = tmp["chroma"][model][metric]

            res = [0, 0, 0, 0]
            for i in range(4):
                res[i] = chroma[i] + milvus[i] + weaviate[i] + qdrant[i]
                tmp = round(res[i] / 4, 4)
                res[i] = tmp

            return res
        
        if model not in average:
            average[model] = {}
        
        average[model]['Precision'] = temp("Precision", tmp)
        average[model]['MAP'] = temp("MAP", tmp)

    return average

if __name__ == "__main__":
    filepath = "metrics/accuracy/GCJ2-4_py"
    results = get_accuracy_scores(filepath)

    for database, data in results.items():
        precision = data['llama3.2']['Precision']
        mean_ap = data['llama3.2']['MAP']

        print(database)
        print(" & ".join([str(num) for num in mean_ap]))
        