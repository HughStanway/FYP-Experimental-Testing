#!/bin/bash

# Start the venv
source .venv/bin/activate
export VOYAGE_API_KEY="pa-DimtsdzkoRsdcBXk_XKOl9uMk6Ocm5DBqAqcNuiRJWs"
pip install -r requirements.txt

if [ -z "$VIRTUAL_ENV" ]; then
  echo "Error: Virtual environment is not active. Exiting."
fi

# Run python scrips
for db in chroma milvus weaviate qdrant
do
	if [ "$db" == "chroma" ]; then
		echo "Using Chroma"

		docker compose up -d
		echo "Running: $operation on $db"
		python src/execution_tests.py POJ/ --embedding-model deepseek-r1:1.5B --database $db --operation query
		docker compose down
	elif [ "$db" == "milvus" ]; then
		echo "Using Milvus"

		cd ../milvus && docker compose up -d && cd ../FYP
		for operation in add query
		do
			echo "Running: $operation on $db"
			python src/execution_tests.py POJ/ --embedding-model deepseek-r1:1.5B --database $db --operation $operation
		done
		cd ../milvus && docker compose down && cd ../FYP
	elif [ "$db" == "weaviate" ]; then
		echo "Using Weaviate"

		cd ../weaviate && docker compose up -d && cd ../FYP
		for operation in add query
		do
			echo "Running: $operation on $db"
			python src/execution_tests.py POJ/ --embedding-model deepseek-r1:1.5B --database $db --operation $operation
		done
		cd ../weaviate && docker compose down && cd ../FYP
	elif [ "$db" == "qdrant" ]; then
		echo "Using Qdrant"

		docker start bold_bose
		for operation in add query
		do
			echo "Running: $operation on $db"
			python src/execution_tests.py POJ/ --embedding-model deepseek-r1:1.5B --database $db --operation $operation
		done
		docker stop bold_bose
	else
		echo "Database is not recognized"
	fi
done

# Deactivate the venv
deactivate
