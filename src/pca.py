import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import hashlib
import diskcache as dc
# Optionally, you can import t-SNE if nonlinear dimensionality reduction is preferred:
# from sklearn.manifold import TSNE

cache = dc.Cache('embedding_cache')
cache_voyage = dc.Cache('embedding_cache_voyage')
embedding_model = "deepseek-r1:1.5B"

embeddings_list = []
categories_list = []

directory: Path = Path("POJ/")

def compute_embedding(file_text: str):
    key = f"{file_text}-{embedding_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()

    if embedding_model != "voyage-code-3" and hashed_key in cache:
        return cache[hashed_key]
    
    if embedding_model == "voyage-code-3" and hashed_key in cache_voyage:
        return cache_voyage[hashed_key]
    
def get_category(file_path: str):
    parts = file_path.split('/')
    return parts[1]

insert_count = 0
for file_path in directory.rglob('*'):
    if file_path.is_file() and file_path.suffix == '.txt':
        try:
            with open(file_path, 'r', encoding='iso8859-1') as f: 
                file_text = f.read()

            # Create embedding for text
            embeddings = compute_embedding(file_text)
            embeddings_list.append(embeddings)
            
            category = get_category(str(file_path))
            categories_list.append(category)
            

        except Exception as e:
            print(f"Error adding to collection: {e}")

# Convert embeddings list to a NumPy array
embeddings = np.array(embeddings_list)

# If categories are not integers, map each unique category to an integer label.
unique_categories = sorted(set(categories_list))
category_to_index = {cat: i for i, cat in enumerate(unique_categories)}
labels = np.array([category_to_index[cat] for cat in categories_list])
num_categories = len(unique_categories)

# Dimensionality reduction to 2D using PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Alternatively, if you'd prefer t-SNE (note: t-SNE can be slower with large datasets)
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)

# Plotting the results
plt.figure(figsize=(12, 10))
# Create a colormap that supports the number of unique categories
cmap = plt.cm.get_cmap('gist_ncar', num_categories)

# Plot each category separately with a unique color
for cat in range(num_categories):
    idx = labels == cat
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                s=10,                      # Adjust marker size as needed
                color=[cmap(cat)],         # Unique color per category
                label=f'Category {unique_categories[cat]}')

plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.xlabel("")
plt.ylabel("")

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.savefig(f"figures/pca/pca_{embedding_model.replace('/', '-')}.png", dpi=300, format='png')
plt.show()
