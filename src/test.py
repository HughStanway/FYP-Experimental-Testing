#pylint: skip-file

# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd

# Data for the embedding model
data = [
    ("voyage-code-3", 256, 10, 0.9935414460748676),
    ("voyage-code-3", 256, 50, 0.9873108259303993),
    ("voyage-code-3", 256, 100, 0.9849744568863041),
    ("voyage-code-3", 256, 300, 0.9799543132093224),
    ("voyage-code-3", 512, 10, 0.9922014533730171),
    ("voyage-code-3", 512, 50, 0.987305766804373),
    ("voyage-code-3", 512, 100, 0.9853961586458679),
    ("voyage-code-3", 512, 300, 0.9813347325416327),
    ("voyage-code-3", 1024, 10, 0.9932101117676412),
    ("voyage-code-3", 1024, 50, 0.9877509999535339),
    ("voyage-code-3", 1024, 100, 0.9857814606804016),
    ("voyage-code-3", 1024, 300, 0.9816292266846793),
    ("voyage-code-3", 2048, 10, 0.9927266861960238),
    ("voyage-code-3", 2048, 50, 0.9877260576773006),
    ("voyage-code-3", 2048, 100, 0.9857616064556797),
    ("voyage-code-3", 2048, 300, 0.9817355352151287)
]

# Convert to DataFrame
df_new = pd.DataFrame(data, columns=["Embedding Model", "Output Dimension", "k", "Mean Average Precision"])

# Plotting
plt.figure(figsize=(12, 8))

# Plot each output dimension
for dimension in df_new["Output Dimension"].unique():
    subset = df_new[df_new["Output Dimension"] == dimension]
    plt.plot(subset["k"], subset["Mean Average Precision"], label=f'Output Dimension {dimension}', marker='o')

# Adding labels and title
plt.xlabel("k")
plt.ylabel("Mean Average Precision")
plt.title("Mean Average Precision vs k for Different Output Dimensions")
plt.legend(title="Output Dimension")
plt.grid(True)

# Show plot
plt.show()