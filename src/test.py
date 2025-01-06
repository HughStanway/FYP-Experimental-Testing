#pylint: skip-file

# Re-importing necessary libraries after reset
import matplotlib.pyplot as plt
import pandas as pd

# Data for the new embedding model
data = [
    ("GCJ2-4/GCJ2-4_cpp/data/", 10, 0.9281376680301745),
    ("GCJ2-4/GCJ2-4_cpp/data/", 50, 0.8051804099506197),
    ("GCJ2-4/GCJ2-4_cpp/data/", 100, 0.744041963952097),
    ("GCJ2-4/GCJ2-4_cpp/data/", 300, 0.6385863211978813),
    ("GCJ2-4/GCJ2-4_java/data/", 10, 0.9110107435015532),
    ("GCJ2-4/GCJ2-4_java/data/", 50, 0.7410564649455262),
    ("GCJ2-4/GCJ2-4_java/data/", 100, 0.6640672115173967),
    ("GCJ2-4/GCJ2-4_java/data/", 300, 0.5529525572631379),
    ("GCJ2-4/GCJ2-4_php/data/", 10, 0.9019627293263742),
    ("GCJ2-4/GCJ2-4_php/data/", 50, 0.7506587911286884),
    ("GCJ2-4/GCJ2-4_php/data/", 100, 0.6786189422032118),
    ("GCJ2-4/GCJ2-4_php/data/", 300, 0.5583730680350294),
    ("GCJ2-4/GCJ2-4_py/data/", 10, 0.9354397145373172),
    ("GCJ2-4/GCJ2-4_py/data/", 50, 0.8269968247554743),
    ("GCJ2-4/GCJ2-4_py/data/", 100, 0.7721126500050829),
    ("GCJ2-4/GCJ2-4_py/data/", 300, 0.6669001105983492)
]

# Convert to DataFrame
df_new = pd.DataFrame(data, columns=["Path", "k", "Mean Average Precision"])

# Plotting
plt.figure(figsize=(12, 8))

# Plot each path
for path in df_new["Path"].unique():
    subset = df_new[df_new["Path"] == path]
    plt.plot(subset["k"], subset["Mean Average Precision"], label=path, marker='o')

# Adding labels and title
plt.xlabel("k")
plt.ylabel("Mean Average Precision")
plt.title("Mean Average Precision vs k using jina-v2-base-code embedding model")
plt.legend(title="Path")
plt.grid(True)

# Show plot
plt.show()
