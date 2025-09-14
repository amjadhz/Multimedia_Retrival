import numpy as np
from matplotlib import pyplot as plt
import random


# Parameters
N = 100
no_bins = int(round(np.sqrt(N)))  # Recommended number of bins
# no_bins = <TRY_A_DIFFERENT_SETTING>
print("Number of bins: ", no_bins)

# Generate random sample data
data = np.array([random.randint(0, no_bins) for _ in range(N)])

# Compute histogram
counts, bins = np.histogram(data, bins=no_bins)

# Plot histogram
fig, ax = plt.subplots()
ax.hist(bins[:-1], bins, weights=counts)
ax.set_title("Your Data")
ax.set_ylabel("Number of samples per bin")
ax.set_xlabel("Bins")
plt.show()
