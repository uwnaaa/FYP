import numpy as np
df = pd_read.csv
# Extracting data for plotting
premise_type = df_ayam['premise_type']
district = df_ayam['district']
price = df_ayam['price']

# Calculate average price for each combination of district and premise type
avg_price = df_ayam.groupby(['district', 'premise_type'])['price'].mean().unstack()

# Extracting unique districts and premise types
districts = avg_price.index
premise_types = avg_price.columns

# Number of districts and premise types
n_districts = len(districts)
n_types = len(premise_types)

# Create an array for the x-axis positions
x = np.arange(n_districts) * (1.5)

# Width of the bars
width = 0.2

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each premise type
for i, premise_type in enumerate(premise_types):
    pos = x + i * width - width * (n_types / 2)
    ax.bar(pos, avg_price[premise_type], width, label=premise_type)

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Average Price')
ax.set_title('Average Price Ayam by Premise Type and District')
ax.set_xticks(x)
ax.set_xticklabels(districts)

# Adding legend
ax.legend(title='Premise Type', loc='upper right')

plt.tight_layout()
plt.show()



