# Extracting data for plotting
states = df['state']
populations = df['population']

# Plotting
plt.figure(figsize=(10, 6))

# Bar chart for Population
plt.bar(states, populations, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('States')
plt.ylabel('Population')
plt.title('Population by State')
plt.xticks(rotation=90)  # Rotate state names for better readability

plt.tight_layout()
plt.show()
