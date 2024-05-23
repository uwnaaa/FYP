pip install census
import matplotlib.pyplot as plt

# Population data for the specified states
states_population = {
    'Johor': 4020000,
    'Kedah': 2200000,
    'Kelantan': 1950000,
    'Melaka': 930000,
    'Negeri Sembilan': 1160000,
    'Pahang': 1660000,
    'Perak': 2540000,
    'Perlis': 260000,
    'Pulau Pinang': 1780000,
    'Sabah': 3930000,
    'Sarawak': 2860000,
    'Selangor': 7070000,
    'Terengganu': 1230000,
    'W.P. Kuala Lumpur': 2140000,
    'W.P. Labuan': 100000,
    'W.P. Putrajaya': 120000
}

# Extract state names and populations for plotting
states_names = list(states_population.keys())
populations = list(states_population.values())

# Plotting
plt.figure(figsize=(8, 6))
plt.barh(states_names, populations, color='green')
plt.xlabel('Population')
plt.ylabel('State')
plt.title('Population by State in Malaysia')
plt.gca().invert_yaxis()  # Invert y-axis to have states with highest population at the top
plt.tight_layout()
plt.show()
