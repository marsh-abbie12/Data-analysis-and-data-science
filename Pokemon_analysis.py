#This code will use data analysis to explore the pokemon data set to provide insights into the stats of different pokemon
#To try find patterns between attributes
#I will then focus my analysis on creating a model to predict the likelihood of a pokemon being legendary based on its stats and attributes

#Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Load our dataset
pokemon_data = pd.read_csv('PokemonData.csv')
print(pokemon_data.head())
#Looking at this data I can see where we have no second type this has been replaced with a NaN value, I will replace this
#with none to prepare for potential modelling later
pokemon_data['Type2'] = pokemon_data['Type2'].fillna('None')
#Next lets focus on the Analysis of our data. I want to first understand the distribution of different types of pokemon
#For simplicity at first I am going to just consider first types. Later I will look at second types and 
#later combine data to look at types overall.
type_counts=pokemon_data['Type1'].value_counts()
plt.barh(type_counts.index,type_counts.values,color='pink')
plt.title('The number of each Pokemon Type based on their first type')
plt.xlabel('Number of Pokemon')
plt.ylabel('Pokemon Type')
plt.show()
#This shows that the most common type of pokemon is water and the least common is flying. I want to now look at the second types specifcally first
#how many have a second type
no_second_type = (pokemon_data['Type2'] == 'None').sum()
has_second_type = (pokemon_data['Type2'] != 'None').sum()
plt.figure(figsize=(6, 5))
plt.pie(
    [no_second_type, has_second_type],
    labels=['No Second Type', 'Has Second Type'],
    colors=['pink', 'purple']
)
plt.title("The number of pokemon with and without a second type")
plt.xlabel("Does the pokemon have a Second Type")
plt.ylabel("Number of Pokemon")
plt.show()
#Roughly as many pokemon have a second type as those that do not. Now lets look at distribution of second types disgarding those without
filtered_data = pokemon_data[pokemon_data['Type2'] != 'None']
second_type_count=filtered_data['Type2'].value_counts()
plt.barh(second_type_count.index, second_type_count.values)
plt.title('Distribution of Second Types')
plt.xlabel('Number of Pokemon')
plt.ylabel('Second Type')
plt.show()

#Next I want to understand if there are difference in the attack characteristics depending on the type of pokemon. 
#I am first going to consider first type being the primary type of the pokemon for analysis and simplicity. I may later consider 
#combining types to get a better understanding of statistics.
Average_attack_type=pokemon_data.groupby("Type1")["Attack"].mean()
plt.barh(Average_attack_type.index,Average_attack_type.values,color='purple')
plt.title('Average Attack by Pokemon Type')
plt.xlabel('Average Attack')
plt.ylabel('Pokemon Type')
plt.show()

#This shows us that dark types have the highest average attack with normal types the lowest. I want to now consider not just the averages
#but the distribution of attack values for each type. I will use boxplots to do this.
plt.figure(figsize=(12, 6))
pokemon_data.boxplot(column='Attack', by='Type1', grid=False, patch_artist=True, boxprops=dict(facecolor='pink', color='black'), medianprops=dict(color='black'))
plt.title('Distribution of Attack by Pokemon Type')
plt.xlabel('Pokemon Type')
plt.ylabel('Attack')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Analysis for machine learning
#My main aim is to create a model where we can predict if a pokemon is legendary based on key attributes. To do this
#I want to do some exploratory data analysis. First I want to understand if there are noticeable legendary number differences in each generation.
gen_comparison = (
    pokemon_data.groupby(['Generation', 'Legendary'])
    .size()
    .unstack(fill_value=0)
)
gen_comparison.columns = ['Regular', 'Legendary']
gen_comparison.plot(
    kind='bar',
    color=["#b372cf", "#4fe8fc"],
    edgecolor='black',
    width=0.7,
    figsize=(9, 5),
)

plt.title('A comparison of the number of Legendary and Regular Pokemon in each Generation')
plt.xlabel('Generation')
plt.ylabel('Number of Pokémon')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
#This does seem to show a slight increase in legendaries for generation 3 whilst elsewhere it is consistent.
#This does suggest a factor variable for generation 3 vs not may be useful in the model.
