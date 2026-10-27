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
plt.bar(
    ['No Second Type', 'Has Second Type'],
    [no_second_type, has_second_type],
    color=['pink', 'purple'],
)
plt.title("The number of pokemon with and without a second type")
plt.xlabel("Does the pokemon have a Second Type")
plt.ylabel("Number of Pokemon")
plt.show()
#Roughly as many pokemon have a second type as those that do not. Now lets look at distribution of second types disgarding those without
filtered_data = pokemon_data[pokemon_data['Type2'] != 'None']
second_type_count=filtered_data['Type2'].value_counts()
plt.pie(second_type_count.values, labels=second_type_count.index, color='purple')
plt.title('Distribution of Second Types')
plt.xlabel('Number of Pokemon')
plt.ylabel('Second Type')
plt.show()