<img src="https://bit.ly/2VnXWr2" width="100" align="left">

# Duel of Sorcerers
You are witnessing an epic battle between two powerful sorcerers: Gandalf and Saruman. Each sorcerer has 10 spells of variable power in their mind and they are going to throw them one after the other. The winner of the duel will be the one who wins more of those clashes between spells. Spells are represented as a list of 10 integers whose value equals the power of the spell.
```
gandalf = [10, 11, 13, 30, 22, 11, 10, 33, 22, 22]
saruman = [23, 66, 12, 43, 12, 10, 44, 23, 12, 17]
```
For example:
- The first clash is won by Saruman: 10 against 23.
- The second clash is won by Saruman: 11 against 66.
- ...

You will create two variables, one for each sorcerer, where the sum of clashes won will be stored. Depending on which variable is greater at the end of the duel, you will show one of the following three results on the screen:
* Gandalf wins
* Saruman wins
* Tie

<img src="Images/content_lightning_bolt_big.jpeg" width="400">

## Tools
You don't necessarily need to use all the tools. Maybe you opt to use some of them or completely different ones, they are given to help you shape the exercise. Programming exercises can be solved in many different ways.

1. Data structures: **lists, dictionaries**
2. Loop: **for loop**
3. Conditional statements: **if-elif-else**
4. Functions: **range(), len(), print()**

## Tasks

<b>Hint: You dont need necessarily to follow the instructions in this notebook to solve the tasks. simply just get the job done effeciently ;) 

#### 1. Create two variables called `gandalf` and `saruman` and assign them the spell power lists. Create a variable called `spells` to store the number of spells that the sorcerers cast. 


```python
spells = 10
gandalf = [10, 11, 13, 30, 22, 11, 10, 33, 22, 22]
saruman = [23, 66, 12, 43, 12, 10, 44, 23, 12, 17]

gandalf_wins=0
saruman_wins=0

for saruman, gandalf in zip(saruman, gandalf):
    if gandalf>saruman:
        gandalf_wins += 1
    elif saruman>gandalf:
        saruman_wins += 1
if gandalf > saruman:
    print( "Gandalf wins with", gandalf_wins, "points and Saruman loses with", saruman_wins)
    
elif saruma > gandalf:
    print( "Saruman wins with :", saruman_wins, "points and GAndalf loses with", gandalf_wins)   

```

    Gandalf wins with 6 points and Saruman loses with 4
    

#### 2. Create two variables called `gandalf_wins` and `saruman_wins`. Set both of them to 0. 
You will use these variables to count the number of clashes each sorcerer wins. 


```python
gandalf_wins=0
saruman_wins=0
```

#### 3. Using the lists of spells of both sorcerers, update variables `gandalf_wins` and `saruman_wins` to count the number of times each sorcerer wins a clash. 


```python

```

#### 4. Who won the battle?
Print `Gandalf wins`, `Saruman wins` or `Tie` depending on the result. 


```python
spells = 10
gandalf = [10, 11, 13, 30, 22, 11, 10, 33, 22, 22]
saruman = [23, 66, 12, 43, 12, 10, 44, 23, 12, 17]

gandalf_wins=0
saruman_wins=0

for saruman, gandalf in zip(saruman, gandalf):
    if gandalf>saruman:
        gandalf_wins += 1
    elif saruman>gandalf:
        saruman_wins += 1
if gandalf_wins > saruman_wins:
    print( "Gandalf wins with", gandalf_wins, "points and Saruman loses with", saruman_wins)
    
elif saruma_wigs > gandalf_wins:
    print( "Saruman wins with :", saruman_wins, "points and GAndalf loses with", gandalf_wins)   
```

    Gandalf wins with 6 points and Saruman loses with 4
    

## Bonus

In this bonus challenge, you'll need to check the winner of the battle but this time, a sorcerer wins if he succeeds in winning 3 spell clashes in a row.

Also, the spells now have a name and there is a dictionary that associates that name to a power.

```
POWER = {
    'Fireball': 50, 
    'Lightning bolt': 40, 
    'Magic arrow': 10, 
    'Black Tentacles': 25, 
    'Contagion': 45
}

gandalf = ['Fireball', 'Lightning bolt', 'Lightning bolt', 'Magic arrow', 'Fireball', 
           'Magic arrow', 'Lightning bolt', 'Fireball', 'Fireball', 'Fireball']
saruman = ['Contagion', 'Contagion', 'Black Tentacles', 'Fireball', 'Black Tentacles', 
           'Lightning bolt', 'Magic arrow', 'Contagion', 'Magic arrow', 'Magic arrow']
```

#### 1. Create variables `POWER`, `gandalf` and `saruman` as seen above. Create a variable called `spells` to store the number of spells that the sorcerers cast. 


```python
spells=10
POWER = {
    'Fireball': 50, 
    'Lightning bolt': 40, 
    'Magic arrow': 10, 
    'Black Tentacles': 25, 
    'Contagion': 45
}

gandalf = ['Fireball', 'Lightning bolt', 'Lightning bolt', 'Magic arrow', 'Fireball', 
           'Magic arrow', 'Lightning bolt', 'Fireball', 'Fireball', 'Fireball']
saruman = ['Contagion', 'Contagion', 'Black Tentacles', 'Fireball', 'Black Tentacles', 
           'Lightning bolt', 'Magic arrow', 'Contagion', 'Magic arrow', 'Magic arrow']

```

    <zip object at 0x00000284975C3C40>
    

#### 2. Create two variables called `gandalf_wins` and `saruman_wins`. Set both of them to 0. 


```python
gandalf_wins=0
saruman_wins=0
```

#### 3. Create two variables called `gandalf_power` and `saruman_power` to store the list of spell powers of each sorcerer.


```python
gandalf_power=0
saruman_power=0
```

#### 4. The battle starts! Using the variables you've created above, code the execution of spell clashes. Remember that a sorcerer wins if he succeeds in winning 3 spell clashes in a row. 
If a clash ends up in a tie, the counter of wins in a row is not restarted to 0. Remember to print who is the winner of the battle. 


```python
spells=10
POWER = {
    'Fireball': 50, 
    'Lightning bolt': 40, 
    'Magic arrow': 10, 
    'Black Tentacles': 25, 
    'Contagion': 45
}

gandalf = ['Fireball', 'Lightning bolt', 'Lightning bolt', 'Magic arrow', 'Fireball', 
           'Magic arrow', 'Lightning bolt', 'Fireball', 'Fireball', 'Fireball']
saruman = ['Contagion', 'Contagion', 'Black Tentacles', 'Fireball', 'Black Tentacles', 
           'Lightning bolt', 'Magic arrow', 'Contagion', 'Magic arrow', 'Magic arrow']

gandalf_wins=0
saruman_wins=0

gandalf_power=0
saruman_power=0

for gandalf_spell, saruman_spell in zip(gandalf,saruman): 
    if POWER[gandalf_spell] > POWER[saruman_spell]: 
        gandalf_wins+=1 
    elif POWER[saruman_spell] > POWER[gandalf_spell]: 
        saruman_wins+=1 

if gandalf_wins > saruman_wins:
    print( "Gandalf wins with", gandalf_wins, "points and Saruman loses with", saruman_wins)
    
elif saruma_wigs > gandalf_wins:
    print( "Saruman wins with :", saruman_wins, "points and GAndalf loses with", gandalf_wins)  
```

    Gandalf wins with 7 points and Saruman loses with 3
    

#### 5. Find the average spell power of Gandalf and Saruman. 


```python
spells=10
POWER = {
    'Fireball': 50, 
    'Lightning bolt': 40, 
    'Magic arrow': 10, 
    'Black Tentacles': 25, 
    'Contagion': 45
}

gandalf = ['Fireball', 'Lightning bolt', 'Lightning bolt', 'Magic arrow', 'Fireball', 
           'Magic arrow', 'Lightning bolt', 'Fireball', 'Fireball', 'Fireball']
saruman = ['Contagion', 'Contagion', 'Black Tentacles', 'Fireball', 'Black Tentacles', 
           'Lightning bolt', 'Magic arrow', 'Contagion', 'Magic arrow', 'Magic arrow']

gandalf_wins=0
saruman_wins=0

gandalf_power=0
saruman_power=0


# spells list value

gandalf_power= [POWER[k] for k in gandalf]


saruman_power= [POWER[k] for k in saruman if k in POWER]


#average

average_gandalf = sum(gandalf_power)/len(gandalf_power)

print("Average of Gandalf: ", round(average_gandalf,3))

average_saruman = sum(saruman_power)/len(saruman_power)

print("Average of Saruman: ", round(average_saruman,3))


```

    Average of Gandalf:  39.0
    Average of Saruman:  30.5
    

#### 6. Find the standard deviation of the spell power of Gandalf and Saruman. 


```python
spells=10
POWER = {
    'Fireball': 50, 
    'Lightning bolt': 40, 
    'Magic arrow': 10, 
    'Black Tentacles': 25, 
    'Contagion': 45
}

gandalf = ['Fireball', 'Lightning bolt', 'Lightning bolt', 'Magic arrow', 'Fireball', 
           'Magic arrow', 'Lightning bolt', 'Fireball', 'Fireball', 'Fireball']
saruman = ['Contagion', 'Contagion', 'Black Tentacles', 'Fireball', 'Black Tentacles', 
           'Lightning bolt', 'Magic arrow', 'Contagion', 'Magic arrow', 'Magic arrow']

gandalf_wins=0
saruman_wins=0

gandalf_power=0
saruman_power=0


# spells list value

gandalf_power= [POWER[k] for k in gandalf]


saruman_power= [POWER[k] for k in saruman if k in POWER]

#calculate standard deviation of list


import numpy as np


print ("Gandalf's desviation", np.std(gandalf_power))
print ("Saruman's desviation", np.std(saruman_power))
```

    Gandalf's desviation 15.132745950421556
    Saruman's desviation 15.56438241627338
    


```python

```
