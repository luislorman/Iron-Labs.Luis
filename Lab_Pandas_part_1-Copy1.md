# Using Pandas


```python
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 200)
## to make it possible to display multiple output inside one cell 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

<b>load the data from the vehicles.csv file into pandas data frame


```python


people_df= pd.read_csv("data/vehicles.csv")
people_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Displacement</th>
      <th>Cylinders</th>
      <th>Transmission</th>
      <th>Drivetrain</th>
      <th>Vehicle Class</th>
      <th>Fuel Type</th>
      <th>Fuel Barrels/Year</th>
      <th>City MPG</th>
      <th>Highway MPG</th>
      <th>Combined MPG</th>
      <th>CO2 Emission Grams/Mile</th>
      <th>Fuel Cost/Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AM General</td>
      <td>DJ Po Vehicle 2WD</td>
      <td>1984</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>Automatic 3-spd</td>
      <td>2-Wheel Drive</td>
      <td>Special Purpose Vehicle 2WD</td>
      <td>Regular</td>
      <td>19.388824</td>
      <td>18</td>
      <td>17</td>
      <td>17</td>
      <td>522.764706</td>
      <td>1950</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AM General</td>
      <td>FJ8c Post Office</td>
      <td>1984</td>
      <td>4.2</td>
      <td>6.0</td>
      <td>Automatic 3-spd</td>
      <td>2-Wheel Drive</td>
      <td>Special Purpose Vehicle 2WD</td>
      <td>Regular</td>
      <td>25.354615</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>683.615385</td>
      <td>2550</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AM General</td>
      <td>Post Office DJ5 2WD</td>
      <td>1985</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>Automatic 3-spd</td>
      <td>Rear-Wheel Drive</td>
      <td>Special Purpose Vehicle 2WD</td>
      <td>Regular</td>
      <td>20.600625</td>
      <td>16</td>
      <td>17</td>
      <td>16</td>
      <td>555.437500</td>
      <td>2100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AM General</td>
      <td>Post Office DJ8 2WD</td>
      <td>1985</td>
      <td>4.2</td>
      <td>6.0</td>
      <td>Automatic 3-spd</td>
      <td>Rear-Wheel Drive</td>
      <td>Special Purpose Vehicle 2WD</td>
      <td>Regular</td>
      <td>25.354615</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>683.615385</td>
      <td>2550</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ASC Incorporated</td>
      <td>GNX</td>
      <td>1987</td>
      <td>3.8</td>
      <td>6.0</td>
      <td>Automatic 4-spd</td>
      <td>Rear-Wheel Drive</td>
      <td>Midsize Cars</td>
      <td>Premium</td>
      <td>20.600625</td>
      <td>14</td>
      <td>21</td>
      <td>16</td>
      <td>555.437500</td>
      <td>2550</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35947</th>
      <td>smart</td>
      <td>fortwo coupe</td>
      <td>2013</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>Auto(AM5)</td>
      <td>Rear-Wheel Drive</td>
      <td>Two Seaters</td>
      <td>Premium</td>
      <td>9.155833</td>
      <td>34</td>
      <td>38</td>
      <td>36</td>
      <td>244.000000</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>35948</th>
      <td>smart</td>
      <td>fortwo coupe</td>
      <td>2014</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>Auto(AM5)</td>
      <td>Rear-Wheel Drive</td>
      <td>Two Seaters</td>
      <td>Premium</td>
      <td>9.155833</td>
      <td>34</td>
      <td>38</td>
      <td>36</td>
      <td>243.000000</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>35949</th>
      <td>smart</td>
      <td>fortwo coupe</td>
      <td>2015</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>Auto(AM5)</td>
      <td>Rear-Wheel Drive</td>
      <td>Two Seaters</td>
      <td>Premium</td>
      <td>9.155833</td>
      <td>34</td>
      <td>38</td>
      <td>36</td>
      <td>244.000000</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>35950</th>
      <td>smart</td>
      <td>fortwo coupe</td>
      <td>2016</td>
      <td>0.9</td>
      <td>3.0</td>
      <td>Auto(AM6)</td>
      <td>Rear-Wheel Drive</td>
      <td>Two Seaters</td>
      <td>Premium</td>
      <td>9.155833</td>
      <td>34</td>
      <td>39</td>
      <td>36</td>
      <td>246.000000</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>35951</th>
      <td>smart</td>
      <td>fortwo coupe</td>
      <td>2016</td>
      <td>0.9</td>
      <td>3.0</td>
      <td>Manual 5-spd</td>
      <td>Rear-Wheel Drive</td>
      <td>Two Seaters</td>
      <td>Premium</td>
      <td>9.417429</td>
      <td>32</td>
      <td>39</td>
      <td>35</td>
      <td>255.000000</td>
      <td>1150</td>
    </tr>
  </tbody>
</table>
<p>35952 rows × 15 columns</p>
</div>



First exploration of the dataset:

- How many observations does it have?
- Look at all the columns: do you understand what they mean?
- Look at the raw data: do you see anything weird?
- Look at the data types: are they the expected ones for the information the column contains?


```python
## Your Code here

How many observations does it have? No obervservations, since there is no #
Look at all the columns: do you understand what they mean? yes
Look at the raw data: do you see anything weird? no
Look at the data types: are they the expected ones for the information the column contains? yes
```

### Cleaning and wrangling data

- Some car brand names refer to the same brand. Replace all brand names that contain the word "Dutton" for simply "Dutton". If you find similar examples, clean their names too. Use `loc` with boolean indexing.

- Convert CO2 Emissions from Grams/Mile to Grams/Km

- Create a binary column that solely indicates if the transmission of a car is automatic or manual. Use `pandas.Series.str.startswith` and .

- convert MPG columns to km_per_liter

Note:
<br>Converting Grams/Mile to Grams/Km

1 Mile = 1.60934 Km

Converting Gallons to Liters

1 Gallon = 3.78541 Liters




```python
- people_df[people_df["Make"].str.contains("Dutton")]  #contains the word Dutton
  people_df.loc[people_df["Make"].str.contains("Dutton"), "Make"] = "Dutton" # Rename what contains word Dutton
  people_df.loc[[11012],["Make"]] #to see the change

-people_df['CO2 Emission Grams/Mile'] = people_df["CO2 Emission Grams/Mile"] / 1.60934 
 people_df.rename(columns={"CO2 Emission Grams/Mile": "CO2 Emission Grams/km"})
-def trans_replace(name):
    if name.startswith("A"):
        return 1
    else:
        return 0
people_df["Transmission 1/0"] = people_df.Transmission.apply(trans_replace)

people_df

-????????????


```


      Input In [9]
        people_df.loc[people_df["Make"].str.contains("Dutton"), "Make"] = "Dutton" # Rename what contains word Dutton
        ^
    IndentationError: unexpected indent
    


### Gathering insights:

- How many car makers are there? How many models? Which car maker has the most cars in the dataset?

- When were these cars made? How big is the engine of these cars?

- What's the frequency of different transmissions, drivetrains and fuel types?

- What's the car that consumes the least/most fuel?


```python
# Your Code here

```

<b> (Optional)

What brand has the worse CO2 Emissions on average?

Hint: use the function `sort_values()`


```python
## your Code here

```

Do cars with automatic transmission consume more fuel than cars with manual transmission on average?


```python
## Your Code is here 

```
