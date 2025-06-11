import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Read csv
df = pandas.read_csv("data.csv")

# Change from string to numerical
d = {'Hazardous': 0, 'Poor': 1, 'Moderate': 2, 'Good': 3}
df['Air Quality'] = df['Air Quality'].map(d)

# List attributes
attributes =[
  'Temperature', 'Humidity',
  'PM2.5', 'PM10', 
  'NO2', 'SO2', 'CO',
  'Proximity_to_Industrial_Areas', 'Population_Density'
]

# X are the attributes we use to predict Y
X = df[attributes]
y = df['Air Quality']

# Run the Decision Tree from sklearn
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Plor the tree with matplotlib
tree.plot_tree(dtree, feature_names=features)

# Try to predict a new entry (Should result in 'Moderate' or '2')
print(dtree.predict([[30, 60, 5, 18, 19, 9, 1.7, 6, 330]]))
