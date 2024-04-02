import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
data = pd.read_csv('Fertilizer Prediction.csv')

# Changing column names
data.rename(columns={'Humidity ': 'Humidity', 'Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer'}, inplace=True)

# Encoding categorical variables
encode_soil = LabelEncoder()
data['Soil_Type'] = encode_soil.fit_transform(data['Soil_Type'])

encode_crop = LabelEncoder()
data['Crop_Type'] = encode_crop.fit_transform(data['Crop_Type'])

encode_ferti = LabelEncoder()
data['Fertilizer'] = encode_ferti.fit_transform(data['Fertilizer'])

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer', axis=1), data['Fertilizer'], test_size=0.2, random_state=1)

# Training a RandomForestClassifier
params = {
    'n_estimators': [350, 400, 450],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 8]
}

rand = RandomForestClassifier()
grid_rand = GridSearchCV(rand, params, cv=3, verbose=3, n_jobs=-1)
grid_rand.fit(x_train, y_train)

# Pickling the trained model
pickle_out = open('classifier.pkl', 'wb')
pickle.dump(grid_rand, pickle_out)
pickle_out.close()

# Pickling the label encoder for Fertilizer
pickle_out = open('fertilizer.pkl', 'wb')
pickle.dump(encode_ferti, pickle_out)
pickle_out.close()

# Example prediction using the trained model
model = pickle.load(open('classifier.pkl', 'rb'))
ferti = pickle.load(open('fertilizer.pkl', 'rb'))

# Example prediction
example_data = [[34, 67, 62, 0, 1, 7, 0, 30]]  # Example input data
predicted_fertilizer_index = model.predict(example_data)[0]
predicted_fertilizer = ferti.classes_[predicted_fertilizer_index]

print("Predicted Fertilizer:", predicted_fertilizer)
