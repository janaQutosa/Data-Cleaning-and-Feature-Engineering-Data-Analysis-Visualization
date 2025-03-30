# Shiyar Mohammad 1210766
# Jana Qutosa 1210331

import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import geopandas as gpd
import contextily as ctx
import seaborn as sns

# Load the dataset 
df = pd.read_csv('Electric_Vehicle_Population_Data.csv')
print(df.columns.tolist())
# Check for missing values 
missing_values = df.isnull().sum()  # Count missing values in each column
missing_percentage = (missing_values / len(df)) * 100  # Calculate the percentage of missing values

# Create a DataFrame to display missing values and their percentage
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

# Filter out columns without missing values for a cleaner output
missing_data_summary = missing_data_summary[missing_data_summary['Missing Values'] > 0]

# Display the summary
print("Missing Values Summary:")
print(missing_data_summary)

#_______________________________PART- 2 - ________________________________________
print("")
print("*******************************************")
print("___Missing Value Strategies___")
print("")

#The shape of the original DataFrame
print("Original Dataset Shape:", df.shape)

#Strategy 1: Dropping rows with missing values
df_dropped = df.dropna()
print("After Dropping Rows with Missing Values:", df_dropped.shape)

# Strategy 2: Mean Imputation for Numerical Features
# Separately handle numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute missing values in numerical columns with mean
df_mean = df.copy()
df_mean[numerical_cols] = df_mean[numerical_cols].fillna(df[numerical_cols].mean())
print("")
print("After applying Mean Strategy on Numerical Features:")
print("")
print(df_mean[numerical_cols].isnull().sum())  # Should be 0 if all missing values are filled

# Strategy 3: Mode Imputation for Categorical Features
df_mode = df.copy()
for col in categorical_cols:
    df_mode[col].fillna(df[col].mode()[0], inplace=True)

print("")
print("After applying Mode Strategy on Categorical Features:")
print("")
print(df_mode[categorical_cols].isnull().sum())  # Should be 0 if all missing values are filled



#__________________________________PART- 3 - ________________________________________
# Feature Encoding using one-hot encoding on the cleaned dataset
print("")
print("*******************************************")
print("___Feature Encoding___")
print("")
 
# We are encoding the categorical columns from df_dropped using one-hot encoding
df_encoded = pd.get_dummies(df_dropped, columns=categorical_cols, drop_first=True)
# Convert boolean values to integers (0 and 1)
df_encoded = df_encoded.astype(int)

print("After One-Hot Encoding, the shape of the dataset is:", df_encoded.shape)
print(df_encoded.head())  # Display the first few rows to see the encoded features

#_________________________________PART- 4 - ________________________________________
print("")
print("*******************************************")
print("___Normalization___")
print("")

# Initialize scalers
standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

# Apply Standard Scaling to df_dropped
df_standard_scaled = df_dropped.copy()
df_standard_scaled[numerical_cols] = standard_scaler.fit_transform(df_dropped[numerical_cols])

print("After Standard Scaling:")
print(df_standard_scaled[numerical_cols].head())

# Apply Min-Max Scaling to df_dropped
df_min_max_scaled = df_dropped.copy()
df_min_max_scaled[numerical_cols] = min_max_scaler.fit_transform(df_dropped[numerical_cols])

print("\nAfter Min-Max Scaling:")
print(df_min_max_scaled[numerical_cols].head())

#__________________________________PART- 5 - ________________________________________
print("")
print("*******************************************")
print("___Descriptive Statistics___")
print("")

# Separately identify numerical columns 
numerical_cols_for_stat = df_dropped.select_dtypes(include=['float64', 'int64']).columns
# Calculate summary statistics for each numerical column
summary_stats = pd.DataFrame({
    'Mean': df_dropped[numerical_cols_for_stat].mean(),
    'Median': df_dropped[numerical_cols_for_stat].median(),
    'Standard Deviation': df_dropped[numerical_cols_for_stat].std(),
    'Min': df_dropped[numerical_cols_for_stat].min(),
    'Max': df_dropped[numerical_cols_for_stat].max()
})

# Display the summary statistics
print("Summary Statistics for Numerical Features:")
print(summary_stats)

#__________________________________PART- 6 - ________________________________________
# print("")
# print("*******************************************")
# print("___Spatial Distribution___")
# print("")


# Handle missing or non-string values in 'Vehicle Location'
df['Longitude'] = df['Vehicle Location'].apply(
    lambda x: float(x.split()[1][1:]) if isinstance(x, str) else None
)
df['Latitude'] = df['Vehicle Location'].apply(
    lambda x: float(x.split()[2][:-1]) if isinstance(x, str) else None
)

# Drop rows with missing longitude or latitude values
df = df.dropna(subset=['Longitude', 'Latitude'])

# Convert to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")

# Set up the plot with a basemap
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='blue', markersize=5, label='EV Locations')

# Add the basemap using OpenStreetMap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf.crs.to_string())

plt.title("Spatial Distribution of Electric Vehicles in Washington State", fontsize=15)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()


#__________________________________PART- 7 - ________________________________________
print("")
print("*******************************************")
print("___Model Popularity___")
print("")


# Count the number of registrations for each EV model
model_counts = df_dropped['Model'].value_counts()

# Create a DataFrame 
model_counts_df = pd.DataFrame(model_counts).reset_index()
model_counts_df.columns = ['Model', 'Count']


#Get the top 15 popular models
top_models = model_counts.head(15)

#Calculate the count of all other models
other_count = model_counts.sum() - top_models.sum()

#Prepare data for the pie chart
data_for_pie = top_models.tolist() + [other_count]
labels_for_pie = top_models.index.tolist() + ['Others']

# Define custom colors for the pie chart
colors = plt.cm.tab20c(range(len(data_for_pie) - 1))  # Colors for top 15 models
colors = list(colors) + ['#ff9999']  

#Create a pie chart
plt.figure(figsize=(10, 10))
plt.pie(data_for_pie, labels=labels_for_pie, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Top 15 Popular Electric Vehicle Models in Washington', pad=15)
plt.axis('equal') 
plt.show()

# If the dataset has a year column, analyze trends over time
if 'Model Year' in df_dropped.columns:
    year_model_counts = df_dropped.groupby(['Model Year', 'Model']).size().unstack().fillna(0)

    plt.figure(figsize=(14, 8))
    year_model_counts.plot(kind='bar', stacked=True, colormap='tab20', figsize=(14, 8))
    plt.title('EV Model Popularity Over Years')
    plt.xlabel('Model Year')
    plt.ylabel('Number of Registrations')
    plt.legend(title='EV Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() 
    plt.show()
else:
    print("No 'Model Year' column found for trend analysis.")
    
    
#__________________________________PART- 8 - ________________________________________
print("")
print("*******************************************")
print("___relationship between every pair of numeric features___")
print("")

# Identify numeric features
numeric_features = df_dropped.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_features.corr()

#  Visualize the correlation matrix 
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numeric Features', pad=20)
plt.show()

#__________________________________PART- 9 - ________________________________________
print("")
print("*******************************************")
print("___Data Exploration Visualizations___")
print("")

sns.set(style="whitegrid")

# Histograms for Numeric Features
numeric_features = df_dropped.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(14, 10))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)  # Adjust the grid size based on the number of features
    sns.histplot(df_dropped[feature], bins=30, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout(pad=3.0)  
plt.subplots_adjust(hspace=0.4, wspace=0.4) 
plt.show()

# Scatter Plots 
# relationship between Electric Range and Model Year
plt.figure(figsize=(10, 12))
sns.scatterplot(data=df_dropped, x='Model Year', y='Electric Range', hue='Make', alpha=0.7)
plt.title('Electric Range vs. Model Year')
plt.xlabel('Model Year')
plt.ylabel('Electric Range (miles)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Make')
plt.tight_layout()
plt.show()

# Count Plot for Categorical Data
plt.figure(figsize=(15, 6))
sns.countplot(data=df_dropped, x='Make', order=df_dropped['Make'].value_counts().index)
plt.title('Count of Electric Vehicles by Make')
plt.xlabel('Make')
plt.ylabel('Count')
plt.xticks(rotation=45)
# add space between the labels
plt.xlim(-0.5, len(df_dropped['Make'].value_counts()) - 0.5)
plt.tight_layout()
plt.show()



#__________________________________PART- 10 - ________________________________________
print("")
print("*******************************************")
print("___Comparative Visualization___")
print("")

# Grouping the data by 'City' to count the number of EVs in each city
city_counts = df_dropped['City'].value_counts().reset_index()
city_counts.columns = ['City', 'EV Count']

# Plotting the distribution of EVs across different cities
plt.figure(figsize=(12, 8))
sns.barplot(x='EV Count', y='City', data=city_counts.head(20))  # Show top 20 cities by EV count
plt.title('Distribution of EVs Across Different Cities (Top 20)')
plt.xlabel('Number of EVs')
plt.ylabel('City')
plt.show()

# Grouping the data by 'County' to count the number of EVs in each county
county_counts = df_dropped['County'].value_counts().reset_index()
county_counts.columns = ['County', 'EV Count']

# Plotting the distribution of EVs across different counties
plt.figure(figsize=(12, 8))
sns.barplot(x='EV Count', y='County', data=county_counts.head(20))  # Show top 20 counties by EV count
plt.title('Distribution of EVs Across Different Counties (Top 20)')
plt.xlabel('Number of EVs')
plt.ylabel('County')
plt.show()

#__________________________________PART- 11 - ________________________________________
print("")
print("*******************************************")
print("___Temporal Analysis___")
print("")

# Ensure 'Model Year' is in integer format
df_dropped['Model Year'] = df_dropped['Model Year'].astype(int)

# Grouping by 'Model Year' to count the number of EVs
adoption_rates = df_dropped['Model Year'].value_counts().sort_index().reset_index()
adoption_rates.columns = ['Model Year', 'EV Count']

# Plotting EV Adoption Rates Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Model Year', y='EV Count', data=adoption_rates, marker='o')
plt.title('EV Adoption Rates Over Time')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Grouping by 'Model Year' and 'Model' to count the number of EVs per model per year
model_popularity = df_dropped.groupby(['Model Year', 'Make', 'Model']).size().reset_index(name='EV Count')

# Plotting the popularity of different models over the years
plt.figure(figsize=(14, 8))
sns.lineplot(data=model_popularity, x='Model Year', y='EV Count', hue='Model', marker='o')
plt.title('Model Popularity Over Time')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()