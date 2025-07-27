import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import folium

# Read the Excel file
try:
    df = pd.read_excel('LOCATIONS.xlsx', sheet_name='Sheet1')
except FileNotFoundError:
    print("Error: LOCATIONS.xlsx not found. Please check the file path.")
    exit(1)
except ValueError:
    print("Error: Sheet 'Sheet1' not found or columns missing. Check sheet name and column names.")
    exit(1)

# Verify data
print("Columns in DataFrame:", df.columns)
print("Missing values:", df[['Latitude', 'Longitude']].isna().sum())
print("Data types:", df[['Latitude', 'Longitude']].dtypes)

# Drop rows with missing or non-numeric coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])
df = df[df['Latitude'].apply(lambda x: isinstance(x, (int, float)))]
df = df[df['Longitude'].apply(lambda x: isinstance(x, (int, float)))]

# Perform K-means clustering
coords = df[['Latitude', 'Longitude']].values
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(coords)
centroids = kmeans.cluster_centers_

# Debug clustering
print("Centroids:", centroids)
print("Centroids shape:", centroids.shape)
print("Cluster values:", df['Cluster'].unique())
print("Any NaN in Cluster:", df['Cluster'].isna().sum())

# Ensure Cluster is integer and no NaN
df = df[df['Cluster'].notna()]
df['Cluster'] = df['Cluster'].astype(int)

# Calculate Distance_to_Centroid (vectorized approach)
coords = df[['Latitude', 'Longitude']].values
cluster_indices = df['Cluster'].values
centroid_coords = centroids[cluster_indices]
df['Distance_to_Centroid'] = np.array([
    geodesic(coord, centroid).kilometers
    for coord, centroid in zip(coords, centroid_coords)
])

# Validate data for mapping
if not all(col in df.columns for col in ['Latitude', 'Longitude', 'Site_id', 'Cluster']):
    raise ValueError("Required columns (Latitude, Longitude, Site_id, Cluster) missing in DataFrame")
if df[['Latitude', 'Longitude']].isna().any().any():
    raise ValueError("NaN values found in Latitude or Longitude")
if not df['Cluster'].dtype == np.int64:
    raise ValueError(f"Cluster column must be integer, got {df['Cluster'].dtype}")

# Define map center and create Folium map
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=10, tiles='OpenStreetMap')

# Define a distinct color palette for clusters
colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFA500']  # Red, Green, Blue, Magenta, Orange
if len(colors) < len(df['Cluster'].unique()):
    raise ValueError("Not enough colors defined for the number of clusters")

# Add site markers
for idx, row in df.iterrows():
    cluster = int(row['Cluster'])  # Ensure integer for indexing colors
    if cluster not in range(len(colors)):
        continue  # Skip invalid cluster indices
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6,
        color=colors[cluster],
        fill=True,
        fill_color=colors[cluster],
        fill_opacity=0.8,
        popup=f"Site ID: {row['Site_id']}<br>Cluster: {cluster}<br>Distance: {row['Distance_to_Centroid']:.2f} km",
        tooltip=f"Site {row['Site_id']}"
    ).add_to(m)

# Add centroid markers
for i, centroid in enumerate(centroids):
    if i >= len(colors):
        break  # Prevent index errors
    folium.Marker(
        location=[centroid[0], centroid[1]],
        icon=folium.Icon(color='black', icon='star', prefix='fa'),
        popup=f"Cluster {i} Centroid<br>Lat: {centroid[0]:.6f}<br>Lon: {centroid[1]:.6f}"
    ).add_to(m)

# Save the map
m.save('site_clusters_map.html')
print("Map saved as 'site_clusters_map.html'. Open in a browser, adjust view, and screenshot to save as PNG.")

# Save to CSV
output_csv = 'site_clusters.xlsx'
df.to_excel(output_csv, index=False)
print(f"CSV file '{output_csv}' generated with cluster assignments and distances.")