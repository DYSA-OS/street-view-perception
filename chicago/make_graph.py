import pandas as pd
import folium

df_origin = pd.read_csv('origin_chicago.csv')[['lat', 'long']]
df_walk = pd.read_csv('walk_chicago.csv')[['lat', 'long']]

map_center = [41.8781, -87.6298]

mymap = folium.Map(location=map_center, zoom_start=11, tiles='CartoDB positron')

for idx, row in df_origin.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=2,
        color='orange',
        fill=True,
        fill_color='orange'
    ).add_to(mymap)

for idx, row in df_walk.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=2,
        color='green',
        fill=True,
        fill_color='green'
    ).add_to(mymap)

mymap.save('chicago_map.html')
