# GeoPandas Complete Reference Card

## Installation & Import

```python
pip install geopandas
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
```

## Core Data Structures

### GeoDataFrame

- Main data structure: extends pandas DataFrame with geometry column
- Each row represents a geographic feature
- Geometry column contains shapely objects (Point, LineString, Polygon, etc.)

### GeoSeries

- One-dimensional array of geometries
- Similar to pandas Series but for geographic data

## Reading & Writing Data

### Reading Formats

```python
# Shapefile
gdf = gpd.read_file('data.shp')

# GeoJSON
gdf = gpd.read_file('data.geojson')

# PostGIS Database
gdf = gpd.read_postgis(sql, connection, geom_col='geom')

# From URL
gdf = gpd.read_file('https://example.com/data.geojson')

# From pandas DataFrame
df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'name': ['A', 'B']})
geometry = [Point(xy) for xy in zip(df.x, df.y)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)
```

### Writing Formats

```python
# Shapefile
gdf.to_file('output.shp')

# GeoJSON
gdf.to_file('output.geojson', driver='GeoJSON')

# PostGIS
gdf.to_postgis('table_name', connection, if_exists='replace')

# Other formats
gdf.to_file('output.gpkg', driver='GPKG')  # GeoPackage
gdf.to_file('output.kml', driver='KML')    # KML
```

## Coordinate Reference Systems (CRS)

```python
# Check current CRS
print(gdf.crs)

# Set CRS (if unknown)
gdf = gdf.set_crs('EPSG:4326')

# Transform CRS
gdf_utm = gdf.to_crs('EPSG:32633')  # UTM Zone 33N
gdf_web = gdf.to_crs('EPSG:3857')   # Web Mercator

# Transform to custom CRS
gdf_custom = gdf.to_crs('+proj=aea +lat_1=29.5 +lat_2=45.5')
```

## Geometry Operations

### Basic Properties

```python
# Geometry column access
gdf.geometry
gdf['geometry']

# Geometry properties
gdf.geometry.area          # Area of polygons
gdf.geometry.length        # Length of lines/perimeter of polygons
gdf.geometry.bounds        # Bounding box coordinates
gdf.total_bounds           # Total bounds of all geometries
gdf.geometry.centroid      # Centroid points
gdf.geometry.representative_point()  # Point guaranteed inside polygon
gdf.geometry.exterior      # Exterior boundary
gdf.geometry.interiors     # Interior holes
```

### Geometric Predicates (Returns Boolean)

```python
# Spatial relationships
gdf1.geometry.intersects(gdf2)    # Geometries intersect
gdf1.geometry.contains(gdf2)      # gdf1 contains gdf2
gdf1.geometry.within(gdf2)        # gdf1 is within gdf2
gdf1.geometry.touches(gdf2)       # Geometries touch boundaries
gdf1.geometry.crosses(gdf2)       # Geometries cross
gdf1.geometry.overlaps(gdf2)      # Geometries overlap
gdf1.geometry.equals(gdf2)        # Geometries are equal
gdf1.geometry.covers(gdf2)        # gdf1 covers gdf2
gdf1.geometry.covered_by(gdf2)    # gdf1 is covered by gdf2
```

### Geometric Operations (Returns New Geometries)

```python
# Set operations
gdf.geometry.union(other)         # Union of geometries
gdf.geometry.intersection(other)  # Intersection
gdf.geometry.difference(other)    # Difference
gdf.geometry.symmetric_difference(other)  # Symmetric difference

# Geometric transformations
gdf.geometry.buffer(distance)     # Buffer around geometries
gdf.geometry.simplify(tolerance)  # Simplify geometries
gdf.geometry.convex_hull          # Convex hull
gdf.geometry.envelope             # Bounding box as polygon
gdf.geometry.boundary             # Boundary of geometry

# Distance operations
gdf.geometry.distance(other)      # Distance between geometries
```

## Complete Methods Reference Table

|Method                |Description          |Example                                        |
|----------------------|---------------------|-----------------------------------------------|
|**Data Access & Info**|                     |                                               |
|`head(n)`             |First n rows         |`gdf.head(5)`                                  |
|`tail(n)`             |Last n rows          |`gdf.tail(5)`                                  |
|`info()`              |DataFrame info       |`gdf.info()`                                   |
|`describe()`          |Statistical summary  |`gdf.describe()`                               |
|`shape`               |Dimensions           |`gdf.shape`                                    |
|`columns`             |Column names         |`gdf.columns`                                  |
|`dtypes`              |Data types           |`gdf.dtypes`                                   |
|**CRS Operations**    |                     |                                               |
|`crs`                 |Current CRS          |`gdf.crs`                                      |
|`set_crs()`           |Set CRS              |`gdf.set_crs('EPSG:4326')`                     |
|`to_crs()`            |Transform CRS        |`gdf.to_crs('EPSG:3857')`                      |
|`estimate_utm_crs()`  |Estimate UTM zone    |`gdf.estimate_utm_crs()`                       |
|**Spatial Operations**|                     |                                               |
|`overlay()`           |Spatial overlay      |`gpd.overlay(gdf1, gdf2, how='intersection')`  |
|`sjoin()`             |Spatial join         |`gpd.sjoin(gdf1, gdf2, predicate='intersects')`|
|`sjoin_nearest()`     |Join to nearest      |`gpd.sjoin_nearest(gdf1, gdf2)`                |
|`clip()`              |Clip geometries      |`gdf.clip(boundary)`                           |
|`dissolve()`          |Dissolve by attribute|`gdf.dissolve(by='column')`                    |
|`explode()`           |Multi-part to single |`gdf.explode()`                                |
|**Geometry Access**   |                     |                                               |
|`geometry`            |Geometry column      |`gdf.geometry`                                 |
|`set_geometry()`      |Set active geometry  |`gdf.set_geometry('new_geom')`                 |
|`rename_geometry()`   |Rename geometry col  |`gdf.rename_geometry('geom')`                  |
|**I/O Operations**    |                     |                                               |
|`read_file()`         |Read spatial file    |`gpd.read_file('data.shp')`                    |
|`to_file()`           |Write spatial file   |`gdf.to_file('output.geojson')`                |
|`read_postgis()`      |Read from PostGIS    |`gpd.read_postgis(sql, conn)`                  |
|`to_postgis()`        |Write to PostGIS     |`gdf.to_postgis('table', conn)`                |
|**Visualization**     |                     |                                               |
|`plot()`              |Plot geometries      |`gdf.plot(column='pop', legend=True)`          |
|`explore()`           |Interactive map      |`gdf.explore()`                                |
|**Aggregation**       |                     |                                               |
|`groupby()`           |Group by column      |`gdf.groupby('region').sum()`                  |
|`agg()`               |Custom aggregation   |`gdf.agg({'pop': 'sum', 'area': 'mean'})`      |

## Spatial Joins

### Types of Spatial Joins

```python
# Basic spatial join
joined = gpd.sjoin(points_gdf, polygons_gdf, predicate='within')

# Spatial join with different predicates
joined_intersects = gpd.sjoin(gdf1, gdf2, predicate='intersects')
joined_contains = gpd.sjoin(gdf1, gdf2, predicate='contains')

# Join to nearest geometry
nearest = gpd.sjoin_nearest(points_gdf, polygons_gdf)

# Join with distance
nearest_with_dist = gpd.sjoin_nearest(points_gdf, polygons_gdf, distance_col='distance')
```

### Join Types

```python
# Left join (default)
left_join = gpd.sjoin(gdf1, gdf2, how='left')

# Inner join
inner_join = gpd.sjoin(gdf1, gdf2, how='inner')

# Right join  
right_join = gpd.sjoin(gdf1, gdf2, how='right')
```

## Spatial Overlay

### Overlay Types

```python
# Intersection - keep only overlapping parts
intersection = gpd.overlay(gdf1, gdf2, how='intersection')

# Union - combine all geometries
union = gpd.overlay(gdf1, gdf2, how='union')

# Difference - remove overlapping parts
difference = gpd.overlay(gdf1, gdf2, how='difference')

# Symmetric difference - non-overlapping parts only
sym_diff = gpd.overlay(gdf1, gdf2, how='symmetric_difference')

# Identity - intersection + difference
identity = gpd.overlay(gdf1, gdf2, how='identity')
```

## Aggregation and Dissolve

```python
# Dissolve by column
dissolved = gdf.dissolve(by='region')

# Dissolve all into single geometry
dissolved_all = gdf.dissolve()

# Dissolve with aggregation
dissolved_agg = gdf.dissolve(by='region', aggfunc={
    'population': 'sum',
    'area': 'mean'
})

# Multi-column dissolve
dissolved_multi = gdf.dissolve(by=['region', 'type'])
```

## Clipping

```python
# Clip to polygon boundary
clipped = gdf.clip(boundary_polygon)

# Clip to another GeoDataFrame
clipped = gdf.clip(boundary_gdf)

# Clip to bounding box
bbox = [-74.05, 40.7, -73.9, 40.8]  # [minx, miny, maxx, maxy]
clipped = gdf.clip_by_rect(bbox[0], bbox[1], bbox[2], bbox[3])
```

## Visualization

### Basic Plotting

```python
# Simple plot
gdf.plot()

# Plot with column coloring
gdf.plot(column='population', legend=True)

# Plot with custom colormap
gdf.plot(column='density', cmap='viridis', legend=True)

# Plot with classification
gdf.plot(column='income', scheme='quantiles', k=5, legend=True)

# Multiple plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
gdf.plot(ax=axes[0], color='blue')
gdf.plot(ax=axes[1], column='pop', legend=True)
```

### Interactive Maps

```python
# Basic interactive map
gdf.explore()

# Interactive map with popup columns
gdf.explore(popup=['name', 'population'])

# Interactive map with styling
gdf.explore(
    column='population',
    cmap='viridis',
    tooltip=['name', 'pop'],
    popup=True,
    tiles='CartoDB positron'
)
```

## Geocoding

```python
# Forward geocoding (address to coordinates)
from geopandas.tools import geocode

addresses = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL']
geocoded = geocode(addresses, provider='nominatim', user_agent='my_app')

# Reverse geocoding (coordinates to address)
from geopandas.tools import reverse_geocode

points = gpd.GeoDataFrame(
    geometry=[Point(-73.9857, 40.7484), Point(-118.2437, 34.0522)]
)
reverse_geocoded = reverse_geocode(points, provider='nominatim', user_agent='my_app')
```

## Working with Different Geometry Types

### Creating Geometries

```python
from shapely.geometry import Point, LineString, Polygon, MultiPoint

# Points
points = [Point(x, y) for x, y in zip([1, 2, 3], [1, 2, 3])]

# Lines
line = LineString([(0, 0), (1, 1), (2, 0)])

# Polygons
polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# Polygon with holes
exterior = [(0, 0), (2, 0), (2, 2), (0, 2)]
hole = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
polygon_with_hole = Polygon(exterior, [hole])
```

### Multi-part Geometries

```python
# Handle multi-part geometries
multipart_gdf = gdf[gdf.geometry.type.isin(['MultiPolygon', 'MultiLineString'])]

# Explode multi-part to single-part
single_part = gdf.explode(index_parts=False)

# Keep track of original features
single_part_with_id = gdf.explode(index_parts=True)
```

## Performance Tips & Best Practices

### Indexing

```python
# Spatial index for faster operations
spatial_index = gdf.sindex

# Use spatial index for intersection
possible_matches_index = list(spatial_index.intersection(query_geometry.bounds))
possible_matches = gdf.iloc[possible_matches_index]
```

### Efficient Operations

```python
# Use vectorized operations
gdf['area_km2'] = gdf.geometry.area / 1e6

# Avoid loops for spatial operations
# Instead of:
# for i, row in gdf.iterrows():
#     result = row.geometry.buffer(100)

# Use:
gdf['buffered'] = gdf.geometry.buffer(100)
```

### Memory Management

```python
# Convert to appropriate CRS before area calculations
gdf_projected = gdf.to_crs('EPSG:3857')  # Web Mercator for area
areas = gdf_projected.geometry.area

# Use categorical data for string columns
gdf['region'] = gdf['region'].astype('category')
```

## Common Workflows

### Workflow 1: Point in Polygon Analysis

```python
# Load data
points = gpd.read_file('points.shp')
polygons = gpd.read_file('polygons.shp')

# Ensure same CRS
points = points.to_crs(polygons.crs)

# Spatial join
result = gpd.sjoin(points, polygons, predicate='within')

# Count points per polygon
counts = result.groupby('index_right').size()
polygons['point_count'] = polygons.index.map(counts).fillna(0)
```

### Workflow 2: Buffer Analysis

```python
# Create buffers
buffered = gdf.geometry.buffer(1000)  # 1km buffer

# Find intersections with buffers
intersections = gpd.overlay(other_gdf, gpd.GeoDataFrame(geometry=buffered), 
                           how='intersection')

# Calculate area of intersection
intersections['intersection_area'] = intersections.geometry.area
```

### Workflow 3: Network Analysis Preparation

```python
# Prepare line network
lines = gpd.read_file('roads.shp')

# Ensure valid geometries
lines = lines[lines.is_valid]

# Get endpoints
lines['start_point'] = lines.geometry.apply(lambda x: Point(x.coords[0]))
lines['end_point'] = lines.geometry.apply(lambda x: Point(x.coords[-1]))

# Create nodes GeoDataFrame
start_points = gpd.GeoDataFrame(geometry=lines['start_point'])
end_points = gpd.GeoDataFrame(geometry=lines['end_point'])
nodes = pd.concat([start_points, end_points]).drop_duplicates()
```

## Error Handling & Validation

```python
# Check for valid geometries
invalid = gdf[~gdf.is_valid]

# Fix invalid geometries
gdf['geometry'] = gdf.geometry.buffer(0)

# Check for empty geometries
empty = gdf[gdf.is_empty]

# Remove empty geometries
gdf = gdf[~gdf.is_empty]

# Check CRS
if gdf.crs is None:
    print("CRS not defined")
    gdf = gdf.set_crs('EPSG:4326')  # Assume WGS84

# Handle topology errors
from shapely.validation import make_valid
gdf['geometry'] = gdf.geometry.apply(make_valid)
```

## Integration with Other Libraries

### With Folium

```python
import folium

# Create base map
m = folium.Map([40.7, -74.0], zoom_start=10)

# Add GeoDataFrame to map
folium.GeoJson(gdf).add_to(m)

# Add with popup
folium.GeoJson(
    gdf,
    popup=folium.GeoJsonPopup(fields=['name', 'population'])
).add_to(m)
```

### With Contextily (Basemaps)

```python
import contextily as ctx

# Plot with basemap
ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.Stamen.TonerLite)
```

### With Rasterio (Raster Data)

```python
import rasterio
from rasterio.mask import mask

# Clip raster with polygon
with rasterio.open('raster.tif') as src:
    out_image, out_transform = mask(src, gdf.geometry, crop=True)
```

This reference card covers the essential GeoPandas functionality for geospatial data analysis. Keep it handy for quick reference during your spatial data projects!