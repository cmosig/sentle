from io import StringIO

import fiona
import geopandas as gpd
import pandas as pd

fiona.drvsupport.supported_drivers['KML'] = "rw"

df = gpd.read_file("../data/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml")

df = df[["Name", "geometry", "Description"]]

df["crs"] = df["Description"].apply(lambda desc: f"EPSG:{pd.read_html(StringIO(desc))[0][1].iloc[1]}")

df = df.rename(columns=dict(Name="name"))

print(df)

df[["name", "geometry", "crs"]].to_file("../data/sentinel2_grid_stripped_with_epsg.gpkg")
