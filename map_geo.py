import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#
lcd_df = pd.read_csv('./data/seoul_bike.csv')
lcd_df.info()
lcd_df

fig = px.scatter_mapbox(
lcd_df,
lat="lat",
lon="long",
size="LCD거치대수",
color="자치구",
hover_name="대여소명", # 마우스 오버 시 표시한 텍스트
hover_data={"lat": False, "long": False, "LCD거치대수": True, "자치구": True},
text="text",
 zoom=11,
 height=650,
 );
 # carto-positron : 무료, 지도 배경 스타일 지정
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
fig.show();


pd.set_option('display.max_columns', None)
import geopandas as gpd
gdf = gpd.read_file("./data/서울시군구/TL_SCCO_SIG_W.shp")
gdf = gdf.to_crs(epsg=4326)
gdf.to_file("./data/seoul_districts.geojson", driver="GeoJSON")

import json
with open('./data/seoul_districts.geojson', encoding='utf-8') as f:geojson_data = json.load(f)
print(geojson_data.keys())
geojson_data['features'][0]
geo_list = geojson_data['features'][2]['geometry']['coordinates']
len(geo_list)

# 산점도 x, y 좌표
x = np.array(geo_list[0])[:,0]
y = np.array(geo_list[0])[:,1]


# 산점도 (Scatter Plot) 생성
fig = go.Figure();
fig.add_trace(
    go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=10, color="red"),
    name="Map Plot"
    )
);
fig

lcd_df = pd.read_csv('./data/seoul_bike.csv')
print(lcd_df.head())

agg_df = (lcd_df.groupby("자치구",as_index=False)["LCD거치대수"].sum())
agg_df.columns = ["자치구", "LCD합계"]
# 컬럼 이름을 GeoJSON과 맞추기
agg_df = agg_df.rename(columns={"자치구": "SIG_KOR_NM"})
print(agg_df.head(2))

import plotly.express as px
 
fig = px.choropleth_mapbox(

    agg_df,

    geojson=geojson_data,

    locations="SIG_KOR_NM",

    featureidkey="properties.SIG_KOR_NM",

    color="LCD합계",

    color_continuous_scale="Blues",

    mapbox_style="carto-positron",

    center={"lat": 37.5665, "lon": 126.9780},

    zoom=10,

    opacity=0.7,

    title="서울시 자치구별 LCD 거치대 수"

)
 
fig.update_layout(

    margin={"r": 0, "t": 30, "l": 0, "b": 0}

)
 
fig.show()

 
