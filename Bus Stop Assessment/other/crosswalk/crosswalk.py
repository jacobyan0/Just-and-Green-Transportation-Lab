import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString
import requests
import json
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import geopandas as gpd

# Step 1: 输入公交站的经纬度
bus_stop_lat = 29.6390542733032  # 示例纬度
bus_stop_lon = -82.3461411818786  # 示例经度
bus_stop_point = Point(bus_stop_lon, bus_stop_lat)

# Step 2: 获取OSM中的road edges和nodes
G = ox.graph_from_point((bus_stop_lat, bus_stop_lon), dist=100, network_type='all')
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)

# Step 3: 计算最近的road edge
def calculate_distance_to_edges(point, edges):
    distances = edges['geometry'].apply(lambda x: point.distance(x))
    nearest_edge_idx = distances.idxmin()
    nearest_edge = edges.loc[nearest_edge_idx]
    return nearest_edge

nearest_edge = calculate_distance_to_edges(bus_stop_point, edges)

# 打印 nearest_edge 的结构，查看可以使用的字段
print(nearest_edge)

# Step 3.1: 从 geometry 获取最近的 road edge 的起点和终点
geometry = nearest_edge['geometry']
start_node, end_node = geometry.coords[0], geometry.coords[-1]  # 获取LineString的起点和终点

# Step 3.2: 使用 start_node 和 end_node 查询 nodes 数据框，获取它们的经纬度
start_latlon = (start_node[1], start_node[0])  # (lat, lon)
end_latlon = (end_node[1], end_node[0])        # (lat, lon)

print(f"Start Node Lat/Lon: {start_latlon}")
print(f"End Node Lat/Lon: {end_latlon}")

# Step 4: 使用Google Street View API获取四张fov=90的图像，覆盖360度
api_key_gsv = 'YOUR_API_KEY'  # 需要提供你的Google Street View API密钥
gsv_base_url = "https://maps.googleapis.com/maps/api/streetview"

params = [{
    'size': '640x640',  # 图片大小
    'location': f'{bus_stop_lat},{bus_stop_lon}',  # 图片位置
    'heading': heading,  # 设置视角方向
    'fov': '90',  # 视野角度
    'pitch': '0',  # 视角仰角
    'key': api_key_gsv
} for heading in [0, 90, 180, 270]]  # 四个方向

images = []
for param in params:
    response = requests.get(gsv_base_url, params=param)
    img = Image.open(BytesIO(response.content))
    images.append(img)

# Step 5: 调用Roboflow的API进行跨步道识别
api_key_rf = 'YOUR_ROBOFLOW_API_KEY'  # 填写你的Roboflow API密钥
model_id = 'crosswalk-pfps1'  # 模型ID
version_number = 2  # 模型版本

rf_url = f"https://detect.roboflow.com/{model_id}/{version_number}?api_key={api_key_rf}"

# Function to send image for inference
def infer_image(image):
    # Convert image to JPEG for sending to the API
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = buffered.getvalue()

    response = requests.post(
        rf_url,
        files={"file": img_str},
        data={"confidence": "0.4", "overlap": "0.5"}
    )
    return response.json()

# 绘制推理结果的函数
def draw_predictions_on_image(image, predictions):
    for pred in predictions:
        x, y = int(pred["x"]), int(pred["y"])
        width, height = int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        confidence = pred["confidence"]

        # 计算左上角和右下角的坐标
        top_left = (x - width // 2, y - height // 2)
        bottom_right = (x + width // 2, y + height // 2)

        # 在图像上绘制边界框
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        # 在图像上绘制标签
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Step 6: 进行推理并解析结果
for idx, img in enumerate(images):
    img_np = np.array(img)  # 将PIL图像转换为NumPy数组
    result = infer_image(img)
    predictions = result.get("predictions", [])

    # 绘制推理结果
    annotated_image = draw_predictions_on_image(img_np, predictions)

    # 保存带标注的图像到当前路径
    output_image_file = f"annotated_image_{idx + 1}.jpg"
    cv2.imwrite(output_image_file, annotated_image)
    print(f"Annotated image saved as {output_image_file}")

# Step 7: 绘制地图并标注公交车站、edges和nodes
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制100米范围内的道路edges，颜色改为深黄色，zorder设置为1以确保在线条的最底部
edges.plot(ax=ax, color='orange', linewidth=1, label='100m range edges', zorder=1)

# 绘制所有节点，zorder为2，确保显示在edges之上
nodes.plot(ax=ax, color='blue', markersize=5, label='nodes', zorder=2)

# 绘制公交站位置，zorder为5，显示在最上面
ax.scatter(bus_stop_lon, bus_stop_lat, c='red', s=100, label='Bus stop', zorder=5)

# 将最近的edge转换为GeoDataFrame来进行绘制，zorder为3确保在线条的上方但低于公交站点
nearest_edge_gdf = gpd.GeoDataFrame(geometry=[nearest_edge['geometry']], crs=edges.crs)
nearest_edge_gdf.plot(ax=ax, color='pink', linewidth=2, label='Nearest edge', zorder=3)

# 在最近的edge上标注起点和终点，zorder为4确保显示在节点之上但低于公交站点
ax.scatter(start_node[0], start_node[1], c='purple', s=100, label='Start node', zorder=4)
ax.scatter(end_node[0], end_node[1], c='purple', s=100, label='End node', zorder=4)

# 设置图例和标题
ax.legend()
ax.set_title("Bus Stop with Edges and Nodes", fontsize=15)

# 保存地图
plt.savefig('bus_stop_map.png')
plt.show()
