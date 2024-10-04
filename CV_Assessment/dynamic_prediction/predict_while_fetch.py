import dis
from email import header
import math
from re import T
import tempfile
import requests
import shutil
import os
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import io
import glob
import csv
from ultralytics import YOLO
import supervision as sv
from collections import Counter
from haversine import haversine
import cv2



counter = 0
DOWNLOAD_LIMIT = 1000
total_counts = Counter()


# class
classes_of_interest = ['bench', 'house', 'large bike rack', 'shelter', 'sign', 'small bike rack']
classes_most_important = ['shelter']

# load model
model = YOLO('./best.pt')

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def enhance_contrast_color(image):
    channels = cv2.split(image)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    return cv2.merge(eq_channels)

def super_resolution(image_path):

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel('./ESPCN_x2.pb')
    sr.setModel('espcn', 2)  


    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image {image_path}")
        return
    print("here"+image_path)

    result = sr.upsample(image)


    cv2.imwrite(image_path, result)
    print(f"Processed and saved over original image: {image_path}")



def count_distance(coord1, coord2):

    return haversine(coord1, coord2) * 1000


def calculate_adjacent_coordinates(lat, lon, distance=10):

    olat = lat
    olon = lon
    R = 6378137  # Earth’s radius
    dLat = distance/R
    dLon = distance/(R * math.cos(math.pi * lat / 180))

    north = lat + dLat * 180/math.pi
    south = lat - dLat * 180/math.pi
    east = lon + dLon * 180/math.pi
    west = lon - dLon * 180/math.pi

    return {"north": (north, lon), "south": (south, lon), "east": (lat, east), "west": (lat, west)}

def get_pano_metadata(lat, lon, api_key):

    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&key={api_key}&size=640*320"
    response = requests.get(url)
    data = response.json()
    # print(data)
    if 'pano_id' in data and 'location' in data:
        return data['pano_id'], (data['location']['lat'], data['location']['lng'])
    else:
        return None, None

def calculate_heading(from_lat, from_lon, to_lat, to_lon):

    from_lat = math.radians(from_lat)
    from_lon = math.radians(from_lon)
    to_lat = math.radians(to_lat)
    to_lon = math.radians(to_lon)

    delta_lon = to_lon - from_lon

    x = math.sin(delta_lon) * math.cos(to_lat)
    y = math.cos(from_lat) * math.sin(to_lat) - (math.sin(from_lat) * math.cos(to_lat) * math.cos(delta_lon))
    heading = math.atan2(x, y)
    heading = math.degrees(heading)
    heading = (heading + 360) % 360

    return heading

def blur_specific_area(image, area):

    blur_area = image.crop(area)

    blur_area = blur_area.filter(ImageFilter.GaussianBlur(5))

    image.paste(blur_area, area)

    return image

def calculate_fov(distance):

    if distance < 10:  
        return 70
    elif distance < 20:
        return 60
    else:
        return 50
    
# 会对当前这个panoid在指定的heading下抓取两次fov   
def save_street_view_image(pano_id, heading, fov, api_key, directory, filename, round_number):
    """
    Save a Google Street View image for a given panoID and heading in the specified directory
    """
    step = 1
    url = f"https://maps.googleapis.com/maps/api/streetview?size=640x320&fov={fov}&pano={pano_id}&heading={heading}&key={api_key}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with Image.open(io.BytesIO(response.content)) as image:
            # 模糊左下角区域
            left_bottom_area = (0, image.height - 22, 63, image.height)
            image = blur_specific_area(image, left_bottom_area)

            # 模糊右下角区域
            right_bottom_area = (image.width - 60, image.height - 25, image.width, image.height)
            image = blur_specific_area(image, right_bottom_area)

            # 保存修改后的图像
            temp_name = filename+'_heading'+str(heading)+'_fov'+str(fov)+'.jpg'
            image.save(os.path.join(directory, temp_name))
            super_resolution(os.path.join(directory, temp_name))
    else:
        print(f"Error: Failed to retrieve image for pano ID {pano_id}")
        return
    

    flag = predict(os.path.join(directory, temp_name))

    if flag == False:
        closer_heading_list = [(heading+fov/4)%360,(heading-fov/4)%360]
        fov = int(fov / 2)
        for heading in closer_heading_list:
            print("no detected, find closer")
            step = step + 1
            url = f"https://maps.googleapis.com/maps/api/streetview?size=640x320&fov={fov}&pano={pano_id}&heading={heading}&key={api_key}"
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                with Image.open(io.BytesIO(response.content)) as image:
   
                    left_bottom_area = (0, image.height - 22, 63, image.height)
                    image = blur_specific_area(image, left_bottom_area)


                    right_bottom_area = (image.width - 60, image.height - 25, image.width, image.height)
                    image = blur_specific_area(image, right_bottom_area)

      
                    temp_name = filename+'_heading'+str(heading)+'_fov'+str(fov)+'.jpg'
                    image.save(os.path.join(directory, temp_name))
                    super_resolution(os.path.join(directory, temp_name))
                    flag = predict(os.path.join(directory, temp_name))
                    
            else:
                print(f"Error: Failed to retrieve image for pano ID {pano_id}")
                return
            if flag:
                break
    return flag, step
    
    

    
def predict(filename):  
    global total_counts
    results = model.predict(filename, save=True, conf=0.9)
    detections = sv.Detections.from_ultralytics(results[0])
    detected_classes = set(detections.class_id)
    flag_any = False

    for class_name in classes_of_interest:
        if classes_of_interest.index(class_name) in detected_classes:
            total_counts[class_name] += 1

    for class_name in classes_most_important:
        if classes_of_interest.index(class_name) in detected_classes:
            flag_any = True
    return flag_any


def one_row(original_coordinates,stop_name, file_name, stop_code, stop_id):
    step = 0
    global total_counts
    total_counts = Counter()
    api_key = 'AIzaSyDvP5c4Xns4m2NvfTM7PNDZxBjccLMdq94'  
    parent_directory = file_name+"_imgs" 
    lat, lon = original_coordinates
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    
    images_directory = os.path.join(parent_directory, f"{stop_id}_{sanitize_filename(stop_name)}")

    if not os.path.exists(images_directory):
        os.makedirs(images_directory)


    # first round
    
    print("Round 1: Original")
    pano_id, pano_coords = get_pano_metadata(*original_coordinates, api_key)
    heading = calculate_heading(*pano_coords, *original_coordinates)
    distance = count_distance(original_coordinates, pano_coords)
    if stop_code:
        filename = f"origin_{lat},{lon}_{stop_code}_{stop_id}"
    else:
        filename = f"origin_{lat},{lon}_{stop_id}"
    fov = calculate_fov(distance)
    result,innerStep = save_street_view_image(pano_id, heading, fov, api_key, images_directory, filename, 1)
    step += innerStep

    if result == False:

        print("Round 2: Look around")


        # In round 2, args: distance is trival

        round2_heading_list = [(heading+fov/2+35)%360, (heading-(fov/2+35))%360]
        for round2_heading in round2_heading_list:
            result,innerStep = save_street_view_image(pano_id, round2_heading, 70, api_key, images_directory, filename, 2)
            step += innerStep
            if result:
                break


        if result == False:
            print("Round 3: Find adjacent coords")
            # Calculate adjacent coordinates
            adjacent_coords = calculate_adjacent_coordinates(*original_coordinates)

            # Get panoIDs, their actual coordinates, and calculate headings for each position
            pano_data = {}
            for direction, coords in adjacent_coords.items():
                pano_id, pano_coords = get_pano_metadata(*coords, api_key)
                if pano_id and pano_id not in pano_data:
                    heading = calculate_heading(*pano_coords, *original_coordinates)
                    distance = count_distance(original_coordinates, pano_coords)
                    pano_data[pano_id] = {"coords": pano_coords, "heading": heading, "move":direction, "distance":distance}

        
            # Retrieve and save images for each panoID
            for pano_id, data in pano_data.items():
                # filename = f"{sanitize_filename(stop_name)}_{data['move']}_{pano_id}_heading_{data['heading']}.jpg"
                if stop_code:
                    filename = f"{data['move']}_{lat},{lon}_{stop_code}_{stop_id}"
                else:
                    filename = f"{data['move']}_{lat},{lon}_{stop_id}"
                
                print(stop_id)
                result, innerStep = save_street_view_image(pano_id, data['heading'], calculate_fov(data['distance']), api_key, images_directory, filename,3)
                step += innerStep
            
    with open(os.path.join(images_directory, 'detection_summary.txt'), 'w') as file:
        for class_name, count in total_counts.items():
            file.write(f'{class_name}: {count}\n')
    
    row = [stop_id, lat, lon] + [total_counts[class_name] for class_name in classes_of_interest] + [step]
    return row


def sanitize_filename(name):
    return "".join([c for c in name if c.isalnum() or c in ' -_().'])

def process_bus_stops_txt(file_path):

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(f'{city_name}_analysis.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['stop_id', 'lat', 'lon'] + classes_of_interest+['step'])

        with open(file_path, 'r') as file:

                headers = file.readline().strip().split(',')
                name_idx = headers.index('stop_name')
                lat_idx = headers.index('stop_lat')
                lon_idx = headers.index('stop_lon')
                if 'stop_code' in headers:
                    code_idx = headers.index('stop_code')
                else:
                    code_idx = None
                id_idx = headers.index('stop_id')

                for line in file:
                    fields = line.strip().split(',')
                    stop_name = fields[name_idx]
                    stop_lat = float(fields[lat_idx])
                    stop_lon = float(fields[lon_idx])
                    if(code_idx):
                        stop_code = fields[code_idx]
                    else:
                        stop_code = None
                    stop_id_str = fields[id_idx]
                    stop_id = int(''.join(filter(str.isdigit, stop_id_str)))

                    row = one_row((stop_lat, stop_lon), stop_name, file_name, stop_code, stop_id)
                    csvwriter.writerow(row)
                    global counter
                    counter += 1
                    print(counter)
                    # print(stop_id)
                    if counter >= DOWNLOAD_LIMIT:
                            print("Reached download limit. Exiting...")
                            sys.exit() 


def process_bus_stops_xlsx(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_excel(file_path)
    for index, row in df.iterrows():
        stop_name = row['STOP_NAME']
        lat, lng = row['LATITUDE'], row['LONGITUDE']
        one_row((lat, lng), stop_name, file_name)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py file.extension")
        return
    
    global city_name
    file_path = sys.argv[1]
    city_name, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.txt':
        

        process_bus_stops_txt(file_path)
    elif file_extension.lower() == '.xlsx':
        process_bus_stops_xlsx(file_path)
    else:
        print("Unsupported file type.")

if __name__ == "__main__":
    main()
