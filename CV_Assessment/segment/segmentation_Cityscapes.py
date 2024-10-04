import tensorflow as tf
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import patches, gridspec


LABEL_NAMES = np.asarray([
    'unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle','ground','dynamic','static'
])

FULL_COLOR_MAP = np.array([
    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
    [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
    [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
])


REQUIRED_LABELS = [1, 2, 9, 10, 12, 14, 16, 13, 20, 21, 22]  

def label_to_color_image(label):
    """将分割标签映射到颜色"""
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    return FULL_COLOR_MAP[label]


def preprocess_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, target_size)
    return resized_image, image_rgb  


def filter_labels(seg_map, required_labels):
    filtered_seg_map = np.copy(seg_map)
    for label in np.unique(seg_map):
        if label not in required_labels:
            filtered_seg_map[seg_map == label] = 0 
    return filtered_seg_map


def save_class_pixel_coordinates(seg_map, output_json_path):
    """保存每个类别的像素点坐标到 JSON 文件"""
    class_pixels = {}


    for label in REQUIRED_LABELS:
       
        coords = np.argwhere(seg_map == label).tolist()  
        if coords:  
            class_pixels[LABEL_NAMES[label]] = coords

    if class_pixels:
        with open(output_json_path, 'w') as json_file:
            json.dump(class_pixels, json_file, indent=4)


def visualize_segmentation(image, seg_map, output_path):

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 1])
    

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('Segmentation Map')
    

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[2])
    legend_elements = [patches.Patch(color=np.array(FULL_COLOR_MAP[lbl])/255.0, label=LABEL_NAMES[lbl])
                       for lbl in unique_labels if lbl in REQUIRED_LABELS]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.axis('off')
    plt.title('Legend')

    plt.savefig(output_path)
    plt.close()


def load_frozen_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def process_image_folder(input_folder, output_folder, frozen_graph_filename):

    graph_def = load_frozen_graph(frozen_graph_filename)


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph_def, name="")
        
  
        input_tensor = sess.graph.get_tensor_by_name('ImageTensor:0')
        output_tensor = sess.graph.get_tensor_by_name('SemanticPredictions:0')


        for file_name in os.listdir(input_folder):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):

                input_image_path = os.path.join(input_folder, file_name)
                preprocessed_image, original_image = preprocess_image(input_image_path)
                
     
                output = sess.run(output_tensor, feed_dict={input_tensor: [preprocessed_image]})
                
    
                seg_map = np.squeeze(output)
                
   
                filtered_seg_map = filter_labels(seg_map, REQUIRED_LABELS)
                

                output_image_name = os.path.splitext(file_name)[0] + "_result.png"
                output_image_path = os.path.join(output_folder, output_image_name)
                
    
                output_json_name = os.path.splitext(file_name)[0] + "_pixels.json"
                output_json_path = os.path.join(output_folder, output_json_name)
                
       
                visualize_segmentation(original_image, filtered_seg_map, output_image_path)
                print(f"已处理并保存: {output_image_path}")
                
 
                save_class_pixel_coordinates(filtered_seg_map, output_json_path)
                print(f"像素坐标已保存到: {output_json_path}")


input_folder = '/Users/xavier/Desktop/Fall 2024/Week 4/input images'  
output_folder = '/Users/xavier/Desktop/Fall 2024/Week 4/output results'  
frozen_graph_filename = '/Users/xavier/Desktop/Fall 2024/Week 4/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb' 

process_image_folder(input_folder, output_folder, frozen_graph_filename)
