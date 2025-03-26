import streamlit as st
import json
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

def process_coco_file(coco_data):
    objects_per_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        objects_per_image[image_id] = objects_per_image.get(image_id, 0) + 1

    total_images = len(objects_per_image)
    max_objects_in_one_image = max(objects_per_image.values(), default=0)
    min_objects_in_one_image = min(objects_per_image.values(), default=0)
    avg_objects_per_image = sum(objects_per_image.values()) / total_images if total_images > 0 else 0

    count_10_20 = sum(1 for count in objects_per_image.values() if 10 <= count < 20)
    count_20_50 = sum(1 for count in objects_per_image.values() if 20 <= count < 50)
    count_50_100 = sum(1 for count in objects_per_image.values() if 50 <= count < 100)
    count_above_100 = sum(1 for count in objects_per_image.values() if count >= 100)

    return {
        "total_images": total_images,
        "total_annotations": len(coco_data['annotations']),
        "max_objects_in_one_image": max_objects_in_one_image,
        "min_objects_in_one_image": min_objects_in_one_image,
        "avg_objects_per_image": avg_objects_per_image,
        "count_10_20": count_10_20,
        "count_20_50": count_20_50,
        "count_50_100": count_50_100,
        "count_above_100": count_above_100,
    }

def combine_statistics(stat_list):
    combined_stats = {
        "total_images": sum(stat["total_images"] for stat in stat_list),
        "total_annotations": sum(stat["total_annotations"] for stat in stat_list),
        "max_objects_in_one_image": max(stat["max_objects_in_one_image"] for stat in stat_list),
        "min_objects_in_one_image": min(stat["min_objects_in_one_image"] for stat in stat_list),
        "avg_objects_per_image": 0,
        "count_10_20": sum(stat["count_10_20"] for stat in stat_list),
        "count_20_50": sum(stat["count_20_50"] for stat in stat_list),
        "count_50_100": sum(stat["count_50_100"] for stat in stat_list),
        "count_above_100": sum(stat["count_above_100"] for stat in stat_list),
    }
    if combined_stats["total_images"] > 0:
        combined_stats["avg_objects_per_image"] = combined_stats["total_annotations"] / combined_stats["total_images"]
    return combined_stats


def load_coco_data(file):
    return json.load(file)

def get_annotations_by_image(coco_data):
    annotations_by_image = {}
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    return annotations_by_image, image_id_to_name

def compute_new_annotations(prev_coco_data, new_coco_data):
    prev_annotations, prev_image_names = get_annotations_by_image(prev_coco_data)
    new_annotations, new_image_names = get_annotations_by_image(new_coco_data)
    
    new_image_ids = set(new_annotations.keys()) - set(prev_annotations.keys())
    added_annotations = {}
    
    for image_id, anns in new_annotations.items():
        if image_id in new_image_ids:
            added_annotations[image_id] = anns
        elif image_id in prev_annotations:
            prev_ann_ids = {ann['id'] for ann in prev_annotations[image_id]}
            new_ann_list = [ann for ann in anns if ann['id'] not in prev_ann_ids]
            if new_ann_list:
                added_annotations[image_id] = new_ann_list
    
    return added_annotations, new_image_names

def calculate_statistics(annotations_by_image):
    objects_per_image = {img_id: len(anns) for img_id, anns in annotations_by_image.items()}
    
    total_images = len(objects_per_image)
    total_annotations = sum(objects_per_image.values())
    max_objects_in_one_image = max(objects_per_image.values(), default=0)
    min_objects_in_one_image = min(objects_per_image.values(), default=0)
    avg_objects_per_image = total_annotations / total_images if total_images > 0 else 0
    
    count_10_20 = sum(1 for count in objects_per_image.values() if 10 <= count < 20)
    count_20_50 = sum(1 for count in objects_per_image.values() if 20 <= count < 50)
    count_50_100 = sum(1 for count in objects_per_image.values() if 50 <= count < 100)
    count_above_100 = sum(1 for count in objects_per_image.values() if count >= 100)
    
    return {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "max_objects_in_one_image": max_objects_in_one_image,
        "min_objects_in_one_image": min_objects_in_one_image,
        "avg_objects_per_image": avg_objects_per_image,
        "count_10_20": count_10_20,
        "count_20_50": count_20_50,
        "count_50_100": count_50_100,
        "count_above_100": count_above_100,
    }

st.set_page_config(page_title="COCO JSON Analyzer", layout="wide")
st.title("📊 COCO JSON Statistics & Comparison Tool")

option = st.selectbox("Choose an option:", ["Combine JSON Files & Compute Stats", "Compare Two JSON Files"])

if option == "Combine JSON Files & Compute Stats":
    uploaded_files = st.file_uploader("Upload COCO JSON files", accept_multiple_files=True, type=["json"])
    if uploaded_files:
        all_statistics = []
        for uploaded_file in uploaded_files:
            coco_data = json.load(uploaded_file)
            stats = process_coco_file(coco_data)
            all_statistics.append(stats)

            with st.expander(f"📁 Statistics for {uploaded_file.name}"):
                st.write(f"**1) Number of images:** {stats['total_images']}")
                st.write(f"**2) Total number of annotations:** {stats['total_annotations']}")
                st.write(f"**3) Maximum number of objects in one image:** {stats['max_objects_in_one_image']}")
                st.write(f"**4) Minimum number of objects in one image:** {stats['min_objects_in_one_image']}")
                st.write(f"**5) Average number of objects in one image:** {stats['avg_objects_per_image']:.2f}")
                st.write(f"**6) Number of images with 10-20 objects:** {stats['count_10_20']}")
                st.write(f"**7) Number of images with 20-50 objects:** {stats['count_20_50']}")
                st.write(f"**8) Number of images with 50-100 objects:** {stats['count_50_100']}")
                st.write(f"**9) Number of images with more than 100 objects:** {stats['count_above_100']}")
                
                fig = px.bar(pd.DataFrame([stats]).T, title=f"Statistics Overview: {uploaded_file.name}", 
                             labels={"index": "Metric", "value": "Count"})
                st.plotly_chart(fig)

        if all_statistics:
            combined_stats = combine_statistics(all_statistics)
            st.subheader("📊 Combined Statistics")
            
            st.write(f"**1) Number of images:** {combined_stats['total_images']}")
            st.write(f"**2) Total number of annotations:** {combined_stats['total_annotations']}")
            st.write(f"**3) Maximum number of objects in one image:** {combined_stats['max_objects_in_one_image']}")
            st.write(f"**4) Minimum number of objects in one image:** {combined_stats['min_objects_in_one_image']}")
            st.write(f"**5) Average number of objects in one image:** {combined_stats['avg_objects_per_image']:.2f}")
            st.write(f"**6) Number of images with 10-20 objects:** {combined_stats['count_10_20']}")
            st.write(f"**7) Number of images with 20-50 objects:** {combined_stats['count_20_50']}")
            st.write(f"**8) Number of images with 50-100 objects:** {combined_stats['count_50_100']}")
            st.write(f"**9) Number of images with more than 100 objects:** {combined_stats['count_above_100']}")
            
            fig_combined = px.bar(pd.DataFrame([combined_stats]).T, title="Overall Statistics", 
                                  labels={"index": "Metric", "value": "Count"})
            st.plotly_chart(fig_combined)
            
            st.download_button("📥 Download Combined Statistics", data=json.dumps(combined_stats, indent=4), 
                               file_name="combined_statistics.json", mime="application/json")

elif option == "Compare Two JSON Files":
    prev_file = st.file_uploader("Upload Previous COCO JSON File", type=["json"])
    new_file = st.file_uploader("Upload New COCO JSON File", type=["json"])

    if prev_file and new_file:
        prev_coco_data = load_coco_data(prev_file)
        new_coco_data = load_coco_data(new_file)
    
        new_annotations, new_image_names = compute_new_annotations(prev_coco_data, new_coco_data)
        statistics = calculate_statistics(new_annotations)
    
        st.subheader("New Annotations Statistics")
        st.write(f"**1) Number of images:** {statistics['total_images']}")
        st.write(f"**2) Total number of annotations:** {statistics['total_annotations']}")
        st.write(f"**3) Maximum number of objects in one image:** {statistics['max_objects_in_one_image']}")
        st.write(f"**4) Minimum number of objects in one image:** {statistics['min_objects_in_one_image']}")
        st.write(f"**5) Average number of objects in one image:** {statistics['avg_objects_per_image']:.2f}")
        st.write(f"**6) Number of images with 10-20 objects:** {statistics['count_10_20']}")
        st.write(f"**7) Number of images with 20-50 objects:** {statistics['count_20_50']}")
        st.write(f"**8) Number of images with 50-100 objects:** {statistics['count_50_100']}")
        st.write(f"**9) Number of images with more than 100 objects:** {statistics['count_above_100']}")

        # Plot bar graph for statistics
        labels = [
        "Total Images", "Total Annotations", "Max Objects/Image", "Min Objects/Image", "Avg Objects/Image", 
        "10-20 Objects", "20-50 Objects", "50-100 Objects", "Above 100 Objects"
        ]
        values = [
        statistics["total_images"], statistics["total_annotations"], statistics["max_objects_in_one_image"],
        statistics["min_objects_in_one_image"], statistics["avg_objects_per_image"], statistics["count_10_20"],
        statistics["count_20_50"], statistics["count_50_100"], statistics["count_above_100"]
        ]
    
        fig, ax = plt.subplots()
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("COCO Annotations Statistics")
    
        st.pyplot(fig)