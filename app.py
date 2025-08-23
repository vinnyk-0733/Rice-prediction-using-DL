import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import re

def model_prediction(test_image):
    model = tf.keras.models.load_model("rice_prediction.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    confidence = np.max(predictions)
    predicted_index = np.argmax(predictions)
    return predicted_index, confidence

# Rice class names and descriptions
class_name = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
description = {
    'Arborio': """
    Introduction: Arborio rice is a short-grain, starchy rice variety that originated in Italy, primarily used for dishes like risotto and rice pudding. It is named after the town of Arborio in the Po Valley. Arborio rice is characterized by:

High starch content (amylopectin)

Creamy texture when cooked

Short, round grains


Arborio is not traditionally grown in India, but with proper adaptation, it can be cultivated in certain regions of India that mimic temperate conditions or have the right irrigation and soil setup.

    
    1. Climate Requirements for Arborio Rice
Temperature: Needs a warm climate with temperatures between 21°C to 26°C.
Growing Season: Grown in spring and summer, harvested in autumn.
Rainfall: Requires moderate rainfall (60-100 cm), but excessive rain can damage the crop.
Sunlight: Requires full sun for optimal growth. 
Temperature: Arborio rice prefers a moderate, temperate climate, ideally 18°C to 30°C.

Rainfall: Requires moderate rainfall or controlled irrigation, especially during the growing season.

Growing Period: About 150 to 180 days from sowing to harvesting.

Sunlight: Needs full sun exposure for healthy growth.


Indian Comparison:

Since India is mostly tropical/subtropical, Arborio is best suited for cooler regions with controlled irrigation—hill regions or winter rice areas may suit this crop.


2. Soil Requirements
Prefers fertile, clayey, or silty loam soil that can retain water well.
Needs a neutral to slightly acidic pH (6.0 - 7.0). Soil Type: Deep, fertile loamy or clay loam soils with high organic content.

Soil pH: Neutral to slightly acidic (6.0 to 7.0).

Drainage: Requires low-lying fields that retain water during early growth stages.

Nutrients: Nitrogen-rich soil is ideal for good grain development.


Indian Regions with Suitable Soils:

Regions with alluvial soils, especially river basins and delta regions with heavy silt or clay content.


3. Water Requirements
Arborio rice is semi-aquatic and needs standing water during its growth period.
Requires flood irrigation, similar to Basmati rice. 
Water Demand: Arborio rice is moderately water-intensive, needing 1000–1200 mm of water during the season.

Flooding: Needs shallow flooding (like traditional paddy) during the transplanting and tillering stages.

Irrigation: Controlled irrigation is preferred over excessive flooding, especially closer to harvest, to preserve grain quality.

Drainage: Proper drainage needed before harvest to prevent fungal issues.


Note: Arborio prefers wet feet but not too much water, making it different from high-water crops like Jasmine or Basmati.


4. Major Growing Regions
Italy (Piedmont, Lombardy), USA (California, Texas), India (Tamil Nadu, Karnataka), Spain, Australia""",

    'Basmati': """
    Introduction: Basmati rice is a premium-quality aromatic long-grain rice, globally known for its distinct fragrance, slender grains, and fluffy texture. India is the largest producer and exporter of Basmati rice in the world.

    
    1. Climate Required for Basmati Rice
Temperature Range:
Requires warm climate with temperatures between 25°C to 35°C during the growing season.

Climate Type:
Sub-tropical climate with clear skies, bright sunlight, and high humidity is ideal.

Growing Season:

Sowing: June–July (Kharif season)

Harvesting: October–November


Sunlight and Rainfall:
Needs long sunny days and rainfall between 100–150 cm for optimum yield.

Example:
In Haryana’s Karnal district, Basmati rice (especially Pusa Basmati 1121) is cultivated in large areas due to the ideal mix of warm temperatures and monsoon rains.


2. Soil Requirements
Requires fertile alluvial soils, which are rich in organic matter. Loam to clay loam soil is best.

pH Level:
Slightly acidic to neutral (pH 5.5 to 7.5)

Drainage:
Soil must retain water for paddy conditions but also have good drainage to avoid stagnation during early growth.

Example:
The Indo-Gangetic Plains—especially areas near the Yamuna and Ganga rivers—offer ideal alluvial soil conditions for high-quality Basmati cultivation.


3. Water Requirements
Water Usage:
Basmati rice is a water-intensive crop, needing continuous standing water (4–5 inches) during most of its growth period.

Irrigation:
Requires 4–6 irrigations during the crop cycle, depending on rainfall and soil type.

Water Management:
Puddling (preparing the soil with water) is crucial before transplanting the seedlings.

Example:
In Punjab’s Amritsar and Tarn Taran districts, efficient canal irrigation systems help maintain ideal water levels for Basmati paddy fields.


4. Major Producing States in India
India grows Basmati rice mainly in the northwestern plains, particularly in the states of:

Punjab

Haryana

Uttar Pradesh (Western UP)

Himachal Pradesh (foothills)

Uttarakhand

Jammu & Kashmir

Delhi (fringe rural areas)

These areas are collectively known as the “Basmati Belt” of India.

Example 1:
Haryana is one of the top Basmati producers, especially districts like Karnal, Kurukshetra, and Kaithal, growing export-grade varieties like Pusa Basmati 1509 and 1121.

Example 2:
Western Uttar Pradesh, especially Meerut, Bijnor, and Moradabad, has become a major hub for Basmati farming due to good soil, water availability, and increasing contract farming models.
""",

    'Ipsala': """
    Introduction: Ipsala rice is a premium variety of long-grain rice that originates from the Ipsala region of Edirne province in Turkey. It is well-known for its high quality, long grains, pleasant aroma, and excellent cooking properties. This rice is largely grown in European countries, especially Turkey, under regulated agricultural practices.

In India, although Ipsala rice is not commonly cultivated, it shares many similarities with long-grain and aromatic rice varieties such as Basmati rice, which are widely grown in Indian conditions. The environmental and agronomic requirements for Ipsala rice are quite similar to those of Basmati and other premium rice types.

    
    
    1. Climate Requirement:
Warm and humid, 20°C to 35°C. Season: 4–6 months. Ipsala rice, like most high-quality rice varieties, requires specific climatic conditions for optimal growth:

Temperature: A warm climate is essential. The ideal temperature range is 20°C to 35°C. Rice does not tolerate frost and requires consistent warmth throughout the growing season.

Rainfall: Moderate to high rainfall is beneficial. However, since paddy rice is typically grown in standing water, regions with good irrigation infrastructure are also suitable.

Humidity: A humid climate is preferred, especially during the early stages of plant growth and flowering.

Growing Season: Ipsala rice needs a relatively long growing season, typically 4 to 6 months from planting to harvest.


Comparable Indian Conditions:

Northern plains and river basins such as in Punjab, Haryana, and Uttar Pradesh experience similar climatic conditions that support Basmati rice, indicating that Ipsala rice can also adapt there.


2. Soil Requirement:
Fertile alluvial or loamy/clayey. pH 5.5–7. Soil is a crucial factor for high-yielding and high-quality rice cultivation:

Soil Type: Fertile alluvial soils and clay loams are ideal for Ipsala rice. These soils retain water well and support healthy root development.

Soil pH: Slightly acidic to neutral soils (pH range of 5.5 to 7.0) are suitable.

Drainage: Rice fields should ideally be low-lying and capable of retaining standing water. Poorly drained soils or those that dry quickly are not suitable for paddy cultivation.


Indian Comparison:

The Indo-Gangetic plains, delta regions of Andhra Pradesh and Tamil Nadu, and eastern states like West Bengal and Bihar have suitable soil conditions for cultivating rice with high water retention and fertility.


3. Water Requirement:
High water needs (1200–1500 mm), flooded fields. Water is perhaps the most critical factor in rice cultivation:

Water Quantity: Ipsala rice, like most paddy varieties, is water-intensive. It requires about 1200 to 1500 mm of water throughout its growing season.

Irrigation: In areas with insufficient rainfall, a well-planned irrigation system is essential. Water should be available continuously during transplanting and early growth stages.

Flooding: Fields are generally kept flooded with 5–10 cm of water during the initial growth stages to suppress weeds and maintain healthy crop growth.


4. Major Producing States in India:
Punjab, Haryana, 
Uttar Pradesh, 
West Bengal, 
Andhra Pradesh, 
Telangana,
Chhattisgarh, 
Bihar,
 Odisha""",

    'Jasmine': """
    Introduction: 
Jasmine rice (also known as Thai fragrant rice or Khao Hom Mali) is a long-grain, aromatic rice variety originally cultivated in Thailand and Cambodia. It is famous for its soft texture, sweet aroma, and nutty flavor when cooked.

Although Jasmine rice is not native to India, the climatic and soil conditions in several Indian states are suitable for its cultivation. India already produces aromatic rice varieties like Basmati, so adapting Jasmine rice is agriculturally feasible.

    
    1. Climate Requirement:
Temperature: Jasmine rice grows best in warm, humid climates, with temperatures ranging between 25°C to 35°C.

Rainfall: It requires moderate to high rainfall (at least 1000–1200 mm annually). The crop also thrives well with proper irrigation systems.

Humidity & Sunlight: High humidity during early growth and sunny weather during ripening improves yield and grain quality.

Growing Season: Jasmine rice takes about 4–5 months from sowing to harvesting, depending on the variety and local climate.


Example (Comparable Indian Regions):

Eastern Uttar Pradesh and Chhattisgarh offer similar warm, humid conditions suitable for fragrant rice.

Odisha and West Bengal have the monsoon-dependent paddy systems ideal for Jasmine rice.


2. Soil Requirement:
Loamy or clay loam, pH 5.5–7.0. Soil Type: Jasmine rice prefers fertile, well-drained alluvial soils or clay loams that retain moisture.

Soil Fertility: Rich in organic matter and nutrients like nitrogen and phosphorus.

Soil pH: Slightly acidic to neutral (5.5 to 7.0).

Water Retention: Soils that can hold water for long periods are preferred, especially during early stages.


Indian Regions with Suitable Soils:

Ganga-Brahmaputra delta (West Bengal), Godavari-Krishna delta (Andhra Pradesh), and the Mahanadi basin (Odisha) have suitable soils for Jasmine rice.


3. Water Requirement:
Needs continuous water, flooded fields ideal. Total Water Needed: Jasmine rice is moderately water-intensive, requiring about 1000–1200 mm during its growth period.

Irrigation: Fields are often flooded during the seedling and tillering stages. However, well-drained conditions are needed later to avoid grain discoloration or rot.

Flooding: Fields may be kept flooded with about 5–10 cm of standing water in early growth stages.

Drainage: Excess water must be drained toward harvest time to ensure high grain quality.


Indian Comparison:

States like Chhattisgarh, Assam, and Jharkhand have sufficient rainfall or irrigation for water-loving rice like Jasmine.


4. Major Producing Regions:
Tamil Nadu, Kerala, West Bengal, Assam, Manipur,""",

    'Karacadag': """
    Introduction: Karacadag rice is an ancient, drought-resistant aromatic rice variety that originates from the Karacadag region of Southeastern Turkey. It is known for:

Growing in dryland areas

Tolerating minimal water

Producing good yields without chemical fertilizers Because of these qualities, Indian researchers and progressive farmers are now exploring its suitability for dryland rice farming.

    
    1. Climate Requirement:
Temperature Range: 20°C to 35°C
Ideal for semi-arid and dry tropical climates.

Climate Type:
Thrives in low rainfall and high-temperature regions. Unlike traditional paddy rice, Karacadag doesn’t require humid or flooded conditions to grow well.

Drought Tolerance:
Can survive prolonged dry spells, making it an excellent choice for climate-resilient farming in India.

Example:
In Telangana's Mahbubnagar district, farmers have started testing Karacadag rice under dryland conditions with minimal irrigation and observed positive results even in seasons with below-average rainfall.


2. Soil Requirement:

Soil Type:
Performs well in well-drained sandy loam to clay loam soils. Avoids waterlogged soils, which are common in paddy cultivation.

pH Level:
Grows best in neutral to slightly alkaline soils (pH 6.5 to 7.5).

Fertility:
Does not demand high fertility and can grow well in marginal soils. Farmers using organic practices find it sustainable and eco-friendly.

Example:
Trials in Karnataka’s dry belt (Raichur and parts of Ballari) showed good adaptation of Karacadag rice in red loamy soil with limited fertilizer input.
 

3. Water Requirement:

Water Usage:
Requires much less water than traditional rice—only needs irrigation during critical growth stages (e.g., tillering and flowering).

Rainfed Capability:
Can be grown successfully in rainfed agriculture systems with occasional supplementary irrigation.

Waterlogging:
Sensitive to waterlogging; fields must be well-drained.

Example:
In Tamil Nadu’s dry zones like Dharmapuri, smallholder farmers used drip irrigation and local rainfall to cultivate Karacadag, saving nearly 40–50% of water compared to regular rice.


4. Major Producing Regions in India:

Currently, Karacadag rice is not a mainstream commercial crop in India, but it is gaining traction through research trials and climate-resilient agriculture projects.

Examples of regions exploring Karacadag rice:

Telangana: Dryland districts like Mahbubnagar, Nalgonda (rainfed trials)

Karnataka: Northern districts like Raichur, Ballari (dry-zone trials)

Tamil Nadu: Dharmapuri, Salem (low-rainfall adaptation)

Andhra Pradesh: Anantapur district (experimental plots on drought-resistant rice"""
}

# Sidebar
st.sidebar.title("Rice Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Rice Prediction"])

# Home Page
if app_mode == "Home":
    st.header("RICE PREDICTION SYSTEM")
    st.markdown("""
    Welcome to the **Rice Type Prediction System**! 🌾🔍

    Upload an image of rice, and our intelligent system will analyze it to identify the rice type using deep learning.

    ### How It Works
    1. **Upload Image**: Go to the *Rice Prediction* page and upload a clear image of the rice sample.
    2. **Get Prediction**: Our model processes the image and identifies the rice type.
    3. **Know More**: Get a short description of the predicted rice variety.

    👉 Click on the *Rice Prediction* page in the sidebar to get started!
    """)

# About Page
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    #### Dataset Info:
    - High-resolution images of rice types (e.g., Basmati, Jasmine, etc.)
    - 80% training, 20% validation split.

    #### Technologies Used:
    - Python
    - TensorFlow / Keras
    - Streamlit
    """)

# Prediction Page
elif app_mode == "Rice Prediction":
    st.header("Rice Prediction")

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "predicted_class" not in st.session_state:
        st.session_state.predicted_class = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = 0.0

    uploaded_file = st.file_uploader("Upload an Image of Rice Sample:")
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file

    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, width=10, use_container_width=True)

        if st.button("Predict"):
            predicted_index, confidence = model_prediction(st.session_state.uploaded_image)
            if confidence >= 0.85:
                predicted_class = class_name[predicted_index]
                st.session_state.predicted_class = predicted_class
                st.session_state.confidence = confidence
            else:
                st.warning("❌ Image not recognized. Please upload a clear rice image.")
                st.session_state.predicted_class = None

    if st.session_state.predicted_class:
        st.subheader("Prediction Result")
        st.success(f"The model predicts it's **{st.session_state.predicted_class}** with {st.session_state.confidence * 100:.2f}% confidence")

        # Dropdown including Introduction
        option = st.selectbox("Select Information Section", 
                              ["Introduction", "Climate Requirement", "Soil Requirement", "Water Requirement", "Major Producing Regions"])

        full_desc = description[st.session_state.predicted_class]

        # Extract section based on selection
        if option == "Introduction":
            section = full_desc.split("1.")[0]
        elif option == "Climate Requirement":
            section = full_desc.split("2.")[0].split("1.")[1]
        elif option == "Soil Requirement":
            section = full_desc.split("3.")[0].split("2.")[1]
        elif option == "Water Requirement":
            section = full_desc.split("4.")[0].split("3.")[1]
        elif option == "Major Producing Regions":
            section = full_desc.split("4.")[1]

        st.markdown(f"### {option}")
        st.markdown(section.strip())
