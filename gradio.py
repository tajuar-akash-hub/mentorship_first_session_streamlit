import gradio as gr
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def predict_price(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, 
                  basement, hotwaterheating, airconditioning, prefarea, furnishingstatus):
    
    # Encode inputs
    mainroad = 1.0 if mainroad == "Yes" else 0.0
    guestroom = 1.0 if guestroom == "Yes" else 0.0
    basement = 1.0 if basement == "Yes" else 0.0
    hotwaterheating = 1.0 if hotwaterheating == "Yes" else 0.0
    airconditioning = 1.0 if airconditioning == "Yes" else 0.0
    prefarea = 1.0 if prefarea == "Yes" else 0.0
    
    furnishing_map = {"furnished": 0.0, "semi-furnished": 1.0, "unfurnished": 2.0}
    furnishingstatus = furnishing_map[furnishingstatus]
    
    # Create input array
    input_data = np.array([[area, bedrooms, bathrooms, stories, 
                           mainroad, guestroom, basement,
                           hotwaterheating, airconditioning,
                           parking, prefarea, furnishingstatus]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    return f"₹ {prediction[0]:,.2f}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area (sq ft)", value=5000),
        gr.Dropdown(choices=[1, 2, 3, 4, 5, 6], label="Bedrooms", value=3),
        gr.Dropdown(choices=[1, 2, 3, 4], label="Bathrooms", value=2),
        gr.Dropdown(choices=[1, 2, 3, 4], label="Stories", value=2),
        gr.Dropdown(choices=[0, 1, 2, 3], label="Parking", value=1),
        gr.Dropdown(choices=["Yes", "No"], label="Main Road", value="Yes"),
        gr.Dropdown(choices=["Yes", "No"], label="Guest Room", value="No"),
        gr.Dropdown(choices=["Yes", "No"], label="Basement", value="No"),
        gr.Dropdown(choices=["Yes", "No"], label="Hot Water Heating", value="No"),
        gr.Dropdown(choices=["Yes", "No"], label="Air Conditioning", value="Yes"),
        gr.Dropdown(choices=["Yes", "No"], label="Preferred Area", value="Yes"),
        gr.Dropdown(choices=["furnished", "semi-furnished", "unfurnished"], label="Furnishing", value="furnished")
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="🏠 House Price Prediction",
    description="Enter house features to predict the price"
)

demo.launch()