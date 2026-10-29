# 07 Streamlit app

Screens/sections: language switch (English/Hindi/Marathi), upload, image preview, result card (disease, confidence bar, status message),
advice (summary + expandable details), feedback buttons, about/how-to, disclaimer.
Behaviour: show clear messages for each status in docs/04; spinner while predicting; handle bad files; never crash on errors.
Load the model once (cache). Show model version in the footer.
Keep the existing green theme. The app imports predict/recommend/feedback from src; no model logic inside the UI file.
Later: swap the UI for the React website that calls this model through a small API (FastAPI) deployed on Hugging Face Spaces.
## Supported conditions
Display names: Black Rot, Esca, Leaf Blight, Healthy Leaves, Not a Grape Leaf.
Show on every result: "Downy Mildew and Powdery Mildew are not supported. A leaf with these diseases may be misclassified, including as Healthy."
