from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Set the path for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model(r"model1.keras")

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels
class_labels = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight',
                'Leaf Mold', 'Septoria Leaf Spot', 'Target Spot',
                'Tomato Yellow Leaf Curl Virus Disease', 'Tomato Mosaic Virus Disease', 'Two Spotted Spider Mite Disease']

# Disease to remedy mapping
disease_remedies = {
    "Bacterial Spot": "Use copper-based sprays, avoid overhead watering, practice crop rotation.",
    "Early Blight": "Apply fungicides like mancozeb or chlorothalonil, remove infected leaves, practice crop rotation.",
    "Healthy": "No remedial action needed.",
    "Late Blight": "Apply copper-based fungicides, remove and destroy affected plants, avoid wet foliage.",
    "Leaf Mold": "Increase air circulation, apply fungicides like chlorothalonil, use resistant varieties.",
    "Septoria Leaf Spot": "Remove affected leaves, apply fungicides, ensure proper plant spacing for airflow.",
    "Target Spot": "Use fungicides like azoxystrobin, remove affected plant debris, ensure good drainage.",
    "Tomato Yellow Leaf Curl Virus Disease": "Use resistant varieties, control whiteflies (the virus vectors), remove infected plants.",
    "Tomato Mosaic Virus Disease": "Disinfect tools, remove infected plants, avoid smoking near plants (virus can spread via hands).",
    "Two Spotted Spider Mite Disease": "Use miticides, introduce natural predators (e.g., ladybugs), increase humidity around plants."
}

# Organic remedies dictionary
organic_remedies = {
    'Bacterial Spot': 'Use copper-based fungicides or neem oil for organic control.',
    'Early Blight': 'Use compost tea or a solution of baking soda and water for organic treatment.',
    'Late Blight': 'Spray with organic-approved copper fungicides or potassium bicarbonate solution.',
    'Leaf Mold': 'Apply a diluted mixture of hydrogen peroxide or use neem oil.',
    'Septoria Leaf Spot': 'Use compost teas and neem oil as organic fungicides.',
    'Target Spot': 'Organic sulfur fungicides or copper-based sprays work well.',
    'Tomato Yellow Leaf Curl Virus Disease': 'Use reflective mulches or insecticidal soap for whitefly control.',
    'Two Spotted Spider Mite Disease': 'Neem oil or insecticidal soap is effective for organic control.'
}

# Product links associated with specific diseases
product_links = {
    "Bacterial Spot": {"Copper Fungicide": "https://www.google.com/url?url=https://www.amazon.in/Weird-Road/dp/B097YMHBWT%3Fsource%3Dps-sl-shoppingads-lpcontext%26ref_%3Dfplfs%26psc%3D1%26smid%3DA2GZIGFVZ7EICN&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjf5auYu7SJAxUAmq8BHZprC7QQ1SkI_gUoAA&usg=AOvVaw0i9BCfuuw3nlO9FKIffrdt","Neem Oil":"https://amzn.in/d/eKW6DUC"},
    "Early Blight": {"Compost Tea": "https://www.google.com/url?url=https://www.etsy.com/in-en/listing/699286563/compost-tea-for-plants-100-organic%3Fgpla%3D1%26gao%3D1%26&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjvsvjiu7SJAxXWh68BHQRtLiAQ1SkI0AYoAA&usg=AOvVaw3vkbl_FJuE1hhhSTtDcnBo","Baking Soda":"https://www.google.com/url?url=https://blinkit.com/prn/puramate-baking-soda/prid/480575%3Flat%3D19.1718975155554%26lon%3D72.8545546518707&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjYvYT_u7SJAxUDkq8BHVUqCXQQ1SkIvQYoAA&usg=AOvVaw1FEo94j68WWFVagRZu8AiR"},
    "Late Blight": {"Copper Fungicide": "https://www.example.com/copper-fungicide","Potassium Bicarbonate":"https://www.google.com/url?url=https://bulkagrochem.com/product/potassium-bicarbonate/%3Fsrsltid%3DAfmBOoqKOTnpUlLRb8YQ-WptdFskuK5L9nAHbgvYn6IyQNssxr60AL6K85Y&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwiAve-NvLSJAxXIYfUHHc99MmYQ1SkI3AYoAA&usg=AOvVaw2hcIYlqsAZ4-srb9v3dStV"},
    "Leaf Mold": {"Neem Oil": "https://amzn.in/d/eKW6DUC","Hydrogen Peroxide":"https://www.google.com/url?url=https://www.1mg.com/otc/hydrogen-peroxide-solution-otc685325%3Fsrsltid%3DAfmBOorbIoPmcWFU-0JeRZTSbmj4WrWMPlK7V7fC488sRbiSIkVu5l9YY04&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjDh9awvLSJAxU7ma8BHS-_B24Q1SkItAcoAA&usg=AOvVaw1QP2hQwH5Z47GDzNsV4XS8"},
    "Septoria Leaf Spot": {"Neem Oil": "https://amzn.in/d/eKW6DUC","Compost Tea": "https://www.google.com/url?url=https://www.etsy.com/in-en/listing/699286563/compost-tea-for-plants-100-organic%3Fgpla%3D1%26gao%3D1%26&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjvsvjiu7SJAxXWh68BHQRtLiAQ1SkI0AYoAA&usg=AOvVaw3vkbl_FJuE1hhhSTtDcnBo"},
    "Target Spot":{"Sulfur Fungicides":"https://www.google.com/url?url=https://wholesale.krushikendra.com/Thiofit-Sulphur-80-WDG-Fungicide%3Fsrsltid%3DAfmBOoovxD_eusLrQ9jq-iDfsPrGzJRRwTYRpvkg78XoeFI4J0v45WRtD_Q&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjGiNCIu7SJAxUzZvUHHdc4KHwQ1SkI2AUoAA&usg=AOvVaw0YWBdgxPw7mB9oB_rdol-d","Copper Spray":"https://www.google.com/url?url=https://kisancenter.in/product/23169334/Tata-Blitox-Fungicide---Blue-Copper-50--WP-%3Fsrsltid%3DAfmBOopOpQx7xSThamE5zhXWHhMUfecbq3gnMDLI6SxR_MA5XK2OPMh0DwU&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjZ9u7ovLSJAxUwbfUHHd1KAAcQ1SkIoQYoAA&usg=AOvVaw2_XIfLI5uVdOO0eVYF-EtS"},
    "Tomato Yellow Leaf Curl Virus Disease":{"Reflective Mulches":"https://www.google.com/url?url=https://allschoolabs.com/product/plastic-mulch-film-reflective-films-silver-black-300m-x-1-2m/%3Fsrsltid%3DAfmBOoroN9PzhSl8Hr36bRvlZX-WciTxMpuLKwsLgBb0Xt0KSKWPZ2vFgsM&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwiA0JL3vLSJAxW1hq8BHe1FGMkQ1SkI_gUoAA&usg=AOvVaw18nnjqkzVVCBqZLKtGJQ57","Insecticidal soap":"https://www.google.com/url?url=https://www.pavithrampets.com/product/31447518/Tik-Out-Soap-75-Grms%3Futm_source%3DGMC%26srsltid%3DAfmBOoqs87YEzW6UANRkoJmSA9ZAkeulOP_vc_06eKN-Gp59t2jeIFjdqVM&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwizuMbqurSJAxVDh68BHW01LucQ2SkItQY&usg=AOvVaw1a2vQ6Dj390SQuO38Xhik5"},
    "Two Spotted Spider Mite Disease":{"Neem Oil":"https://amzn.in/d/eKW6DUC","Insecticidal Soap":"https://www.google.com/url?url=https://www.pavithrampets.com/product/31447518/Tik-Out-Soap-75-Grms%3Futm_source%3DGMC%26srsltid%3DAfmBOoqKQRABzTmR3z-BQmDP1bo5pyb92oen4MiC6sO9ZD8qBQCLMWOx39M&rct=j&q=&esrc=s&opi=95576897&sa=U&ved=0ahUKEwjT856XvbSJAxVAgq8BHUIlGtgQ1SkIsAYoAA&usg=AOvVaw3ZcCT8W_B_1tQZ5Q0wTWtU"}
    # Continue adding product links for other diseases as needed
}

def check_image_quality(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var >= 100  # Return True if image quality is sufficient

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_disease(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    return class_labels[predicted_class], predictions[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Check image quality
        if not check_image_quality(file_path):
            return "The image is too blurry, please upload a clearer image."

        # Detect disease
        detected_disease, probabilities = predict_disease(file_path)

        # Get the remedy for the detected disease
        remedy = organic_remedies.get(detected_disease, disease_remedies.get(detected_disease, "No remedy found."))

        # Get product links specific to the detected disease
        specific_links = product_links.get(detected_disease, {})

        return render_template('result.html', result=detected_disease, remedy=remedy, purchase_links=specific_links)

if __name__ == '__main__':
    app.run(debug=True)
