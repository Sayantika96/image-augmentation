from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import torch
from torchvision import transforms
import uuid

# Initialize Flask app
app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define augmentation transforms
augmented_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop((128, 128)),
    transforms.ToTensor()
])

# Helper function to apply transformations
def apply_augmentation(image_path):
    # Open the image
    image = Image.open(image_path)

    # Apply the augmentation
    augmented_image = augmented_transform(image)

    # Convert the tensor back to PIL image for saving
    augmented_image = transforms.ToPILImage()(augmented_image)

    # Generate a unique filename for the augmented image
    output_filename = f"augmented_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    
    # Save the augmented image
    augmented_image.save(output_path)
    
    return output_filename

# Route for the homepage with upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        
        # Check if the file has a valid name
        if file.filename == '':
            return "No selected file", 400
        
        # Save the uploaded file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Apply augmentation to the uploaded image
        augmented_filename = apply_augmentation(file_path)

        # Render the augmented image in the response
        return render_template('index.html', uploaded_image=filename, augmented_image=augmented_filename)

    # Render the upload form on GET request
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
