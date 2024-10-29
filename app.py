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

# Define augmented transformations
augmented_transformations = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ColorJitter(contrast=0.5),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ColorJitter(saturation=0.5),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ColorJitter(hue=0.5),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomResizedCrop((128, 128), scale=(0.5, 1.0)),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomAffine(degrees=15),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])
]

# Helper function to apply transformations
def apply_augmentation(image_path):
    # Open the image
    image = Image.open(image_path)
    augmented_images = []
    output_filenames = []
    
    # Apply each augmentation
    for idx, transform in enumerate(augmented_transformations):
        # Apply the transformation
        augmented_image = transform(image)

        # Convert the tensor back to PIL image for saving
        augmented_image = transforms.ToPILImage()(augmented_image)

        # Generate a unique filename for the augmented image
        output_filename = f"augmented_{uuid.uuid4().hex}_{idx}.jpg"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # Save the augmented image
        augmented_image.save(output_path)
        
        # Store the output filename
        output_filenames.append(output_filename)

    return output_filenames

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
        augmented_filenames = apply_augmentation(file_path)

        # Render the uploaded and augmented images in the response
        return render_template('index.html', uploaded_image=filename, augmented_images=augmented_filenames)

    # Render the upload form on GET request
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
