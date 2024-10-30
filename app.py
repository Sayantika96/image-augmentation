from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import imageio
import torch
from torchvision import transforms
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)

augmented_transformations = [
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
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
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.1),
    ]),
    transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
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
        transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor()
    ])
]

def apply_augmentation(image_path):
    # Check if the input image is a GIF
    if image_path.endswith('.gif'):
        reader = imageio.get_reader(image_path)
        augmented_images = []

        for frame in reader:
            # Convert each frame to PIL Image
            image = Image.fromarray(frame)
            image = image.convert('RGB')  # Ensure the image is in RGB mode
            for idx, transform in enumerate(augmented_transformations):
                augmented_image = transform(image)

                # Convert to PIL Image and ensure it's in RGB mode
                augmented_image = transforms.ToPILImage()(augmented_image).convert('RGB')

                output_filename = f"augmented_{uuid.uuid4().hex}_{idx}.jpg"
                output_path = os.path.join(UPLOAD_FOLDER, output_filename)
                augmented_image.save(output_path)

                augmented_images.append(output_filename)

        return augmented_images
    else:
        # Handle single image augmentation as before
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure the image is in RGB mode
        augmented_images = []
        output_filenames = []
        for idx, transform in enumerate(augmented_transformations):
            augmented_image = transform(image)

            # Convert to PIL Image and ensure it's in RGB mode
            augmented_image = transforms.ToPILImage()(augmented_image).convert('RGB')

            output_filename = f"augmented_{uuid.uuid4().hex}_{idx}.jpg"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)

            augmented_image.save(output_path)
            output_filenames.append(output_filename)

        return output_filenames


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file", 400
        
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        augmented_filenames = apply_augmentation(file_path)

        return render_template('index.html', uploaded_image=filename, augmented_images=augmented_filenames)

    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
