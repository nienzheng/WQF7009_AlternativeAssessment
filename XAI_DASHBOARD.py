# cd C:\Users\NZ\OneDrive\Codes\UM2
# streamlit run XAI_DASHBOARD.py

# libraries: machine learning / AI
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
# libraries: GUI/Visualization
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
# libraries: Utilities
import numpy as np
import os
# libraries: LIME
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.metrics import classification_report
import pandas as pd

st.set_page_config(
    page_title="Pneumonia Classifier"
    , page_icon=":camera_with_flash:"
    , layout="wide"
    )

def get_vgg16_model():
    global device
    global model_file_path
    
    model = models.vgg16()
    if not os.path.exists(model_file_path): # use pre-trained weight when trained model not exists
        print("Using VGG16 model with default weight for task")
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
    # Modify the first convolutional layer to accept grayscale input
    new_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    new_conv1.weight.data = model.features[0].weight.data.mean(dim=1, keepdim=True)
    model.features[0] = new_conv1
    # Replace the final classifier layer for binary classification
    model.classifier[6] = nn.Linear(4096, 2)
    
    # Replace all ReLU activations with inplace=False
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    
    if os.path.exists(model_file_path): 
        print("Using local trained VGG16 model for task")
        model.load_state_dict(torch.load(model_file_path, map_location=device, weights_only=True))
    
    return model

def predict_image(model, image_path=None,image=None,uploadfile=None):
    global image_size, device
    
    # use image when image path not available
    if(image_path != None):
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
            transforms.Resize((image_size, image_size)),                # Resize to match input size
            transforms.ToTensor(),                        # Convert to Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values
        ])
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        
    elif(image!=None):
        image = image.unsqueeze(0)
    elif(uploadfile!=None):
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
            transforms.Resize((image_size, image_size)),                # Resize to match input size
            transforms.ToTensor(),                        # Convert to Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values
        ])
        image = Image.open(uploaded_file).convert('RGB')
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        
    # Move image to the device
    image = image.to(device)
    
    # Perform inference
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(image)  # Get raw outputs (logits)

    # Calculate probabilities using softmax
    probabilities = F.softmax(outputs, dim=1)

    # Get predicted class and confidence level
    confidence, predicted_class = torch.max(probabilities, dim=1)
    
    return predicted_class.item(), confidence.item()

def preprocess_image(uploaded_file):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
        transforms.Resize((image_size, image_size)),                # Resize to match input size
        transforms.ToTensor(),                        # Convert to Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values
    ])
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    # image = preprocess(image).unsqueeze(0)  # Add batch dimension
    image = preprocess(image)  # Add batch dimension
    return image

# Grad-CAM implementation
def generate_grad_cam(model, image, target_layer_name, class_index=None):
    # Ensure the model is in evaluation mode
    model.eval()

    # Hook to capture activations and gradients
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Register hooks on the target layer
    target_layer = dict(model.named_children())[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Forward pass
    outputs = model(image)
    if class_index is None:
        class_index = outputs.argmax().item()

    # Compute gradients for the target class
    model.zero_grad()
    target_score = outputs[0, class_index]
    target_score.backward()

    # Compute weights (global average pooling of gradients)
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Generate Grad-CAM heatmap
    grad_cam = (weights * activations).sum(dim=1).squeeze().detach().cpu().numpy()
    grad_cam = np.maximum(grad_cam, 0)  # ReLU
    grad_cam = grad_cam / grad_cam.max()  # Normalize to [0, 1]

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return grad_cam, class_index

# Visualize Grad-CAM heatmap using Pillow
def visualize_grad_cam(image, heatmap, actual_class, predicted_class):
    # Convert image tensor to numpy array
    image_np = image.permute(1, 2, 0).cpu().numpy()  # C, H, W -> H, W, C
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0, 1]

    # Resize heatmap using Pillow
    heatmap_pil = Image.fromarray(np.uint8(heatmap * 255))  # Convert heatmap to a PIL image
    heatmap_resized = heatmap_pil.resize((image_np.shape[1], image_np.shape[0]), Image.BICUBIC)
    heatmap_resized = np.array(heatmap_resized) / 255.0  # Normalize resized heatmap to [0, 1]
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Apply colormap

    # Overlay heatmap on the original image
    overlay = 0.5 * image_np + 0.5 * heatmap_colored
    
    return overlay


def lime_explanation(model, single_image):
    model.eval()  # Set model to evaluation mode

    # Convert the grayscale image to "fake" RGB by duplicating the channel
    image_np = single_image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    image_rgb = np.repeat(image_np, 3, axis=-1)  # Repeat grayscale channel to simulate RGB

    def predict_fn(images):
        # Convert RGB to grayscale by averaging the channels
        grayscale_images = np.mean(images, axis=-1, keepdims=True)  # Average RGB channels
        
        # Convert to PyTorch tensor
        images_tensor = torch.tensor(grayscale_images.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            preds = model(images_tensor)
            return torch.nn.functional.softmax(preds, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb,
        predict_fn,
        top_labels=2,  # Top classes to explain
        hide_color=0,
        # num_samples=1000  # Number of perturbations
        num_samples=100  # Number of perturbations
    )
    
    # Visualize explanation for the top predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
    )
    return temp, mask


def show_images(folder,image_class,num_cols=1):
    class_folder=folder+image_class+'\\'
    
    if os.path.exists(class_folder):
        st.write("# "+image_class)
        image_files = [f for f in os.listdir(class_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
        
        if image_files:
            if(num_cols>=1):
                cols=st.columns(num_cols)
                for index, image_file in enumerate(image_files):
                    col_number = index % num_cols
                    with cols[col_number]:
                        img_path = os.path.join(class_folder, image_file)
                        img = Image.open(img_path).convert('RGB')
                        st.image(img, caption=image_file, use_container_width =True)
                        
                        if(index>30):
                            break
            else:
                full_image_files = [os.path.join(class_folder, img_file) for img_file in image_files]
                st.image(full_image_files,caption=image_files,use_container_width =True)
        else:
            st.write("No images found in folder.")
    else:
        st.write("folder not found.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_file_path = "vgg16_trained_model_2.pth"
class_labels = ['NORMAL','PNEUMONIA']
image_size = 224

model = get_vgg16_model()
model.to(device)
model.eval()

# Sidebar for user to upload an image
st.sidebar.header("Upload Chest X-Ray Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
# Main tabs
tab_prediction, tab_metrics, tab_data, tab_about_dataset, tab_about_model= st.tabs(["Model Prediction", "Model Performance", "Data", 'About Dataset', 'About Model'])

with tab_prediction:
    if uploaded_file:
        predicted_class,confidence = predict_image(model,uploadfile=uploaded_file)
        
        class_label=class_labels[predicted_class]
        st.write(f"### Prediction: {class_label}")
        
        precision = 0.9882
        if(predicted_class==1):
            precision = 0.8527
        # Calculate the chance of being wrong
        chance_of_wrong = (1 - confidence) + confidence * (1 - precision)
        
        st.markdown(f"<h3><b>Prediction Confidence:</b> {confidence * 100:.2f}%</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3><b>Precision:</b> {precision * 100:.2f}%</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='color: red;'><b>Chance of being wrong:</b> {chance_of_wrong * 100:.2f}%</h3>", 
            unsafe_allow_html=True
        )
        
        cols_prediction = st.columns(3)
        
        # Display the uploaded image
        cols_prediction[0].image(uploaded_file, caption='Uploaded Chest X-Ray', use_container_width=True)
        
        
        # Grad-CAM explanation
        # cols_prediction[1].write("## Grad-CAM Visualization")
        
        img_tensor = preprocess_image(uploaded_file)
        # Generate Grad-CAM
        grad_cam, predicted_class = generate_grad_cam(model, img_tensor, target_layer_name="features")
        
        # Visualize Grad-CAM
        grad_cam_overlay = visualize_grad_cam(img_tensor, grad_cam, predicted_class, class_label)
        cols_prediction[1].image(grad_cam_overlay, caption='Grad-CAM Heatmap', use_container_width=True)
        
        # # LIME explanation
        # st.write("## LIME Visualization")
        temp, mask = lime_explanation(model,img_tensor)
        
        # Normalize temp (the image returned by LIME)
        temp_normalized = (temp - temp.min()) / (temp.max() - temp.min())  # Scale to [0, 1]
        
        # Generate marked boundaries
        lime_visualization = mark_boundaries(temp_normalized, mask)
        
        # Normalize the visualization to [0, 1] for Streamlit
        lime_visualization_normalized = (lime_visualization - lime_visualization.min()) / (lime_visualization.max() - lime_visualization.min())

        # cols_prediction[2].image(lime_visualization_normalized, caption='LIME Heatmap', width=600)
        cols_prediction[2].image(lime_visualization_normalized, caption='LIME Heatmap', use_container_width=True)
        
with tab_data:
    st.write("## Data: Chest X-Ray Images")
    # Specify the path to the train folder
    classes = ['NORMAL','PNEUMONIA']
    folder = "C:\\Users\\NZ\\OneDrive\\Codes\\UM2\\chest_xray\\test\\"
    folder = "C:\\Users\\NZ\\OneDrive\\Codes\\UM2\\chest_xray\\examples 2\\"
    
    tab_data_train, tab_data_validation, tab_data_test= st.tabs(["Train", "Validation", "Test"])
    
    num_cols_per_class = 3
    with tab_data_train:
        folder = "C:\\Users\\NZ\\OneDrive\\Codes\\UM2\\chest_xray\\train\\"
        cols = st.columns(2)
        for index, class_label in enumerate(classes):
            with cols[index]:
                show_images(folder, class_label,num_cols=num_cols_per_class)
    with tab_data_validation:
        folder = "C:\\Users\\NZ\\OneDrive\\Codes\\UM2\\chest_xray\\val\\"
        cols = st.columns(2)
        for index, class_label in enumerate(classes):
            with cols[index]:
                show_images(folder, class_label,num_cols=num_cols_per_class)
    with tab_data_test:
        folder = "C:\\Users\\NZ\\OneDrive\\Codes\\UM2\\chest_xray\\test\\"
        cols = st.columns(2)
        for index, class_label in enumerate(classes):
            with cols[index]:
                show_images(folder, class_label,num_cols=num_cols_per_class)
    
with tab_metrics:
    
    
    st.write("## Model Performance Metrics")
    
    st.write("### Model used: VGG-16")
    st.write('')
    

    # Dummy performance metrics
    st.write("### Confusion Matrix")
    st.table([[167, 67], [2, 388]])  # Example confusion matrix
    expand_cm = st.expander("Info", icon=":material/info:")
    expand_cm.write("- **Confusion Matrix**: A scoreboard that shows how well the model predicts pneumonia.")
    expand_cm.table([['True Negatives (TN)', 'False Positives (FP)'], ['False Negatives (FN)', 'True Positives (TP)']])
    expand_cm.write("    - **True Positives (TP)**: Correctly identified pneumonia cases.")
    expand_cm.write("    - **True Negatives (TN)**: Correctly identified non-pneumonia cases.")
    expand_cm.write("    - **False Positives (FP)**: Incorrectly identified pneumonia when it's not there.")
    expand_cm.write("    - **False Negatives (FN)**: Missed pneumonia cases.")
    
    st.write("### Overall Accuracy")
    st.write("- Out of all predictions, how many were correct? **Accuracy: 88.9%**")
    
    st.write("### Accuracy: PNEUMONIA case")
    st.write("- When the model says it's PNEUMONIA, how often is it right? **Precision: 85.3%**")
    st.write("- Out of all actual PNEUMONIA cases, how many did the model catch? **Recall: 99.5%**")
    st.write("- A single score that balances precision and recall. **F1-Score**: 91.8%**")
    
    st.write("### Accuracy: NORMAL case")
    st.write("- When the model says it's NORMAL, how often is it right? **Precision: 98.8%**")
    st.write("- Out of all actual NORMAL cases, how many did the model catch? **Recall: 71.4%**")
    st.write("- A single score that balances precision and recall. **F1-Score**: 82.9%**")
    
with tab_about_dataset:
    st.write("## Dataset Description")
    
    
    st.write("### Context")
    st.write("[Link to source article](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)")
    st.write("The dataset contains chest X-ray images used to analyze cases of pneumonia. It shows examples of normal lungs, bacterial pneumonia, and viral pneumonia.")
    path = 'data\\Examples.png'
    st.image(path)
    
    st.write("### Content")
    st.write("- The dataset is organized into three folders: train, test, and validation.")
    st.write("- Each folder contains subfolders for each category: Pneumonia and Normal.")
    st.write("- Contains 5,863 X-Ray images (JPEG format) labeled as Pneumonia or Normal.")
    st.write("- Images were selected from pediatric patients aged 1 to 5 years at Guangzhou Women and Children’s Medical Center.")
    st.write("- All chest X-rays were checked for quality, removing low-quality or unreadable scans. Diagnoses were reviewed by two experts and verified by a third to ensure accuracy.")

    st.write("### Acknowledgements")
    st.write("- Data Source: [Mendeley Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)")
    st.write("- License: CC BY 4.0")
    st.write("- Citation: [Cell Journal](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)")

    st.write("### Inspiration")
    st.write("This dataset can be used to develop automated methods to detect and classify human diseases from medical images.")
with tab_about_model:
    # Title of the app
    st.title("VGG16 Model Overview")
    
    # Description of the VGG16 model
    st.header("Introduction")
    st.write("""
    VGG16 is a convolutional neural network (CNN) architecture introduced by the Visual Geometry Group (VGG) at the University of Oxford. It is renowned for its simplicity and effectiveness in image recognition tasks. The model achieved excellent results in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2014.
    """)
    
    st.header("Key Features")
    st.write("""
    1. **Depth:** VGG16 has 16 layers with learnable parameters, including 13 convolutional layers and 3 fully connected layers.
    2. **Convolutional Layers:** 
       - Uses small receptive fields (3x3 kernels) with stride 1, which captures fine-grained features.
       - Incorporates padding to maintain spatial resolution.
    3. **Pooling:** 
       - Max pooling layers (2x2 kernels with stride 2) reduce spatial dimensions.
    4. **Fully Connected Layers:** 
       - Three dense layers towards the end: the first two have 4096 units each, and the last layer outputs class probabilities.
    5. **Activation Function:** 
       - Relu (Rectified Linear Unit) activation is used throughout to introduce non-linearity.
    6. **Parameters:**
       - About 138 million trainable parameters, making it computationally intensive.
    """)
    
    st.header("Advantages")
    st.write("""
    - Excellent feature extraction due to deep and consistent architecture.
    - Simple and modular design, making it a popular choice for transfer learning tasks.
    """)
    
    st.header("Limitations")
    st.write("""
    - High computational cost and memory requirements due to the large number of parameters.
    - Less efficient compared to modern architectures like ResNet or EfficientNet.
    """)
    
    st.info("Despite its age, VGG16 remains a cornerstone in computer vision and is widely used in transfer learning for tasks like image classification, object detection, and more.")