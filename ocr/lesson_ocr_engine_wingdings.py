##https://nanonets.com/blog/build-your-own-ocr-engine-for-wingdings/
##https://github.com/balaramas/Wingding_to_English_OCR/blob/main/wingding_word_images.zip

import loguru
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import pandas as pd
from tqdm import tqdm

def pad_image(image, target_size=(224, 224)):
    """Pad image to target size while maintaining aspect ratio"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get current size
    width, height = image.size
    
    # Calculate padding
    aspect_ratio = width / height
    if aspect_ratio > 1:
        # Width is larger
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        # Height is larger
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    
    # Resize image maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with padding
    new_image = Image.new('RGB', target_size, (255, 255, 255))
    
    # Paste resized image in center
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    return new_image


def data_preprocess():
    df = pd.read_csv('metadata.csv')
    # Create output directory for processed images
    processed_dir = 'processed_images'
    os.makedirs(processed_dir, exist_ok=True)

    # Process each image
    new_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        # Load image
        img_path = row['image_path']
        img = Image.open(img_path)
        
        # Pad image
        processed_img = pad_image(img)
        
        # Save processed image
        new_path = os.path.join(processed_dir, f'processed_{os.path.basename(img_path)}')
        processed_img.save(new_path)
        new_paths.append(new_path)

    # Update dataframe with new paths
    df['processed_image_path'] = new_paths
    df.to_csv('processed_metadata.csv', index=False)
    
def build_datasets():
    df = pd.read_csv('processed_metadata.csv')

    # First split: train and temporary
    train_df, temp_df = train_test_split(df, train_size=0.7, random_state=42)

    # Second split: validation and test from temporary
    val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42)

    # Save splits to CSV
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
def plot_samples(df, num_samples=5, title="Sample Images"):
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.titlesize': 16,     # Subplot title font size
        'figure.titlesize': 20    # Main title font size
    })
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    fig.suptitle(title, fontsize=20, y=1.05)
    
    # Randomly sample images
    sample_df = df.sample(n=num_samples)
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img = Image.open(row['processed_image_path'])
        axes[idx].imshow(img)
        axes[idx].set_title(f"Label: {row['english_word_label']}", fontsize=16, pad=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def plot_prediction_samples(image_paths, true_labels, pred_labels, num_samples=10):
    # Set figure size and font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'figure.titlesize': 22
    })
    
    # Calculate grid dimensions
    num_rows = 2
    num_cols = 5
    num_samples = min(num_samples, len(image_paths))
    
    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
    fig.suptitle('Sample Predictions from Test Set', fontsize=22, y=1.05)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i in range(num_samples):
        ax = axes_flat[i]
        
        # Load and display image
        img = Image.open(image_paths[i])
        ax.imshow(img)
        
        # Create label text
        true_text = f"True: {true_labels[i]}"
        pred_text = f"Pred: {pred_labels[i]}"
        
        # Set color based on correctness
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        # Add text above image
        ax.set_title(f"{true_text}\n{pred_text}", 
                    fontsize=14,
                    color=color,
                    pad=10,
                    bbox=dict(facecolor='white', 
                             alpha=0.8,
                             edgecolor='none',
                             pad=3))
        
        # Remove axes
        ax.axis('off')
    
    # Remove any empty subplots
    for i in range(num_samples, num_rows * num_cols):
        fig.delaxes(axes_flat[i])
    
    plt.tight_layout()
    plt.show()   

    
class WingdingsDataset(Dataset):
    def __init__(self, csv_path, processor, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['processed_image_path'])
        label = row['english_word_label']
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Process label
        encoding = self.tokenizer(
            label,
            padding="max_length",
            max_length=16,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': pixel_values.squeeze(),
            'labels': encoding.input_ids.squeeze(),
            'text': label
        }
        
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Create datasets
    train_dataset = WingdingsDataset('train.csv', processor, tokenizer)
    val_dataset = WingdingsDataset('val.csv', processor, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 20 #(change according to need)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    # Save the model
    model.save_pretrained('wingdings_ocr_model')
    
    
def model_evaluation(test_loader,tokenizer,test_dataset,accuracy_score):
    # Load the trained model
    model = VisionEncoderDecoderModel.from_pretrained('wingdings_ocr_model')
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create test dataset and dataloader
    test_dataset = WingdingsDataset('test.csv', processor, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    # Evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    predictions = []
    ground_truth = []
    image_paths = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            texts = batch['text']
            
            outputs = model.generate(pixel_values)
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            predictions.extend(pred_texts)
            ground_truth.extend(texts)
            image_paths.extend([row['processed_image_path'] for _, row in test_dataset.df.iterrows()])

    # Calculate and print accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Display sample predictions in grid
    print("\nDisplaying sample predictions:")
    plot_prediction_samples(image_paths, ground_truth, predictions)
    
if __name__ == '__main__':
    loguru.logger.info(f"ocr engine started")
    
    
    

    
    