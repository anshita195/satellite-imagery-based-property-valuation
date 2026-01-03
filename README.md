# Satellite Imagery Based Property Valuation

A multimodal regression pipeline that predicts property market value using both tabular data and satellite imagery. This project integrates traditional housing features with visual environmental context extracted from satellite images to improve property valuation accuracy.

## Project Overview

This project addresses the challenge of property valuation by combining two distinct data modalities:
- **Tabular Data**: Traditional property features (bedrooms, bathrooms, square footage, location coordinates, etc.)
- **Satellite Imagery**: Visual environmental context captured from satellite images using property coordinates

The goal is to build a multimodal deep learning model that leverages both data types to predict property prices more accurately than traditional tabular-only approaches.

## Project Structure

```
submission/
├── data_fetcher.py              # Script to download satellite images from API
├── eda_and_baseline.ipynb       # Exploratory data analysis and baseline model
├── model_training.ipynb         # Multimodal model training and evaluation
├── submission.csv                # Final predictions on test dataset
├── model_architecture.png        # Architecture diagram of the multimodal model
└── README.md                     # This file
```

## Requirements

### Python Packages
- TensorFlow (2.x)
- Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- Requests
- python-dotenv

### API Access
- Google Maps Static API key OR Mapbox Static Images API key

### Hardware
- GPU recommended for model training (can run on CPU but will be slower)
- Sufficient storage for satellite images (approximately 256x256 pixels per image)

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python requests python-dotenv
```

3. Set up API credentials:
   - Create a `.env` file in the project root
   - Add your API key:
     ```
     MAPS_API_KEY=your_api_key_here
     ```
   - Alternatively, set as environment variable:
     ```bash
     export MAPS_API_KEY=your_api_key_here
     ```

## Usage

### Step 1: Download Satellite Images

Run the data fetcher script to download satellite images for all properties:

```bash
python data_fetcher.py
```

This script will:
- Read the training and test CSV files
- For each property, fetch a satellite image using its latitude/longitude coordinates
- Save images to `images_train/` and `images_test/` directories
- Use rate limiting to avoid API quota issues

**Note**: Ensure your CSV files (`train(1)(train(1)).csv` and `test2(test(1)).csv`) are in the project directory before running.

### Step 2: Exploratory Data Analysis and Baseline

Open `eda_and_baseline.ipynb` in Jupyter Notebook or Google Colab:

1. Load and explore the dataset
2. Visualize price distributions and correlations
3. Perform geospatial analysis
4. Train a baseline Random Forest model (tabular data only)
5. Evaluate baseline performance

**Expected Baseline Results:**
- RMSE: ~$130,640
- MAE: ~$71,824
- R² Score: ~0.864

### Step 3: Multimodal Model Training

Open `model_training.ipynb` in Jupyter Notebook or Google Colab:

1. **Data Loading and Preprocessing** (Cell 2-3):
   - Load tabular data and preprocess features
   - Standardize numerical features
   - Create image path mappings
   - Set up data generators for efficient loading

2. **Model Architecture** (Cell 4-5):
   - Define multimodal architecture:
     - Tabular branch: Dense layers for numerical features
     - Image branch: Convolutional Neural Network (CNN) for satellite imagery
     - Fusion layer: Concatenation of both branches
     - Regression head: Final dense layers for price prediction
   - Visualize model architecture

3. **Model Training** (Cell 6):
   - Train the multimodal model
   - Monitor training and validation loss
   - Adjust hyperparameters as needed

4. **Model Explainability** (Cell 7):
   - Generate Grad-CAM visualizations
   - Identify which regions in satellite images influence predictions
   - Analyze model attention patterns

5. **Prediction and Evaluation** (Cell 8-9):
   - Generate predictions on test set
   - Evaluate model performance on validation set
   - Compare with baseline model

6. **Generate Submission File** (Cell 8):
   - Create `submission.csv` with predictions in required format

## Model Architecture

The multimodal model consists of:

1. **Tabular Input Branch**:
   - Input: 18 numerical features (bedrooms, bathrooms, square footage, etc.)
   - Processing: Two dense layers (64 → 32 neurons) with ReLU activation

2. **Image Input Branch**:
   - Input: 128x128x3 RGB satellite images
   - Processing: 
     - Conv2D layer (32 filters)
     - MaxPooling2D
     - Conv2D layer (64 filters) - named for Grad-CAM
     - MaxPooling2D
     - Flatten
     - Dense layer (64 neurons)

3. **Fusion Layer**:
   - Concatenation of tabular (32) and image (64) embeddings
   - Combined feature vector: 96 dimensions

4. **Regression Head**:
   - Dense layer (64 neurons)
   - Output layer: Single neuron with linear activation for price prediction

## Results

### Baseline Model (Tabular Only)
- **RMSE**: $130,639.85
- **MAE**: $71,824.34
- **R² Score**: 0.8640

### Multimodal Model (Tabular + Satellite Images)
- **RMSE**: $171,712.20
- **MAE**: $108,564.51
- **R² Score**: 0.7653

**Note**: The multimodal model performance suggests that further hyperparameter tuning and architecture optimization may be needed. The integration of satellite imagery shows potential but requires additional refinement to outperform the baseline.

## Key Features

1. **Programmatic Image Acquisition**: Automated satellite image fetching using property coordinates
2. **Multimodal Fusion**: Late fusion architecture combining tabular and image features
3. **Model Explainability**: Grad-CAM implementation to visualize which image regions influence predictions
4. **Efficient Data Loading**: Custom data generator for handling large image datasets
5. **Comprehensive Evaluation**: Comparison between baseline and multimodal approaches

## Dataset

The project uses a housing dataset with the following key features:
- **Target Variable**: `price` (property market value)
- **Tabular Features**: bedrooms, bathrooms, sqft_living, sqft_lot, condition, grade, view, waterfront, etc.
- **Geospatial Data**: latitude (`lat`) and longitude (`long`) for satellite image fetching
- **Training Set**: 16,209 samples
- **Test Set**: 5,404 samples

## Technical Details

- **Image Size**: 256x256 pixels (resized to 128x128 for model input)
- **Zoom Level**: 18 (high detail)
- **Image Provider**: Mapbox Static Images API (configurable to Google Maps)
- **Train/Validation Split**: 80/20
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)

## Future Improvements

1. Experiment with different CNN architectures (ResNet, EfficientNet)
2. Try early fusion or attention mechanisms for better feature integration
3. Implement data augmentation for satellite images
4. Fine-tune hyperparameters more systematically
5. Explore ensemble methods combining multiple models
6. Add more sophisticated feature engineering for tabular data

## Acknowledgments

- Satellite imagery provided by Mapbox Static Images API

