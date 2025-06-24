# Skin Disease Detection System

This is a Streamlit-based web application for detecting skin diseases using deep learning. The system uses a pre-trained model to analyze skin images and provide predictions along with confidence scores.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir models
mkdir models_improved
mkdir preprocessed_data
mkdir uploaded_images
```

## Running the Application

1. Make sure your virtual environment is activated

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## Default Login Credentials

- Username: admin
- Password: admin123

## Project Structure

- `app.py`: Main application file
- `models/`: Directory for storing model files
- `models_improved/`: Directory for storing improved model files
- `preprocessed_data/`: Directory for storing preprocessed data
- `uploaded_images/`: Directory for storing uploaded images
- `skin_disease_app.db`: SQLite database file (created automatically)

## Note

Before running the application, make sure you have the required model files in the `models/` and `models_improved/` directories. The application expects:
- `models/best_model.h5`
- `models_improved/best_model.h5`
- `preprocessed_data/classes.npy`

If you don't have these files, you'll need to obtain them or train the models first.

## System Overview

This system provides a complete solution for skin disease detection:

1. **Web Application**: User-friendly interface with authentication, disease detection, and feedback
2. **Deep Learning Model**: Transfer learning with EfficientNetB0 for accurate skin disease classification
3. **Admin Dashboard**: Manage users, doctors, and review feedback
4. **Doctor Recommendations**: Suggests specialists based on detected conditions

## Supported Skin Conditions

- Atopic Dermatitis
- Basal Cell Carcinoma
- Benign Keratosis
- Eczema
- Fungal Infections
- Melanocytic Nevi
- Melanoma
- Psoriasis
- Seborrheic Keratoses
- Viral Infections

## Key Features

### Authentication & User Management
- User registration and login
- Password reset functionality
- Role-based access control (User/Admin)

### Disease Detection Workflow
- Image upload and preprocessing
- Analysis using the CNN model
- Results visualization with confidence metrics
- Prediction history tracking

### Feedback Mechanism
- Submit feedback on prediction accuracy
- Rate the system's performance
- Administrator review of user feedback

### Doctor Integration
- Database of specialists by condition
- Contact information for doctors
- Automatic doctor recommendations based on detected conditions

### Admin Dashboard
- User management (add/update users)
- Doctor management (add/update/delete doctors)
- Feedback analysis with visualizations

## System Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- OpenCV
- SQLite
- Other dependencies listed in requirements.txt

## Project Structure

- **Web Application**
  - `app.py` - Streamlit web application
  - `setup_database.py` - Database initialization script
  - `run_app.bat`/`run_app.sh` - Launcher scripts

- **Data Processing**
  - `preprocess.py` - Data preprocessing pipeline
  - `split_dataset.py` - Dataset splitting utility

- **Model Training**
  - `train.py` - Standard model training
  - `train_improved.py` - Enhanced model with class weighting

- **Model Inference**
  - `predict.py` - Standard prediction script
  - `predict_improved.py` - Enhanced prediction with better visualization

- **Utility Scripts**
  - `run_pipeline.py` - Complete pipeline orchestration
  - `verify_setup.py` - System verification tool

## Installation & Setup

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Set up the database**:
   ```
   python setup_database.py
   ```

3. **Verify dataset structure**:
   Make sure your dataset is in the `dataset/` folder with each skin condition in its own subdirectory.

4. **Preprocess the data** (if not already done):
   ```
   python preprocess.py
   ```

5. **Train the model** (if not already done):
   ```
   python train_improved.py
   ```

## Running the Application

### Web Interface

Start the web application:
```
streamlit run app.py
```

Or use the provided scripts:
- Windows: `run_app.bat`
- Linux/Mac: `bash run_app.sh`

### Default Credentials

- **Admin User**:
  - Username: `admin`
  - Password: `admin123`

- **Test User**:
  - Username: `testuser`
  - Password: `password123`

### Command-Line Pipeline

For the complete pipeline:
```
python run_pipeline.py
```

With options:
```
python run_pipeline.py --use-improved  # Use improved model
python run_pipeline.py --skip-preprocess  # Skip preprocessing
```

## Troubleshooting

- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Database issues**: Delete `skin_disease_app.db` and run `python setup_database.py`
- **Model loading errors**: Ensure models are in the correct directories (`models/` and `models_improved/`)
- **Image upload errors**: Check that the `uploaded_images` directory exists
- **Streamlit port conflicts**: Use `streamlit run app.py --server.port 8502`

## Development and Customization

- Add new skin conditions: Update preprocessing and retrain models
- Add doctor specialties: Update the doctors table in the database
- Customize UI: Modify the Streamlit app.py file
- Improve model: Adjust training parameters in train_improved.py

## Disclaimer

This system is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## License

This project is open source and available under the MIT License. 