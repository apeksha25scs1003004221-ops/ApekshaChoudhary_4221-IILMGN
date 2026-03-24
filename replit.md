# Overview

This is a **Spam Email Classifier** application built with Streamlit that allows users to train and compare multiple machine learning models for email spam detection. The application provides an interactive interface for training models on sample or custom datasets, making predictions on new emails, and visualizing model performance through various metrics and charts.

The primary purpose is to demonstrate and compare different classification algorithms (Naive Bayes, Logistic Regression, and SVM) for spam detection, with built-in visualizations for performance analysis.

## Recent Changes (November 10, 2025)

**Added Advanced Features:**
1. ✅ CSV upload functionality - Upload custom email datasets with automatic validation and model retraining
2. ✅ Model comparison - Compare Naive Bayes, Logistic Regression, and Linear SVM side-by-side
3. ✅ Feature importance visualization - See which words are most indicative of spam vs ham
4. ✅ ROC curve analysis - Visual comparison of all models with AUC scores
5. ✅ Model export - Download trained models with vectorizers for use in other applications
6. ✅ Enhanced UI - 7 tabs organizing all features: Dataset & Training, Test Classifier, Model Comparison, ROC Curves, Feature Importance, Export Model, Learn More

**Security Improvements:**
- Removed unsafe model import functionality to prevent pickle deserialization vulnerabilities
- Export-only workflow for model persistence

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application framework
- **Design Pattern**: Single-page application with interactive widgets
- **State Management**: Streamlit's native caching mechanism using `@st.cache_data` decorator for performance optimization
- **Layout**: Wide layout configuration for better visualization space
- **Rationale**: Streamlit was chosen for rapid prototyping and ease of creating interactive ML demos without requiring separate frontend/backend codebases

## Backend Architecture
- **Language**: Python 3.x
- **ML Pipeline**: Scikit-learn for preprocessing, training, and evaluation
  - Text vectorization using TfidfVectorizer (Term Frequency-Inverse Document Frequency)
  - Three classification algorithms: MultinomialNB, LogisticRegression, and LinearSVC
- **Data Processing**: Pandas for dataset management and NumPy for numerical operations
- **Model Persistence**: Joblib for saving/loading trained models
- **Rationale**: Scikit-learn provides a consistent API across multiple algorithms, making it ideal for model comparison tasks

## Data Architecture
- **Data Storage**: In-memory processing with sample data embedded in the application
- **Data Format**: Tuple-based structure (email_text, label) converted to Pandas DataFrames
- **Training/Test Split**: Configurable split ratio using scikit-learn's train_test_split
- **Text Preprocessing**: Regular expression-based cleaning (assumed, based on imported `re` module)
- **Feature Engineering**: TF-IDF vectorization converts text to numerical features
- **Rationale**: In-memory processing is sufficient for demo purposes; no persistent database needed for this use case

## Visualization Architecture
- **Library**: Plotly (both graph_objects and express modules)
- **Chart Types**: 
  - Performance comparison charts
  - Confusion matrices
  - ROC curves with AUC metrics
- **Interactivity**: Plotly provides interactive charts that users can zoom, pan, and hover over
- **Rationale**: Plotly was chosen over static visualization libraries for its interactivity and professional appearance in web applications

## Model Training Architecture
- **Multi-Model Approach**: Three different algorithms trained simultaneously for comparison
  - **Multinomial Naive Bayes**: Probabilistic classifier, fast and efficient for text
  - **Logistic Regression**: Linear classifier with probability estimates
  - **Linear SVM**: Maximum margin classifier for binary classification
- **Evaluation Metrics**: 
  - Accuracy score
  - Classification reports (precision, recall, F1-score)
  - Confusion matrices
  - ROC curves and AUC
- **Rationale**: Comparing multiple algorithms allows users to understand trade-offs between different approaches

# External Dependencies

## Python Libraries
- **streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning algorithms and utilities
  - TfidfVectorizer for text feature extraction
  - MultinomialNB, LogisticRegression, LinearSVC for classification
  - train_test_split for dataset splitting
  - accuracy_score, classification_report, confusion_matrix, roc_curve, auc for evaluation
- **plotly**: Interactive visualization library (graph_objects and express)
- **joblib**: Model serialization and deserialization
- **re**: Regular expressions for text preprocessing (Python standard library)
- **io**: Input/output operations (Python standard library)

## No External Services
This application runs entirely locally without external API calls, cloud services, or remote databases. All data processing and model training occur in-memory on the host machine.

## File System Dependencies
- **Model Persistence**: The application uses joblib to save trained models to disk and load them for later use
- **Dataset Upload**: Supports custom dataset uploads (implied by the architecture, though implementation details are in the truncated code)