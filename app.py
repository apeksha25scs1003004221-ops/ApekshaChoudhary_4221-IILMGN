import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
import joblib
import io

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")

st.title("📧 Spam Email Classifier")
st.markdown("""
Welcome to the **Advanced Spam Email Classifier**! Train multiple ML models, upload custom datasets, 
and compare performance with detailed visualizations.
""")

@st.cache_data
def load_sample_data():
    emails = [
        ("WINNER!! You have won $1,000,000! Click here to claim your prize now!", "spam"),
        ("Congratulations! You've been selected for a FREE iPhone. Act now!", "spam"),
        ("URGENT: Your account will be closed. Verify your information immediately.", "spam"),
        ("Make money fast! Work from home and earn $5000 per week!", "spam"),
        ("Buy cheap medications online. No prescription needed!", "spam"),
        ("You have inherited $5 million from a distant relative. Contact us!", "spam"),
        ("Hot singles in your area want to meet you! Click here!", "spam"),
        ("Lose 20 pounds in 2 weeks with this miracle pill!", "spam"),
        ("CONGRATULATIONS! You are our lucky winner. Claim your reward!", "spam"),
        ("Get a loan approved in 5 minutes. Bad credit OK!", "spam"),
        ("FREE GIFT CARD! Click now to receive your $100 Walmart card!", "spam"),
        ("Your computer is infected! Download this software now!", "spam"),
        ("Make thousands weekly from home. No experience required!", "spam"),
        ("SPECIAL OFFER: Luxury watches at 90% off. Limited time!", "spam"),
        ("You've won a free vacation to the Bahamas! Claim now!", "spam"),
        ("Enlarge your income with our proven system!", "spam"),
        ("ALERT: Suspicious activity on your account. Verify now!", "spam"),
        ("Get rich quick with this investment opportunity!", "spam"),
        ("FREE trial - Premium subscription. Cancel anytime!", "spam"),
        ("Your tax refund is ready. Click to claim $2,500!", "spam"),
        ("Hi John, can we reschedule our meeting to 3pm tomorrow?", "ham"),
        ("Your order #12345 has been shipped. Track your package here.", "ham"),
        ("Reminder: Team lunch today at noon in the cafeteria.", "ham"),
        ("Thanks for your email. I'll review the documents and get back to you.", "ham"),
        ("Your electricity bill for March is ready. Amount due: $85.50", "ham"),
        ("Meeting notes from yesterday are attached. Please review.", "ham"),
        ("Happy birthday! Hope you have a wonderful day!", "ham"),
        ("Your appointment with Dr. Smith is confirmed for May 15th at 2pm.", "ham"),
        ("Project deadline extended to Friday. Let me know if you have questions.", "ham"),
        ("Thank you for your purchase. Your receipt is attached.", "ham"),
        ("The quarterly report is due next Monday. Please submit on time.", "ham"),
        ("Your subscription renewal is coming up on June 1st.", "ham"),
        ("Great job on the presentation today! The client was impressed.", "ham"),
        ("Can you send me the latest version of the budget spreadsheet?", "ham"),
        ("Your flight to Boston is confirmed. Departure: 8am on Tuesday.", "ham"),
        ("Reminder: Please complete the annual survey by end of week.", "ham"),
        ("The conference room is booked for your meeting at 10am.", "ham"),
        ("Your package was delivered today at 2:35pm.", "ham"),
        ("Welcome to our newsletter! Here's this week's update.", "ham"),
        ("Your payment of $125.00 has been received. Thank you!", "ham"),
        ("Let me know when you're available for a quick call.", "ham"),
        ("The system maintenance is scheduled for Saturday night.", "ham"),
        ("Your library books are due next week. Please return or renew.", "ham"),
        ("Congratulations on your promotion! Well deserved.", "ham"),
        ("Your insurance policy renewal documents are attached.", "ham"),
    ]
    df = pd.DataFrame(emails, columns=['email', 'label'])
    return df

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def prepare_dataset(df):
    df = df.copy()
    df['cleaned_email'] = df['email'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_email'], df['label'], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_all_models(X_train, X_test, y_train, y_test):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred
        
        label_map = {'ham': 0, 'spam': 1}
        y_test_binary = [label_map[label] for label in y_test]
        fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'report': report,
            'y_pred': y_pred,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
    
    return results

def get_feature_importance(vectorizer, model, model_name, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    
    if model_name == 'Naive Bayes':
        log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_spam_indices = np.argsort(log_prob_diff)[-top_n:][::-1]
        top_ham_indices = np.argsort(log_prob_diff)[:top_n]
    elif model_name == 'Logistic Regression':
        coef = model.coef_[0]
        top_spam_indices = np.argsort(coef)[-top_n:][::-1]
        top_ham_indices = np.argsort(coef)[:top_n]
    elif model_name == 'Linear SVM':
        coef = model.coef_[0]
        top_spam_indices = np.argsort(coef)[-top_n:][::-1]
        top_ham_indices = np.argsort(coef)[:top_n]
    else:
        return None, None
    
    spam_features = [(feature_names[i], log_prob_diff[i] if model_name == 'Naive Bayes' else model.coef_[0][i]) 
                     for i in top_spam_indices]
    ham_features = [(feature_names[i], log_prob_diff[i] if model_name == 'Naive Bayes' else model.coef_[0][i]) 
                    for i in top_ham_indices]
    
    return spam_features, ham_features

if 'dataset' not in st.session_state:
    st.session_state.dataset = load_sample_data()
    st.session_state.dataset_name = "Sample Dataset"
    st.session_state.models_trained = False

if 'vectorizer' not in st.session_state or 'model_results' not in st.session_state:
    with st.spinner("Training models..."):
        X_train, X_test, y_train, y_test, vectorizer = prepare_dataset(st.session_state.dataset)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.vectorizer = vectorizer
        st.session_state.model_results = train_all_models(X_train, X_test, y_train, y_test)
        st.session_state.models_trained = True

with st.sidebar:
    st.header("📊 Quick Stats")
    best_model = max(st.session_state.model_results.items(), key=lambda x: x[1]['accuracy'])
    st.metric("Best Model", best_model[0])
    st.metric("Best Accuracy", f"{best_model[1]['accuracy']*100:.1f}%")
    st.metric("Dataset Size", len(st.session_state.dataset))
    st.metric("Active Dataset", st.session_state.dataset_name)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Dataset & Training", 
    "🔍 Test Classifier", 
    "📊 Model Comparison",
    "📈 ROC Curves",
    "🎯 Feature Importance",
    "💾 Export Model",
    "🧠 Learn More"
])

with tab1:
    st.header("Dataset Management & Model Training")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Upload Custom Dataset")
        st.markdown("Upload a CSV file with two columns: `email` and `label` (spam/ham)")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                df_upload.columns = df_upload.columns.str.lower().str.strip()
                
                if 'email' not in df_upload.columns or 'label' not in df_upload.columns:
                    st.error("CSV must have 'email' and 'label' columns!")
                else:
                    df_upload = df_upload[['email', 'label']].dropna()
                    df_upload['label'] = df_upload['label'].str.lower().str.strip()
                    
                    valid_labels = df_upload['label'].isin(['spam', 'ham'])
                    if not valid_labels.all():
                        st.warning(f"Found {(~valid_labels).sum()} invalid labels. Removing those rows.")
                        df_upload = df_upload[valid_labels]
                    
                    if len(df_upload) < 10:
                        st.error("Dataset too small! Need at least 10 emails.")
                    else:
                        st.success(f"Loaded {len(df_upload)} emails!")
                        
                        spam_count = (df_upload['label'] == 'spam').sum()
                        ham_count = (df_upload['label'] == 'ham').sum()
                        st.info(f"Spam: {spam_count}, Ham: {ham_count}")
                        
                        if st.button("🚀 Use This Dataset & Retrain Models", type="primary"):
                            st.session_state.dataset = df_upload
                            st.session_state.dataset_name = uploaded_file.name
                            
                            with st.spinner("Retraining all models..."):
                                X_train, X_test, y_train, y_test, vectorizer = prepare_dataset(df_upload)
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                st.session_state.vectorizer = vectorizer
                                st.session_state.model_results = train_all_models(X_train, X_test, y_train, y_test)
                                st.session_state.models_trained = True
                            
                            st.success("Models retrained successfully!")
                            st.rerun()
            
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
        
        if st.button("🔄 Reset to Sample Dataset"):
            st.session_state.dataset = load_sample_data()
            st.session_state.dataset_name = "Sample Dataset"
            
            with st.spinner("Retraining models..."):
                X_train, X_test, y_train, y_test, vectorizer = prepare_dataset(st.session_state.dataset)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.vectorizer = vectorizer
                st.session_state.model_results = train_all_models(X_train, X_test, y_train, y_test)
            
            st.success("Reset to sample dataset!")
            st.rerun()
    
    with col2:
        st.subheader("📋 Current Dataset Info")
        st.metric("Total Emails", len(st.session_state.dataset))
        spam_count = (st.session_state.dataset['label'] == 'spam').sum()
        ham_count = (st.session_state.dataset['label'] == 'ham').sum()
        
        col_a, col_b = st.columns(2)
        col_a.metric("Spam Emails", spam_count)
        col_b.metric("Ham Emails", ham_count)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Spam', 'Ham'],
            values=[spam_count, ham_count],
            marker_colors=['#dc3545', '#28a745']
        )])
        fig_pie.update_layout(title="Dataset Distribution", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎓 Model Training Results")
    
    results_data = []
    for name, result in st.session_state.model_results.items():
        results_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']*100:.2f}%",
            'Spam Precision': f"{result['report']['spam']['precision']*100:.1f}%",
            'Spam Recall': f"{result['report']['spam']['recall']*100:.1f}%",
            'ROC AUC': f"{result['roc_auc']:.3f}"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

with tab2:
    st.header("Test the Spam Classifier")
    
    model_choice = st.selectbox(
        "Select Model to Use:",
        list(st.session_state.model_results.keys())
    )
    
    user_input = st.text_area(
        "Email Text:",
        height=150,
        placeholder="Type or paste an email message here..."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        classify_button = st.button("🔍 Classify Email", type="primary", use_container_width=True)
    
    if classify_button and user_input:
        cleaned_input = preprocess_text(user_input)
        input_tfidf = st.session_state.vectorizer.transform([cleaned_input])
        
        model = st.session_state.model_results[model_choice]['model']
        prediction = model.predict(input_tfidf)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_tfidf)[0]
            ham_prob = probability[0] * 100
            spam_prob = probability[1] * 100
        else:
            spam_prob = 100 if prediction == 'spam' else 0
            ham_prob = 100 - spam_prob
        
        st.markdown("---")
        st.subheader("Classification Result:")
        
        if prediction == "spam":
            st.error(f"🚨 **This email is SPAM** (Confidence: {spam_prob:.1f}%)")
        else:
            st.success(f"✅ **This email is HAM (Not Spam)** (Confidence: {ham_prob:.1f}%)")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Ham (Not Spam)', 'Spam'],
                y=[ham_prob, spam_prob],
                marker_color=['#28a745', '#dc3545'],
                text=[f'{ham_prob:.1f}%', f'{spam_prob:.1f}%'],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title=f"Prediction Confidence ({model_choice})",
            yaxis_title="Probability (%)",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🔧 See Preprocessing Steps"):
            st.markdown(f"**Original Text:**\n```\n{user_input}\n```")
            st.markdown(f"**Cleaned Text:**\n```\n{cleaned_input}\n```")
    
    elif classify_button:
        st.warning("⚠️ Please enter some email text to classify.")

with tab3:
    st.header("Model Comparison")
    
    comparison_data = []
    for name, result in st.session_state.model_results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision (Spam)': result['report']['spam']['precision'],
            'Recall (Spam)': result['report']['spam']['recall'],
            'F1-Score (Spam)': result['report']['spam']['f1-score'],
            'ROC AUC': result['roc_auc']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    fig_comp = go.Figure()
    metrics = ['Accuracy', 'Precision (Spam)', 'Recall (Spam)', 'F1-Score (Spam)', 'ROC AUC']
    
    for _, row in comp_df.iterrows():
        fig_comp.add_trace(go.Bar(
            name=row['Model'],
            x=metrics,
            y=[row[m] for m in metrics],
            text=[f"{row[m]*100:.1f}%" for m in metrics],
            textposition='auto',
        ))
    
    fig_comp.update_layout(
        title="Model Performance Comparison",
        yaxis_title="Score",
        barmode='group',
        height=500
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Confusion Matrices")
    
    cols = st.columns(3)
    for idx, (name, result) in enumerate(st.session_state.model_results.items()):
        with cols[idx]:
            fig_cm = px.imshow(
                result['confusion_matrix'],
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Ham', 'Spam'],
                y=['Ham', 'Spam'],
                color_continuous_scale='Blues',
                text_auto=True,
                title=name
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

with tab4:
    st.header("ROC Curves")
    st.markdown("ROC curves show the trade-off between True Positive Rate and False Positive Rate")
    
    fig_roc = go.Figure()
    
    for name, result in st.session_state.model_results.items():
        fig_roc.add_trace(go.Scatter(
            x=result['fpr'],
            y=result['tpr'],
            name=f"{name} (AUC = {result['roc_auc']:.3f})",
            mode='lines',
            line=dict(width=2)
        ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig_roc.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=600,
        hovermode='closest'
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    with st.expander("ℹ️ Understanding ROC Curves"):
        st.markdown("""
        - **ROC Curve**: Shows model performance at different classification thresholds
        - **AUC (Area Under Curve)**: Higher is better (1.0 is perfect, 0.5 is random)
        - **Diagonal Line**: Represents a random classifier
        - **Closer to top-left**: Better performance (high True Positive, low False Positive)
        """)

with tab5:
    st.header("Feature Importance Analysis")
    st.markdown("Discover which words are most indicative of spam or ham emails")
    
    model_select = st.selectbox(
        "Select Model for Feature Analysis:",
        list(st.session_state.model_results.keys()),
        key="feature_model"
    )
    
    model = st.session_state.model_results[model_select]['model']
    spam_features, ham_features = get_feature_importance(
        st.session_state.vectorizer, 
        model, 
        model_select,
        top_n=15
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Top Spam Indicators")
        spam_words = [f[0] for f in spam_features]
        spam_scores = [f[1] for f in spam_features]
        
        fig_spam = go.Figure(go.Bar(
            x=spam_scores,
            y=spam_words,
            orientation='h',
            marker_color='#dc3545',
            text=[f"{s:.3f}" for s in spam_scores],
            textposition='auto'
        ))
        fig_spam.update_layout(
            title="Words Most Associated with Spam",
            xaxis_title="Importance Score",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_spam, use_container_width=True)
    
    with col2:
        st.subheader("✅ Top Ham Indicators")
        ham_words = [f[0] for f in ham_features]
        ham_scores = [f[1] for f in ham_features]
        
        fig_ham = go.Figure(go.Bar(
            x=ham_scores,
            y=ham_words,
            orientation='h',
            marker_color='#28a745',
            text=[f"{s:.3f}" for s in ham_scores],
            textposition='auto'
        ))
        fig_ham.update_layout(
            title="Words Most Associated with Ham",
            xaxis_title="Importance Score",
            height=500,
            yaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_ham, use_container_width=True)

with tab6:
    st.header("Export Models")
    
    st.subheader("💾 Download Trained Models")
    st.markdown("Save your trained models for future use or to share with others!")
    
    export_model = st.selectbox(
        "Select Model to Export:",
        list(st.session_state.model_results.keys()),
        key="export_select"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Model Accuracy", f"{st.session_state.model_results[export_model]['accuracy']*100:.1f}%")
        st.metric("ROC AUC", f"{st.session_state.model_results[export_model]['roc_auc']:.3f}")
    
    with col2:
        if st.button("📥 Prepare Download", type="primary", use_container_width=True):
            model_bundle = {
                'model': st.session_state.model_results[export_model]['model'],
                'vectorizer': st.session_state.vectorizer,
                'model_name': export_model,
                'accuracy': st.session_state.model_results[export_model]['accuracy'],
                'roc_auc': st.session_state.model_results[export_model]['roc_auc'],
                'dataset_size': len(st.session_state.dataset),
                'dataset_name': st.session_state.dataset_name
            }
            
            buffer = io.BytesIO()
            joblib.dump(model_bundle, buffer)
            buffer.seek(0)
            
            st.download_button(
                label=f"⬇️ Download {export_model}.pkl",
                data=buffer,
                file_name=f"spam_classifier_{export_model.lower().replace(' ', '_')}.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
            st.success(f"✅ {export_model} is ready to download!")
    
    st.markdown("---")
    st.info("""
    **What's included in the export:**
    - ✅ Trained machine learning model
    - ✅ TF-IDF vectorizer with learned vocabulary
    - ✅ Model performance metrics
    - ✅ Dataset information
    
    **Use cases:**
    - Save your best-performing model for production use
    - Share models with teammates or the community
    - Keep backups of different training runs
    - Deploy models in other Python applications
    """)
    
    with st.expander("📖 How to use exported models in your own code"):
        st.code("""
import joblib

# Load the model
model_bundle = joblib.load('spam_classifier_naive_bayes.pkl')
model = model_bundle['model']
vectorizer = model_bundle['vectorizer']

# Classify new emails
def classify_email(text):
    # Preprocess
    cleaned = text.lower()
    cleaned = ''.join(c for c in cleaned if c.isalpha() or c.isspace())
    
    # Vectorize and predict
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    return prediction  # Returns 'spam' or 'ham'

# Test it
result = classify_email("Win $1,000,000 now!")
print(result)  # 'spam'
        """, language="python")

with tab7:
    st.header("🧠 Understanding the Machine Learning Pipeline")
    
    st.markdown("""
    ### What is Spam Classification?
    Spam classification is a text classification problem where we train a computer to automatically 
    identify unwanted emails (spam) from legitimate ones (ham).
    
    ### The Models:
    """)
    
    st.markdown("#### 🎯 Naive Bayes")
    st.info("""
    A probabilistic classifier based on Bayes' theorem. Fast and effective for text classification.
    Best for: Quick training, good baseline performance.
    """)
    
    st.markdown("#### 📊 Logistic Regression")
    st.info("""
    A linear model that predicts probabilities using a logistic function.
    Best for: Interpretable results, balanced performance.
    """)
    
    st.markdown("#### ⚡ Linear SVM")
    st.info("""
    Support Vector Machine finds the optimal boundary between spam and ham.
    Best for: High-dimensional data, robust performance.
    """)
    
    st.markdown("---")
    st.markdown("### 📚 The Process:")
    
    steps = [
        ("1️⃣ Data Collection", "Gather labeled emails (spam/ham) for training"),
        ("2️⃣ Text Preprocessing", "Clean text: lowercase, remove punctuation, normalize"),
        ("3️⃣ Feature Extraction", "TF-IDF converts text to numerical features"),
        ("4️⃣ Model Training", "Train multiple algorithms on the data"),
        ("5️⃣ Evaluation", "Compare models using accuracy, ROC curves, confusion matrices"),
        ("6️⃣ Prediction", "Use the best model to classify new emails")
    ]
    
    for title, desc in steps:
        st.markdown(f"**{title}**")
        st.info(desc)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Advanced Spam Classifier with Model Comparison, Feature Analysis & Export/Import ❤️</p>
</div>
""", unsafe_allow_html=True)
