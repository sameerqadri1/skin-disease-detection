import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import seaborn as sns
import io
import pandas as pd
import hashlib
import sqlite3
import uuid
from datetime import datetime

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Skin Disease Detection System",
    page_icon="🔬",
    layout="wide"
)

# --- Custom CSS for Modern UI ---
st.markdown('''
    <style>
    body {
        background: #f6f8fa !important;
    }
    .main > div {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.07);
        padding: 2.5em 2em 2em 2em;
        margin-top: 2em;
        margin-bottom: 2em;
    }
    .stButton > button, .stForm button {
        background: linear-gradient(90deg, #1976d2 0%, #43a047 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6em 1.5em !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        box-shadow: 0 2px 8px 0 rgba(25,118,210,0.08);
        transition: background 0.2s;
    }
    .stButton > button:hover, .stForm button:hover {
        background: linear-gradient(90deg, #43a047 0%, #1976d2 100%) !important;
    }
    .stTextInput > div > input, .stTextArea > div > textarea {
        border-radius: 6px !important;
        border: 1.5px solid #e0e0e0 !important;
        padding: 0.5em !important;
        font-size: 1em !important;
    }
    .stDataFrame, .stTable {
        background: #fff !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px 0 rgba(25,118,210,0.04);
        padding: 1em !important;
    }
    .stSidebar {
        background: #232f3e !important;
        color: #fff !important;
        border-top-right-radius: 18px !important;
        border-bottom-right-radius: 18px !important;
    }
    .stSidebar .css-1d391kg, .stSidebar .css-1v0mbdj {
        color: #fff !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1976d2;
        font-weight: 700;
    }
    .stMarkdown a {
        color: #1976d2 !important;
        text-decoration: underline !important;
    }
    @media (max-width: 900px) {
        .main > div {
            padding: 1em 0.5em 1em 0.5em;
        }
    }
    </style>
''', unsafe_allow_html=True)

# Paths
MODEL_STANDARD = "models/best_model.h5"
MODEL_IMPROVED = "models_improved/best_model.h5"
CLASSES_PATH = "preprocessed_data/classes.npy"
DB_PATH = "skin_disease_app.db"

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create doctors table
    c.execute('''
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        specialization TEXT NOT NULL,
        contact TEXT NOT NULL,
        hospital TEXT,
        bio TEXT
    )
    ''')
    
    # Create feedback table
    c.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        prediction TEXT NOT NULL,
        actual_diagnosis TEXT,
        rating INTEGER,
        comments TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create predictions history table
    c.execute('''
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        image_path TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Insert default admin user if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        admin_password = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)",
                 ("admin", admin_password, "admin@skindisease.com", "admin"))
    
    # Insert some sample doctors if not exists
    c.execute("SELECT * FROM doctors LIMIT 1")
    if not c.fetchone():
        sample_doctors = [
            ("Dr. Sarah Johnson", "Melanoma, Skin Cancer", "sarah.j@hospital.com", "Central Hospital", "Specialist in melanoma and skin cancer with 10 years of experience"),
            ("Dr. Michael Chen", "Eczema, Dermatitis", "michael.c@hospital.com", "Dermatology Clinic", "Expert in treating various forms of dermatitis and eczema"),
            ("Dr. Priya Patel", "Psoriasis, Fungal Infections", "priya.p@hospital.com", "Skin Care Center", "Specializes in chronic skin conditions and fungal treatments"),
        ]
        c.executemany("INSERT INTO doctors (name, specialization, contact, hospital, bio) VALUES (?, ?, ?, ?, ?)", sample_doctors)
    
    conn.commit()
    conn.close()

# Initialize database at startup
init_db()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result and result[1] == hash_password(password):
        return {"id": result[0], "role": result[2]}
    return None

def register_user(username, email, password, role="user"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        c.execute("INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                 (username, email, password_hash, role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def reset_password_request(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, username FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    
    if result:
        # Here we would typically send an email, but for demo purposes
        # we'll just create a reset token and store it in session state
        reset_token = str(uuid.uuid4())
        st.session_state.reset_tokens = getattr(st.session_state, "reset_tokens", {})
        st.session_state.reset_tokens[reset_token] = {"user_id": result[0], "username": result[1], "timestamp": datetime.now()}
        
        # In a real application, send email with reset link
        # send_reset_email(email, result[1], reset_token)
        
        return reset_token
    return None

def reset_password(user_id, new_password):
    password_hash = hash_password(new_password)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    conn.commit()
    conn.close()

@st.cache_resource
def load_skin_model(model_path):
    """Load the trained model with caching to improve performance"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load the class names with caching"""
    try:
        if os.path.exists(CLASSES_PATH):
            classes = np.load(CLASSES_PATH, allow_pickle=True)
            return classes
        else:
            st.error(f"Classes file not found at {CLASSES_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return None

def preprocess_image(img):
    """Preprocess the image for prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_skin_disease(img_array, model, classes):
    """Predict skin disease from preprocessed image"""
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    predicted_class = classes[predicted_class_index]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(classes[i], predictions[0][i]) for i in top_indices]
    
    return predicted_class, confidence, top_3_predictions, predictions[0]

def save_uploaded_image(uploaded_file):
    """Save the uploaded image and return the file path"""
    upload_dir = "uploaded_images"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create unique filename
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def save_prediction_to_history(user_id, image_path, prediction, confidence):
    """Save prediction to the user's history"""
    if not user_id:
        return False
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute("""
        INSERT INTO prediction_history (user_id, image_path, prediction, confidence)
        VALUES (?, ?, ?, ?)
        """, (user_id, image_path, prediction, confidence))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        st.error(f"Error saving prediction: {e}")
        return False

def save_feedback(user_id, prediction, actual_diagnosis, rating, comments):
    """Save user feedback on prediction"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute("""
        INSERT INTO feedback (user_id, prediction, actual_diagnosis, rating, comments)
        VALUES (?, ?, ?, ?, ?)
        """, (user_id, prediction, actual_diagnosis, rating, comments))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        st.error(f"Error saving feedback: {e}")
        return False

def get_recommended_doctors(disease):
    """Get list of doctors specialized in the predicted disease"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Search for doctors with the disease in their specialization
    c.execute("""
    SELECT * FROM doctors
    WHERE specialization LIKE ?
    """, (f"%{disease}%",))
    
    doctors = c.fetchall()
    conn.close()
    
    # If no exact match, return all doctors
    if not doctors:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM doctors")
        doctors = c.fetchall()
        conn.close()
    
    return doctors

def login_page():
    """Display login page for user authentication (modern UI/UX)"""
    # Branding and logo
    st.markdown("""
        <div style='text-align:center;'>
            <img src='https://img.icons8.com/color/96/000000/dermatology.png' width='80'/>
            <h1 style='margin-bottom:0;'>Skin Disease Detection System</h1>
            <p style='color:#888; margin-top:0;'>AI-powered diagnosis & specialist recommendations</p>
        </div>
        <hr style='margin-bottom:2em;'>
    """, unsafe_allow_html=True)

    # If reset_password flag is set, show reset form instead
    if st.session_state.get("reset_password", False):
        reset_password_page()
        return

    # Layout: Login (left) | Register (right)
    login_col, reg_col = st.columns(2)

    # --- LOGIN ---
    with login_col:
        st.markdown("<h3 style='margin-bottom:0.5em;'>Sign In</h3>", unsafe_allow_html=True)
        login_form = st.form("login_form")
        username = login_form.text_input("Username", help="Enter your username")
        login_pw_show = login_form.checkbox("Show password", key="login_pw_show")
        password = login_form.text_input(
            "Password", type="default" if login_pw_show else "password", help="Enter your password"
        )
        remember_me = login_form.checkbox("Remember me", key="remember_me")
        login_submit = login_form.form_submit_button("Login", use_container_width=True)

        if login_submit:
            with st.spinner("Authenticating..."):
                user = verify_user(username, password)
            if user:
                st.session_state.is_authenticated = True
                st.session_state.user_id = user["id"]
                st.session_state.role = user["role"]
                st.session_state.username = username
                st.success("✅ **Login successful!**")
                st.rerun()
            else:
                st.error("❌ **Invalid username or password.**")

        # Proper clickable 'Forgot password?' link
        if st.button("Forgot password?", key="forgot_pw_btn"):
            st.session_state.reset_password = True
            st.rerun()

    # --- REGISTRATION ---
    with reg_col:
        st.markdown("<h3 style='margin-bottom:0.5em;'>Create Account</h3>", unsafe_allow_html=True)
        reg_form = st.form("reg_form")
        reg_username = reg_form.text_input("Username", help="Choose a unique username")
        reg_email = reg_form.text_input("Email", help="Enter a valid email address")
        reg_pw_show = reg_form.checkbox("Show password", key="reg_pw_show")
        reg_password = reg_form.text_input(
            "Password", type="default" if reg_pw_show else "password", help="At least 6 characters"
        )
        reg_confirm_password = reg_form.text_input(
            "Confirm Password", type="default" if reg_pw_show else "password", help="Re-enter your password"
        )
        # Password strength meter
        pw_strength = min(len(reg_password) / 12, 1.0) if reg_password else 0
        pw_color = "#e53935" if pw_strength < 0.5 else ("#fbc02d" if pw_strength < 0.8 else "#43a047")
        reg_form.markdown(f"""
            <div style='height:8px;background:#eee;border-radius:4px;margin-bottom:0.5em;'>
                <div style='width:{pw_strength*100:.0f}%;height:100%;background:{pw_color};border-radius:4px;'></div>
            </div>
        """, unsafe_allow_html=True)
        reg_submit = reg_form.form_submit_button("Register", use_container_width=True)

        # Email validation helper
        import re
        def is_valid_email(email):
            return re.match(r"[^@]+@[^@]+\.[^@]+", email)

        if reg_submit:
            if not reg_username or not reg_email or not reg_password or not reg_confirm_password:
                st.error("⚠️ **All fields are required.**")
            elif not is_valid_email(reg_email):
                st.error("⚠️ **Invalid email address.**")
            elif reg_password != reg_confirm_password:
                st.error("⚠️ **Passwords do not match!**")
            elif len(reg_password) < 6:
                st.error("⚠️ **Password must be at least 6 characters long.**")
            else:
                with st.spinner("Registering account..."):
                    success = register_user(reg_username, reg_email, reg_password)
                if success:
                    st.success("✅ **Registration successful!** You can now log in.")
                else:
                    st.error("❌ **Username or email already exists.**")

def reset_password_page():
    """Display password reset page (two-step: email -> token+new password)"""
    st.title("Reset Password")

    # Step 1: Request reset token
    if "reset_token_sent" not in st.session_state:
        st.session_state.reset_token_sent = False
    if "reset_token_value" not in st.session_state:
        st.session_state.reset_token_value = None
    if not st.session_state.reset_token_sent:
        with st.form("reset_email_form"):
            reset_email = st.text_input("Enter your email")
            send_link = st.form_submit_button("Send Reset Link")
        if send_link:
            reset_token = reset_password_request(reset_email)
            if reset_token:
                st.session_state.reset_token_sent = True
                st.session_state.reset_token_value = reset_token
                st.success("A password reset token has been generated.")
                st.info(f"For this demo, use this token to reset your password: **{reset_token}**")
            else:
                st.error("Email not found.")
        st.markdown("<small>Enter your registered email address. In a real app, a reset link would be sent to your email.</small>", unsafe_allow_html=True)
    # Step 2: Enter token and new password
    if st.session_state.reset_token_sent:
        st.markdown("---")
        st.subheader("Enter Reset Token and New Password")
        with st.form("reset_pw_form"):
            token = st.text_input("Enter Reset Token", value=st.session_state.reset_token_value or "")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            reset_pw_submit = st.form_submit_button("Reset Password")
        if reset_pw_submit:
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                reset_tokens = getattr(st.session_state, "reset_tokens", {})
                if token in reset_tokens:
                    user_data = reset_tokens[token]
                    reset_password(user_data["user_id"], new_password)
                    st.success("Password has been reset successfully! You can now login.")
                    del reset_tokens[token]
                    st.session_state.reset_token_sent = False
                    st.session_state.reset_token_value = None
                    st.session_state.reset_password = False
                else:
                    st.error("Invalid or expired token")
        if st.button("Back to Login"):
            st.session_state.reset_token_sent = False
            st.session_state.reset_token_value = None
            st.session_state.reset_password = False
            st.rerun()

def user_management():
    """Admin page for managing users"""
    st.title("User Management")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, username, email, role, created_at FROM users", conn)
    conn.close()
    
    st.dataframe(df)
    
    st.subheader("Add New User")
    
    col1, col2 = st.columns(2)
    with col1:
        new_username = st.text_input("Username")
        new_email = st.text_input("Email")
    with col2:
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["user", "admin"])
    
    if st.button("Add User"):
        if register_user(new_username, new_email, new_password, new_role):
            st.success("User added successfully!")
            st.rerun()
        else:
            st.error("Username or email already exists")
    
    st.subheader("Update User Role")
    
    user_id = st.number_input("User ID", min_value=1, step=1)
    role = st.selectbox("New Role", ["user", "admin"])
    
    if st.button("Update Role"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()
        conn.close()
        st.success("User role updated!")
        st.rerun()

def doctor_management():
    """Admin page for managing doctors"""
    st.title("Doctor Management")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM doctors", conn)
    conn.close()
    
    st.dataframe(df)
    
    st.subheader("Add New Doctor")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
        specialization = st.text_input("Specialization (comma separated)")
        contact = st.text_input("Contact Email/Phone")
    with col2:
        hospital = st.text_input("Hospital/Clinic")
        bio = st.text_area("Bio")
        image_path = st.text_input("Image Path (optional)")
    
    if st.button("Add Doctor"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT INTO doctors (name, specialization, contact, hospital, bio, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (name, specialization, contact, hospital, bio, image_path))
        conn.commit()
        conn.close()
        st.success("Doctor added successfully!")
        st.rerun()
    
    st.subheader("Update/Delete Doctor")
    
    doctor_id = st.number_input("Doctor ID", min_value=1, step=1)
    action = st.radio("Action", ["Update", "Delete"])
    
    if action == "Update":
        field = st.selectbox("Field to Update", ["name", "specialization", "contact", "hospital", "bio", "image_path"])
        value = st.text_input("New Value")
        
        if st.button("Update Doctor"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(f"UPDATE doctors SET {field} = ? WHERE id = ?", (value, doctor_id))
            conn.commit()
            conn.close()
            st.success("Doctor updated!")
            st.rerun()
    else:
        if st.button("Delete Doctor"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM doctors WHERE id = ?", (doctor_id,))
            conn.commit()
            conn.close()
            st.success("Doctor deleted!")
            st.rerun()

def feedback_management():
    """Admin page for viewing and managing feedback"""
    st.title("Feedback Management")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Join feedback with usernames
    c.execute("""
    SELECT f.id, u.username, f.prediction, f.actual_diagnosis, 
           f.rating, f.comments, f.created_at
    FROM feedback f
    LEFT JOIN users u ON f.user_id = u.id
    ORDER BY f.created_at DESC
    """)
    
    feedback_data = c.fetchall()
    conn.close()
    
    if feedback_data:
        df = pd.DataFrame(feedback_data, columns=[
            "ID", "User", "Prediction", "Actual Diagnosis", 
            "Rating", "Comments", "Date"
        ])
        
        st.dataframe(df)
        
        # Rating distribution chart
        st.subheader("Rating Distribution")
        rating_counts = df["Rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(rating_counts.index, rating_counts.values)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_xticks(range(1, 6))
        st.pyplot(fig)
        
        # Prediction accuracy analysis (when actual diagnosis is provided)
        accuracy_df = df[df["Actual Diagnosis"].notna()]
        if len(accuracy_df) > 0:
            st.subheader("Prediction Accuracy Analysis")
            accuracy_df["Accurate"] = accuracy_df.apply(
                lambda x: x["Prediction"].lower() == x["Actual Diagnosis"].lower(), axis=1
            )
            accuracy_rate = accuracy_df["Accurate"].mean() * 100
            st.metric("Prediction Accuracy", f"{accuracy_rate:.1f}%")
    else:
        st.info("No feedback data available yet.")

def user_profile():
    """Allow users to view and edit their profile"""
    st.title("Your Profile")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, email, role, created_at FROM users WHERE id = ?", (st.session_state.user_id,))
    user_data = c.fetchone()
    conn.close()
    
    if user_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Information")
            st.write(f"**Username:** {user_data[0]}")
            st.write(f"**Email:** {user_data[1]}")
            st.write(f"**Role:** {user_data[2]}")
            st.write(f"**Joined:** {user_data[3]}")
        
        with col2:
            st.subheader("Update Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Password"):
                # Verify current password
                user = verify_user(user_data[0], current_password)
                if not user:
                    st.error("Current password is incorrect")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 6:
                    st.error("New password must be at least 6 characters long")
                else:
                    reset_password(st.session_state.user_id, new_password)
                    st.success("Password updated successfully")
    else:
        st.error("User data not found")
    
    # Prediction history
    st.subheader("Your Prediction History")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    SELECT image_path, prediction, confidence, timestamp
    FROM prediction_history
    WHERE user_id = ?
    ORDER BY timestamp DESC
    """, (st.session_state.user_id,))
    
    history = c.fetchall()
    conn.close()
    
    if history:
        history_df = pd.DataFrame(history, columns=["Image", "Prediction", "Confidence", "Date"])
        def format_confidence(x):
            if isinstance(x, (float, int)):
                return f"{float(x):.2%}"
            elif isinstance(x, bytes):
                try:
                    return f"{float(x.decode()):.2%}"
                except Exception:
                    try:
                        return f"{float.fromhex(x.hex()):.2%}"
                    except Exception:
                        return "Invalid"
            else:
                try:
                    return f"{float(x):.2%}"
                except Exception:
                    return str(x)
        history_df["Confidence"] = history_df["Confidence"].apply(format_confidence)
        st.dataframe(history_df)
    else:
        st.info("You haven't made any predictions yet.")

def disease_detection():
    """Main disease detection functionality"""
    st.title("Skin Disease Detection")
    
    classes = load_class_names()
    if classes is None:
        st.error("Could not load class names. Please make sure preprocessing has been done.")
        st.info("Contact the administrator for assistance.")
        return
    
    # Sidebar for model selection
    with st.sidebar:
        model_option = st.radio(
            "Select Model Version:",
            ("Standard", "Improved (Recommended)")
        )
        
        # Display supported conditions
        with st.expander("Supported Skin Conditions"):
            for cls in classes:
                st.write(f"- {cls}")
    
    model_path = MODEL_IMPROVED if model_option == "Improved (Recommended)" else MODEL_STANDARD
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please contact the administrator.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a skin image for analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the image
        col1, col2 = st.columns(2)
        
        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Save the uploaded image
            image_path = save_uploaded_image(uploaded_file)
        
        # Load model
        model = load_skin_model(model_path)
        if model is None:
            st.error(f"Failed to load model from {model_path}")
            return
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            img_array = preprocess_image(img)
            predicted_class, confidence, top_3_predictions, all_predictions = predict_skin_disease(
                img_array, model, classes
            )
            
            # Save prediction to history
            if hasattr(st.session_state, "user_id"):
                save_prediction_to_history(
                    st.session_state.user_id,
                    image_path,
                    predicted_class,
                    confidence
                )
        
        # Display results
        with col2:
            st.success("Analysis Complete!")
            st.subheader("Prediction Results")
            
            st.metric("Predicted Condition", predicted_class)
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Display confidence gauge
            conf_level = "Low" if confidence < 0.5 else "Medium" if confidence < 0.8 else "High"
            conf_color = "red" if confidence < 0.5 else "orange" if confidence < 0.8 else "green"
            st.markdown(f"<p>Confidence Level: <span style='color:{conf_color};font-weight:bold'>{conf_level}</span></p>", unsafe_allow_html=True)
            
            # Plot top 3 predictions
            fig, ax = plt.subplots(figsize=(8, 4))
            class_names = [name for name, _ in top_3_predictions]
            class_probs = [prob for _, prob in top_3_predictions]
            y_pos = np.arange(len(class_names))
            
            bars = ax.barh(y_pos, class_probs, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Probability')
            ax.set_title('Top 3 Predictions')
            
            # Color the bars
            for i, bar in enumerate(bars):
                if i == 0:  # Highest confidence prediction
                    bar.set_color('green')
                else:
                    bar.set_color('lightblue')
            
            st.pyplot(fig)
            
            # Display warning for low confidence
            if confidence < 0.5:
                st.warning("⚠️ Low confidence prediction. Consider consulting a healthcare professional.")
        
        # Doctor recommendations
        st.subheader("Recommended Specialists")
        doctors = get_recommended_doctors(predicted_class)
        
        if doctors:
            doctor_cols = st.columns(min(3, len(doctors)))
            
            for i, doctor in enumerate(doctors[:3]):  # Show up to 3 doctors
                with doctor_cols[i]:
                    st.markdown(f"### {doctor[1]}")  # Name
                    st.write(f"**Specialization:** {doctor[2]}")
                    st.write(f"**Hospital:** {doctor[4]}")
                    st.write(f"**Contact:** {doctor[3]}")
        else:
            st.info("No specialists available for this condition.")
        
        # Feedback section
        st.subheader("Provide Feedback")
        st.write("Help us improve our system by providing feedback on this prediction.")
        
        actual_diagnosis = st.text_input("Actual Diagnosis (if known)")
        rating = st.slider("Rating (1-5)", 1, 5, 3)
        comments = st.text_area("Comments")
        
        if st.button("Submit Feedback"):
            if hasattr(st.session_state, "user_id"):
                if save_feedback(
                    st.session_state.user_id,
                    predicted_class,
                    actual_diagnosis,
                    rating,
                    comments
                ):
                    st.success("Thank you for your feedback!")
            else:
                st.error("Please log in to submit feedback")
    
    # Information section
    st.markdown("---")
    st.subheader("About the System")
    st.markdown("""
    This skin disease detection system uses a deep learning model based on EfficientNetB0
    architecture with transfer learning. The model has been trained on thousands of dermatological
    images and can recognize various skin conditions.
    
    **Disclaimer**: This application is for educational purposes only and should not be used as a
    substitute for professional medical advice, diagnosis, or treatment.
    """)

def main():
    # Initialize session state
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "main_page" not in st.session_state:
        st.session_state.main_page = "Home"

    # Check if user is logged in
    if not st.session_state.is_authenticated:
        # Show authentication page (no sidebar, no menu bar)
        if "reset_password" in st.session_state and st.session_state.reset_password:
            reset_password_page()
        else:
            login_page()
        return

    # --- Sidebar navigation (rollback to old style) ---
    with st.sidebar:
        st.markdown(f"""
            <div style='padding:0.5em 0;'>
                <span style='color:#bbb;'>Logged in as:</span><br>
                <b style='color:#fff;font-size:1.1em;'>{st.session_state.username}</b>
            </div>
        """, unsafe_allow_html=True)
        if st.session_state.role == "admin":
            st.sidebar.title("")
            nav = st.radio(
                "",
                ["Home", "Disease Detection", "User Profile", "Manage Users", "Manage Doctors", "Feedback Analysis"],
                index=["Home", "Disease Detection", "User Profile", "Manage Users", "Manage Doctors", "Feedback Analysis"].index(st.session_state.get("main_page", "Home")),
                key="admin_nav"
            )
        else:
            st.sidebar.title("")
            nav = st.radio(
                "",
                ["Home", "Disease Detection", "User Profile"],
                index=["Home", "Disease Detection", "User Profile"].index(st.session_state.get("main_page", "Home")),
                key="user_nav"
            )
        if st.sidebar.button("Logout"):
            st.session_state.is_authenticated = False
            st.session_state.user_id = None
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.main_page = "Home"
            st.rerun()
        st.session_state.main_page = nav

    # Main content area
    if st.session_state.main_page == "Home":
        st.markdown(f"""
            <div style='text-align:center;'>
                <img src='https://img.icons8.com/color/96/000000/dermatology.png' width='80'/>
                <h1 style='margin-bottom:0.2em;'>Welcome, {st.session_state.username.title()}!</h1>
                <p style='color:#888; margin-top:0;'>This is your dashboard. Use the quick links below to get started.</p>
            </div>
        """, unsafe_allow_html=True)
        # Quick links as cards
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔬 Disease Detection", use_container_width=True):
                st.session_state.main_page = "Disease Detection"
                st.rerun()
        with col2:
            if st.button("👤 User Profile", use_container_width=True):
                st.session_state.main_page = "User Profile"
                st.rerun()
        if st.session_state.role == "admin":
            with col3:
                if st.button("🛠️ Admin Tools", use_container_width=True):
                    st.session_state.main_page = "Manage Users"
                    st.rerun()
        st.markdown("---")
        with st.expander("About the System", expanded=True):
            st.markdown("""
            This skin disease detection system uses a deep learning model based on EfficientNetB0
            architecture with transfer learning. The model has been trained on thousands of dermatological
            images and can recognize various skin conditions.
            
            **Disclaimer**: This application is for educational purposes only and should not be used as a
            substitute for professional medical advice, diagnosis, or treatment.
            """)
    elif st.session_state.main_page == "Disease Detection":
        disease_detection()
    elif st.session_state.main_page == "User Profile":
        user_profile()
    elif st.session_state.main_page == "Manage Users":
        user_management()
    elif st.session_state.main_page == "Manage Doctors":
        doctor_management()
    elif st.session_state.main_page == "Feedback Analysis":
        feedback_management()

if __name__ == "__main__":
    main() 