# --- Predicting Survival Outcomes in Patients with Heart Failure Using Clinical Data ---
#
# This application predicts the survival outcomes of patients with heart failure using clinical data
# and advanced machine learning techniques. By analyzing key health indicators, the tool
# provides timely and accurate prognostic insights, helping healthcare professionals make
# informed decisions and improving patient care. User-friendly and data-driven, it transforms 
# complex medical data into clear predictions to support life-saving interventions.
#
# Developed by Thoriq Putra Belligan & Daffa Pratama Putra Ryantoso
# Course: Machine Learning 
# Lecturer: Budi Sumanto, S.Si., M.Eng.
# Vocational College of Universitas Gadjah Mada 2025

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import sys
import os

# --- Paths and Model Loading ---
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "heart_failure_nn_model.h5")
scaler_path = os.path.join(BASE_DIR, "scaler.save")
logo_path = os.path.join(BASE_DIR, "logo.png")  # Rectangle logo

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# --- Features and UI Settings ---
FEATURES = [
    ('Age (years)', float),
    ('Ejection fraction (percentage)', int),
    ('Serum creatinine (mg/dL)', float),
    ('Serum sodium (mEq/L)', int),
]

BG_COLOR = "#f4f4f9"
BTN_COLOR = "#a8d0e6"
BTN_HOVER = "#89bdd3"
ENTRY_BG = "#ffffff"
TITLE_COLOR = "#374785"
FONT_LABEL = ("Verdana", 10, "bold")
FONT_ENTRY = ("Verdana", 10)
FONT_BUTTON = ("Verdana", 10, "bold")
FONT_TITLE = ("Verdana", 18, "bold")

# --- Prediction Logic ---
def predict():
    try:
        input_data = []
        for (label, dtype), entry in zip(FEATURES, entries):
            value = entry.get().strip()
            entry.config(bg=ENTRY_BG)
            if value == "":
                entry.config(bg="#ffe6e6")
                messagebox.showwarning("Input Error", f"Please fill in: {label}")
                return
            input_data.append(dtype(value))
        array = np.array(input_data).reshape(1, -1)
        scaled = scaler.transform(array)
        prob = model.predict(scaled)[0][0]
        prediction = int(prob > 0.5)
        result = f"🩺 Risk Classification: {prediction}\n📊 Death Probability: {prob:.2%}"
        messagebox.showinfo("Prediction", result)
    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid value.\n{e}")

def clear_fields():
    """Clear all input fields and reset their background."""
    for entry in entries:
        entry.delete(0, tk.END)
        entry.config(bg=ENTRY_BG)
    entries[0].focus_set()

def on_enter(e):
    """Button hover effect."""
    e.widget.config(bg=BTN_HOVER)

def on_leave(e):
    """Button leave effect."""
    e.widget.config(bg=BTN_COLOR)

# --- GUI Setup ---
root = tk.Tk()
root.title("Heart Failure Predictor")
root.geometry("500x400")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# Logo at top left
logo_frame = tk.Frame(root, bg=BG_COLOR)
logo_frame.pack(pady=(10, 15), anchor='w', fill='x')
try:
    logo_img = Image.open(logo_path)
    base_height = 40  # Desired logo height
    h_percent = (base_height / float(logo_img.size[1]))
    w_size = int((float(logo_img.size[0]) * float(h_percent)))
    logo_img = logo_img.resize((w_size, base_height), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(logo_frame, image=logo_photo, bg=BG_COLOR)
    logo_label.image = logo_photo  # Keep a reference
    logo_label.pack(side='left', padx=(10, 0))
except Exception as e:
    print(f"Error loading logo: {e}")

# Title
tk.Label(
    root,
    text="Heart Failure Predictor",
    font=FONT_TITLE,
    fg=TITLE_COLOR,
    bg=BG_COLOR
).pack(pady=10)

# Form for input fields
form_frame = tk.Frame(root, bg=BG_COLOR)
form_frame.pack(pady=10)

entries = []
for i, (label_text, _) in enumerate(FEATURES):
    tk.Label(
        form_frame,
        text=label_text,
        bg=BG_COLOR,
        font=FONT_LABEL,
        anchor='w',
        width=25
    ).grid(row=i, column=0, sticky='w', padx=10, pady=5)
    entry = tk.Entry(form_frame, width=25, font=FONT_ENTRY, bg=ENTRY_BG)
    entry.grid(row=i, column=1, padx=5, pady=5)
    # Bind Enter key for navigation or prediction
    if i < len(FEATURES) - 1:
        entry.bind("<Return>", lambda e, idx=i: entries[idx + 1].focus_set())
    else:
        entry.bind("<Return>", lambda e: predict())
    entries.append(entry)

# Buttons
btn_frame = tk.Frame(root, bg=BG_COLOR)
btn_frame.pack(pady=20)

predict_btn = tk.Button(
    btn_frame, text="🔍 Predict", font=FONT_BUTTON, bg=BTN_COLOR, width=15, command=predict
)
predict_btn.grid(row=0, column=0, padx=10)
predict_btn.bind("<Enter>", on_enter)
predict_btn.bind("<Leave>", on_leave)

clear_btn = tk.Button(
    btn_frame, text="🧹 Clear", font=FONT_BUTTON, bg=BTN_COLOR, width=15, command=clear_fields
)
clear_btn.grid(row=0, column=1, padx=10)
clear_btn.bind("<Enter>", on_enter)
clear_btn.bind("<Leave>", on_leave)

# Focus on first entry
entries[0].focus_set()

root.mainloop()
