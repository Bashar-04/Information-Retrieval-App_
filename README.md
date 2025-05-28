# Information Retrieval GUI Project

This is a simple GUI-based Information Retrieval (IR) system built using Python and CustomTkinter. It supports multiple IR models to search through PDF documents:

- **Boolean Model**
- **Vector Space Model (TF-IDF)**
- **BM25 Model**

## 📂 Features
- Upload and parse multiple PDF documents.
- Clean and preprocess text (remove stop words, lowercase, punctuation).
- Search using different IR models.
- View ranked results with similarity scores.

## 🚀 How to Run

1. Clone the repository:

```
git clone https://github.com/yourusername/IR_GUI_Project.git
cd IR_GUI_Project
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
python Bashar_project(IR).py
```

## 📋 Requirements

- Python 3.8+
- Libraries: see `requirements.txt`

## 🧪 Models Explained

- **Boolean Model**: Checks presence/absence of query terms (binary).
- **TF-IDF (VSM)**: Weights terms by importance across documents.
- **BM25**: Enhances TF-IDF by considering document length and term saturation.

## 📍 Author
Created by Bashar

## 📝 License
MIT License (or your preferred license)
