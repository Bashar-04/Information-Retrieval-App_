
import os
import fitz  # مكتبة لقراءة ملفات PDF
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def read_pdfs(paths):
    content_list = []
    for file in paths:
        text = ""
        with fitz.open(file) as pdf:
            for page in pdf:
                text += page.get_text()
        content_list.append(text)
    return content_list

def clean_text(text):
    tokens = text.lower().split()
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

 # هذا النموذج بشتغل على فكرة: هل الكلمة موجودة أو لا برجع الناتج عل(0 أو 1)
 #ما بهتم في ترتيب
class BooleanModel:
    def __init__(self, docs):
        self.vectorizer = CountVectorizer(binary=True)
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def query(self, q):
        q_clean = clean_text(q)
        q_vector = self.vectorizer.transform([q_clean])
        score = (self.doc_vectors.multiply(q_vector)).sum(axis=1)
        return np.array(score).ravel()

 # هذا النموذج يستخدم TF-IDF لحساب وزن كل كلمة حسب أهميتها
# TF = عدد مرات تكرار الكلمة في الوثيقة
 # IDF = هل الكلمة نادرة أم منتشرة في باقي الوثائق

class VSMModel:
    def __init__(self, docs):
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def query(self, q):
        q_clean = clean_text(q)
        q_vector = self.vectorizer.transform([q_clean])
        return cosine_similarity(q_vector, self.doc_vectors).ravel()
# BM25 هو نموذج مشابه لـ TF-IDF لكنه أذكى
 # يأخذ بعين الاعتبار:
# - تكرار الكلمة (لكن بحد معين)
# - طول الوثيقة
 # - ندرة الكلمة
class BM25Model:
    def __init__(self, docs):
        tokenized = [doc.split() for doc in docs]
        self.model = BM25Okapi(tokenized)

    def query(self, q):
        q_words = clean_text(q).split()
        return self.model.get_scores(q_words)

class IRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Information Retrieval Project")
        self.geometry("850x600")

        self.docs = []
        self.filenames = []
        self.models = {}
        self.model_choice = ctk.StringVar(value="Boolean")

        title_label = ctk.CTkLabel(self, text="Information Retrieval System", font=("Segoe UI", 20, "bold"))
        title_label.pack(pady=15)

        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=10, padx=20, fill="x")

        self.load_button = ctk.CTkButton(input_frame, text="Upload PDF Files", command=self.load_pdfs)
        self.load_button.pack(pady=10)

        ctk.CTkLabel(input_frame, text="Enter your query:").pack(pady=5)
        self.query_entry = ctk.CTkEntry(input_frame, width=600)
        self.query_entry.pack(pady=5)

        ctk.CTkLabel(input_frame, text="Select a model:").pack(pady=5)
        model_frame = ctk.CTkFrame(input_frame)
        model_frame.pack(pady=5)

        for model in ["Boolean", "VSM", "BM25"]:
            ctk.CTkRadioButton(model_frame, text=model, variable=self.model_choice, value=model).pack(side="left", padx=10)

        self.search_button = ctk.CTkButton(self, text="Search", command=self.run_query)
        self.search_button.pack(pady=10)

        self.results_box = ctk.CTkTextbox(self, width=800, height=300, font=("Courier New", 12))
        self.results_box.pack(padx=20, pady=10)

    def load_pdfs(self):
        paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if paths:
            raw_docs = read_pdfs(paths)
            self.docs = [clean_text(doc) for doc in raw_docs]
            self.filenames = [os.path.basename(p) for p in paths]
            self.models = {
                "Boolean": BooleanModel(self.docs),
                "VSM": VSMModel(self.docs),
                "BM25": BM25Model(self.docs),
            }
            messagebox.showinfo("Done", str(len(paths)) + " PDF files loaded.")

    def run_query(self):
        query = self.query_entry.get()
        if not query:
            messagebox.showwarning("Warning", "Please enter a query first.")
            return

        model_name = self.model_choice.get()
        model = self.models.get(model_name)
        scores = model.query(query)
        ranked_results = sorted(zip(self.filenames, scores), key=lambda x: -x[1])

        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", "Search results using model: " + model_name + "\n\n")
        for filename, score in ranked_results:
            self.results_box.insert("end", filename + " — Score: " + str(round(score, 4)) + "\n")

if __name__ == "__main__":
    app = IRApp()
    app.mainloop()
