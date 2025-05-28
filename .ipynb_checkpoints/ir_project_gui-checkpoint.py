# مشروع Information Retrieval باستخدام 3 نماذج وواجهة GUI

import os
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ======================== معالجة PDF ========================
def extract_text_from_pdfs(pdf_paths):
    docs = []
    for path in pdf_paths:
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        docs.append(text)
    return docs

# ======================== تنظيف النصوص ========================
def preprocess(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

# ======================== النماذج ========================

class BooleanModel:
    def __init__(self, docs):
        self.vectorizer = CountVectorizer(binary=True)
        self.doc_vectors = self.vectorizer.fit_transform(docs)
        self.features = self.vectorizer.get_feature_names_out()

    def query(self, q):
        q = preprocess(q)
        q_vec = self.vectorizer.transform([q])
        scores = (self.doc_vectors.multiply(q_vec)).sum(axis=1)
        return np.array(scores).ravel()

class VSMModel:
    def __init__(self, docs):
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def query(self, q):
        q = preprocess(q)
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.doc_vectors)
        return sims.ravel()

class BM25Model:
    def __init__(self, docs):
        self.tokenized = [doc.split() for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized)

    def query(self, q):
        q = preprocess(q).split()
        scores = self.bm25.get_scores(q)
        return scores

# ======================== واجهة GUI ========================

class IRApp:
    def __init__(self, master):
        self.master = master
        master.title("Information Retrieval System")

        self.docs = []
        self.filenames = []
        self.models = {}

        self.model_choice = tk.StringVar(value="Boolean")

        tk.Button(master, text="📁 تحميل ملفات PDF", command=self.load_pdfs).pack(pady=5)
        tk.Label(master, text="📝 اكتب الاستعلام:").pack()
        self.query_entry = tk.Entry(master, width=50)
        self.query_entry.pack(pady=5)

        tk.Label(master, text="🔍 اختر النموذج:").pack()
        for model in ["Boolean", "VSM", "BM25"]:
            tk.Radiobutton(master, text=model, variable=self.model_choice, value=model).pack(anchor='w')

        tk.Button(master, text="🔎 استرجاع", command=self.run_query).pack(pady=5)

        self.results = tk.Text(master, height=15, width=80)
        self.results.pack()

    def load_pdfs(self):
        paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if paths:
            raw_docs = extract_text_from_pdfs(paths)
            self.docs = [preprocess(doc) for doc in raw_docs]
            self.filenames = [os.path.basename(p) for p in paths]
            self.models = {
                "Boolean": BooleanModel(self.docs),
                "VSM": VSMModel(self.docs),
                "BM25": BM25Model(self.docs),
            }
            messagebox.showinfo("نجاح", f"تم تحميل {len(paths)} ملفات PDF")

    def run_query(self):
        query = self.query_entry.get()
        if not query:
            messagebox.showwarning("تحذير", "يرجى إدخال استعلام.")
            return

        model_name = self.model_choice.get()
        model = self.models.get(model_name)
        scores = model.query(query)
        ranked = sorted(zip(self.filenames, scores), key=lambda x: -x[1])

        self.results.delete('1.0', tk.END)
        self.results.insert(tk.END, f"نتائج النموذج: {model_name}\n\n")
        for filename, score in ranked:
            self.results.insert(tk.END, f"{filename}: {score:.4f}\n")

# ======================== تشغيل البرنامج ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = IRApp(root)
    root.mainloop()
