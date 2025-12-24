from __future__ import annotations

import argparse
import json
import os
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from preprocessing import clean_text, tokenize_vietnamese

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_data_from_folders(data_path: str) -> Tuple[List[str], List[str], List[str]]:
    documents: List[str] = []
    labels: List[str] = []

    data_path = os.path.expanduser(data_path)
    categories = [
        item
        for item in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, item))
    ]

    if not categories:
        raise RuntimeError(f"Không tìm thấy thư mục con nào trong {data_path}")

    for category in categories:
        category_path = os.path.join(data_path, category)
        files = [f for f in os.listdir(category_path) if f.endswith(".txt")]

        if not files:
            print(f"⚠️  Bỏ qua '{category}': không có file .txt")
            continue

        for file_name in files:
            file_path = os.path.join(category_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        documents.append(content)
                        labels.append(category)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Lỗi đọc {file_path}: {exc}")

    if not documents:
        raise RuntimeError("Không có dữ liệu văn bản hợp lệ được tải")

    return documents, labels, categories


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def get_doc_vector_w2v(tokens: List[str], model: Word2Vec) -> np.ndarray:
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def get_doc_vector_d2v(tokens: List[str], model: Doc2Vec) -> np.ndarray:
    return model.infer_vector(tokens)


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train_models(data_dir: str, model_dir: str, test_size: float = 0.2) -> Dict:
    print(f"📂 Đang nạp dữ liệu từ: {data_dir}")
    documents, labels, categories = load_data_from_folders(data_dir)

    df = pd.DataFrame({"text": documents, "label": labels})
    df["clean"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean"].apply(tokenize_vietnamese)

    X_text = df["clean"].values
    X_tokens = df["tokens"].values
    y = df["label"].values

    (
        X_text_train,
        X_text_test,
        X_tokens_train,
        X_tokens_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_text,
        X_tokens,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    summary = {
        "total_samples": len(df),
        "labels": {label: int(sum(y == label)) for label in np.unique(y)},
        "train_samples": len(X_text_train),
        "test_samples": len(X_text_test),
        "categories": categories,
    }

    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    # ------------------------------ TF-IDF ------------------------------
    print("\n🧮 Huấn luyện TF-IDF + classifiers")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_text_train)
    X_test_tfidf = tfidf.transform(X_text_test)

    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.joblib"))

    tfidf_classifiers = {
        "Rocchio": NearestCentroid(),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": MultinomialNB(),
    }

    for name, classifier in tfidf_classifiers.items():
        classifier.fit(X_train_tfidf, y_train)
        y_pred = classifier.predict(X_test_tfidf)

        joblib.dump(
            classifier,
            os.path.join(model_dir, f"tfidf_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"),
        )

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append(
            {
                "representation": "TF-IDF",
                "classifier": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
            }
        )

    # ----------------------------- Word2Vec -----------------------------
    print("\n🧠 Huấn luyện Word2Vec + classifiers")
    w2v_model = Word2Vec(
        sentences=[list(tokens) for tokens in X_tokens_train],
        vector_size=200,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
    )
    w2v_model.save(os.path.join(model_dir, "word2vec.model"))

    X_train_w2v = np.array([get_doc_vector_w2v(tokens, w2v_model) for tokens in X_tokens_train])
    X_test_w2v = np.array([get_doc_vector_w2v(tokens, w2v_model) for tokens in X_tokens_test])

    w2v_classifiers = {
        "Rocchio": NearestCentroid(),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    for name, classifier in w2v_classifiers.items():
        classifier.fit(X_train_w2v, y_train)
        y_pred = classifier.predict(X_test_w2v)

        joblib.dump(
            classifier,
            os.path.join(model_dir, f"w2v_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"),
        )

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append(
            {
                "representation": "Word2Vec",
                "classifier": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
            }
        )

    # ----------------------------- Doc2Vec ------------------------------
    print("\n📄 Huấn luyện Doc2Vec + classifiers")
    tagged_train = [
        TaggedDocument(words=list(tokens), tags=[str(i)])
        for i, tokens in enumerate(X_tokens_train)
    ]

    d2v_model = Doc2Vec(
        vector_size=200,
        window=5,
        min_count=2,
        workers=4,
        dm=1,
        epochs=20,
        dbow_words=1,
    )
    d2v_model.build_vocab(tagged_train)
    d2v_model.train(tagged_train, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

    d2v_model.save(os.path.join(model_dir, "doc2vec.model"))

    X_train_d2v = np.array([d2v_model.dv[str(i)] for i in range(len(tagged_train))])
    X_test_d2v = np.array([get_doc_vector_d2v(tokens, d2v_model) for tokens in X_tokens_test])

    d2v_classifiers = {
        "Rocchio": NearestCentroid(),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    for name, classifier in d2v_classifiers.items():
        classifier.fit(X_train_d2v, y_train)
        y_pred = classifier.predict(X_test_d2v)

        joblib.dump(
            classifier,
            os.path.join(model_dir, f"d2v_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"),
        )

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append(
            {
                "representation": "Doc2Vec",
                "classifier": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
            }
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_summary": summary,
        "results": results,
    }

    metrics_path = os.path.join(model_dir, "evaluation_results.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu mô hình và thống kê vào {model_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện và lưu các mô hình phân loại văn bản")
    parser.add_argument("data_dir", help="Đường dẫn tới thư mục dữ liệu (mỗi nhãn là một thư mục con)")
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Thư mục lưu mô hình và kết quả (mặc định: models)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Tỷ lệ dữ liệu test (mặc định 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(args.data_dir, args.model_dir, args.test_size)


if __name__ == "__main__":
    main()
