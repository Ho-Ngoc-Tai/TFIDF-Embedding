from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from gensim.models import Doc2Vec, Word2Vec
from pydantic import BaseModel

from preprocessing import clean_text, tokenize_vietnamese


app = FastAPI(title="Vietnamese Text Classification Inference Service")


class ModelRegistry:
    """Lazily loads vectorizers, embeddings và classifiers."""

    _tfidf_vectorizer = None
    _word2vec_model: Word2Vec | None = None
    _doc2vec_model: Doc2Vec | None = None
    _classifiers: Dict[str, Dict[str, Any]] | None = None

    @classmethod
    def load(cls) -> None:
        if cls._classifiers is not None:
            return

        model_base = "models"

        try:
            cls._tfidf_vectorizer = joblib.load(f"{model_base}/tfidf_vectorizer.joblib")
            cls._word2vec_model = Word2Vec.load(f"{model_base}/word2vec.model")
            cls._doc2vec_model = Doc2Vec.load(f"{model_base}/doc2vec.model")

            cls._classifiers = {
                "TF-IDF": {
                    "Rocchio": joblib.load(f"{model_base}/tfidf_Rocchio.joblib"),
                    "KNN (k=5)": joblib.load(f"{model_base}/tfidf_KNN_k=5.joblib"),
                    "Naive Bayes": joblib.load(f"{model_base}/tfidf_Naive_Bayes.joblib"),
                },
                "Word2Vec": {
                    "Rocchio": joblib.load(f"{model_base}/w2v_Rocchio.joblib"),
                    "KNN (k=5)": joblib.load(f"{model_base}/w2v_KNN_k=5.joblib"),
                    "Naive Bayes": joblib.load(f"{model_base}/w2v_Naive_Bayes.joblib"),
                },
                "Doc2Vec": {
                    "Rocchio": joblib.load(f"{model_base}/d2v_Rocchio.joblib"),
                    "KNN (k=5)": joblib.load(f"{model_base}/d2v_KNN_k=5.joblib"),
                    "Naive Bayes": joblib.load(f"{model_base}/d2v_Naive_Bayes.joblib"),
                },
            }
        except FileNotFoundError as exc:  # noqa: BLE001
            raise RuntimeError("Không tìm thấy mô hình đã huấn luyện. Hãy chạy train.py trước.") from exc

    @classmethod
    def tfidf_vectorizer(cls):
        cls.load()
        return cls._tfidf_vectorizer

    @classmethod
    def word2vec_model(cls) -> Word2Vec:
        cls.load()
        return cls._word2vec_model  # type: ignore[return-value]

    @classmethod
    def doc2vec_model(cls) -> Doc2Vec:
        cls.load()
        return cls._doc2vec_model  # type: ignore[return-value]

    @classmethod
    def classifiers(cls) -> Dict[str, Dict[str, Any]]:
        cls.load()
        return cls._classifiers  # type: ignore[return-value]


def get_doc_vector_w2v(tokens: List[str], model: Word2Vec) -> np.ndarray:
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def get_doc_vector_d2v(tokens: List[str], model: Doc2Vec) -> np.ndarray:
    return model.infer_vector(tokens)


class PredictRequest(BaseModel):
    text: str


class PipelineResult(BaseModel):
    representation: str
    classifier: str
    prediction: str
    confidence: float
    extra: Dict[str, Any] | None = None


class PredictResponse(BaseModel):
    input_length: int
    tokens_preview: List[str]
    results: List[PipelineResult]


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản đầu vào không được để trống")

    cleaned = clean_text(payload.text)
    tokens = tokenize_vietnamese(cleaned)

    if not tokens:
        raise HTTPException(status_code=400, detail="Không thể tách từ từ văn bản đầu vào")

    classifiers = ModelRegistry.classifiers()

    # TF-IDF vectors
    tfidf_vec = ModelRegistry.tfidf_vectorizer().transform([cleaned])

    # Word2Vec / Doc2Vec vectors
    w2v_vec = get_doc_vector_w2v(tokens, ModelRegistry.word2vec_model()).reshape(1, -1)
    d2v_vec = get_doc_vector_d2v(tokens, ModelRegistry.doc2vec_model()).reshape(1, -1)

    results: List[PipelineResult] = []

    for representation, clf_map in classifiers.items():
        for clf_name, clf in clf_map.items():
            if representation == "TF-IDF":
                features = tfidf_vec
            elif representation == "Word2Vec":
                features = w2v_vec
            else:
                features = d2v_vec

            prediction = clf.predict(features)[0]

            if hasattr(clf, "predict_proba"):
                proba = float(np.max(clf.predict_proba(features)))
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(features)
                proba = float(np.max(scores)) if np.ndim(scores) > 1 else float(scores)
            else:
                proba = 0.0

            extra: Dict[str, Any] | None = None
            if representation == "TF-IDF":
                vector = tfidf_vec.toarray()[0]
                vocab = ModelRegistry.tfidf_vectorizer().get_feature_names_out()
                top_idx = np.argsort(vector)[-5:][::-1]
                extra = {
                    "top_tokens": [
                        {"token": vocab[idx], "weight": round(float(vector[idx]), 4)}
                        for idx in top_idx
                        if vector[idx] > 0
                    ]
                }

            results.append(
                PipelineResult(
                    representation=representation,
                    classifier=clf_name,
                    prediction=prediction,
                    confidence=round(proba, 4),
                    extra=extra,
                )
            )

    return PredictResponse(
        input_length=len(payload.text),
        tokens_preview=tokens[:20],
        results=results,
    )
