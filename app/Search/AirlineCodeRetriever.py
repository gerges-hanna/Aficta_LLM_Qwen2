import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

class AirlineCodeRetriever:
    _instance = None  # Singleton instance

    EN_STOPWORDS = {"airline", "airlines", "air", "aviation"}
    AR_STOPWORDS = {"خطوط", "جوية", "الخطوط", "طيران", "شركة", "الطيران"}
    SEARCH_BASE_MODEL = "sentence-transformers/LaBSE"

    def __new__(cls, dataset_path, name_en, name_ar, airline_code):
        if cls._instance is None:
            cls._instance = super(AirlineCodeRetriever, cls).__new__(cls)
            cls._instance._initialize(dataset_path, name_en, name_ar, airline_code)
        return cls._instance

    def _initialize(self, dataset_path, name_en, name_ar, airline_code):
        self.model = SentenceTransformer(self.SEARCH_BASE_MODEL)
        self.df = self.prepare_dataset(dataset_path, name_en, name_ar, airline_code)
        self.name_embeddings = self.model.encode(
            self.df["combined_cleaned"].tolist(), convert_to_tensor=True
        )

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^\w\s]', '', text)  # punctuation
        text = re.sub(r'[ًٌٍَُِّْـ]', '', text)  # Arabic diacritics
        text = text.lower()
        tokens = text.split()
        cleaned = [t for t in tokens if t not in self.EN_STOPWORDS and t not in self.AR_STOPWORDS]
        return " ".join(cleaned)

    def prepare_dataset(self, xlsx_path, name_en, name_ar, airline_code):
        df = pd.read_excel(xlsx_path)
        df[name_en] = df.apply(lambda row: row[name_ar] if pd.isna(row[name_en]) and pd.notna(row[name_ar]) else row[name_en], axis=1)
        df[name_ar] = df.apply(lambda row: row[name_en] if pd.isna(row[name_ar]) and pd.notna(row[name_en]) else row[name_ar], axis=1)
        df = df.dropna(subset=[name_en, name_ar]).reset_index(drop=True)
        df = df[df[airline_code].str.len() <= 3]
        df = df[df[airline_code].str.isalpha()]
        df["cleaned_name_en"] = df[name_en].apply(self.clean_text)
        df["cleaned_name_ar"] = df[name_ar].apply(self.clean_text)
        df["combined_cleaned"] = df["cleaned_name_en"] + " || " + df["cleaned_name_ar"]
        df["combined_name"] = df[name_en] + " || " + df[name_ar]
        return df

    def retrieve(self, query: str, top_k: int = 1, threshold: float = 0.2):
        query_cleaned = self.clean_text(query)
        query_embedding = self.model.encode(query_cleaned, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.name_embeddings)[0]
        top_results = scores.topk(k=top_k)

        for idx, score in zip(top_results.indices, top_results.values):
            row = self.df.iloc[int(idx)]
            if score >= threshold:
                return row["airline_code"]
        print("⚠️ No reliable match found (below threshold).")
        return ""
