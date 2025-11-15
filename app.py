
# Paste the full Streamlit code here (Step 2)
import streamlit as st
from serpapi import GoogleSearch
import joblib
import re

# Load trained model and vectorizer
# Load trained model and vectorizer (local files)
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# SerpAPI key
SERPAPI_KEY = "5f5af99f41df64da312f46618e0360a6d9e37edd1777c3c9d89da1d751518030"

# Trusted news sources
trusted_sources = [
    'bbc.com', 'reuters.com', 'ndtv.com', 'timesofindia.indiatimes.com',
    'hindustantimes.com', 'cnn.com', 'thehindu.com', 'indiatoday.in',
    'aljazeera.com', 'theguardian.com'
]

def get_search_keywords(text):
    words = text.split()
    keywords = [w for w in words if len(w) > 3]
    return " ".join(keywords[:6])

def verify_news(news_text):
    # Transform news text using the vectorizer
    news_vec = vectorizer.transform([news_text])
    
    # ML prediction using PassiveAggressiveClassifier
    ml_pred = model.predict(news_vec)[0]  # 0 = fake, 1 = real
    ml_confidence = 100 if ml_pred == 1 else 0

    # Generate smart query keywords
    query = get_search_keywords(news_text)
    params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": 10}
    search = GoogleSearch(params)
    results = search.get_dict()

    # Check for trusted news sources
    sources_found = []
    if "news_results" in results:
        for res in results["news_results"]:
            link = res.get("link", "")
            for src in trusted_sources:
                if src in link.lower():
                    sources_found.append(src)
    if not sources_found and "organic_results" in results:
        for res in results["organic_results"]:
            link = res.get("link", "")
            for src in trusted_sources:
                if src in link.lower():
                    sources_found.append(src)

    # Adjusted web reliability scoring
   # Adjusted web reliability scoring: each trusted source gives 40 points, max 100
    # Adjusted web reliability scoring: each trusted source gives 40 points, max 100
    # Adjusted web reliability scoring: each trusted source gives 40 points, max 100
    reliability_score = min(len(set(sources_found)) * 40, 100)

# Combined confidence: 30% ML, 70% web
    combined_score = round((ml_confidence * 0.3) + (reliability_score * 0.7), 2)

# Determine final label
    if combined_score >= 50:
        label = "🟢 Real News"
    elif 30 <= combined_score < 50:
        label = "🟠 Uncertain / Possibly Real"
    else:
        label = "🔴 Fake News"
        
    return {
        "ml_confidence": ml_confidence,
        "reliability_score": reliability_score,
        "combined_score": combined_score,
        "label": label,
        "trusted_sources": set(sources_found)
    }

# --- Streamlit UI ---
st.title("📰 Fake News Detection")

news_input = st.text_area("Enter a news headline or text:", height=100)

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text to check.")
    else:
        result = verify_news(news_input)
        st.success(f"✅ Final Verdict: {result['label']}")
        st.info(f"ML Model Confidence: {result['ml_confidence']}%")
        st.info(f"Web Source Reliability: {result['reliability_score']}%")
        st.info(f"Combined Confidence: {result['combined_score']}%")
        st.write(f"Trusted Sources Found: {result['trusted_sources']}")
