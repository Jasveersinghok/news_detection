import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Download NLTK data (if not already done)
nltk.download('stopwords')

# Load the model and vectorizer (assuming they are in the same directory)
try:
    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    raise FileNotFoundError("🚫 Model files not found in url_analysis.py. Ensure 'fake_news_model.pkl' and 'vectorizer.pkl' are in the same directory.")

# Function to validate URL
def is_valid_url(url):
    regex = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url is not None and regex.search(url)

# Function to scrape text from URL
def scrape_article_text(url):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to get title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs if p.get_text().strip()])
        text = text.strip() if text else None
        
        return title_text, text
    except Exception as e:
        return None, None

# Function to perform prediction for URL (returns results for main app to display)
def analyze_url(url):
    if not url.strip():
        return {"error": "Please enter a valid URL."}
    if not is_valid_url(url):
        return {"error": "Invalid URL format. Please enter a valid URL starting with http:// or https://."}
    
    title, text = scrape_article_text(url)
    if not title or not text:
        return {"error": "Unable to extract text from the URL. Please check the link or try pasting the text directly."}
    
    # Clean the text
    ps = PorterStemmer()
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    cleaned_text = [ps.stem(word) for word in cleaned_text if word not in stopwords.words('english')]
    cleaned_text = ' '.join(cleaned_text)
    
    transformed_news = vectorizer.transform([cleaned_text]).toarray()
    result = model.predict(transformed_news)[0].upper()
    
    # Confidence threshold
    confidence = 0
    uncertain = False
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(transformed_news)[0]
        confidence = max(prob) * 100
        if confidence < 60:
            uncertain = True
    
    return {
        "title": title,
        "result": "UNCERTAIN" if uncertain else result,
        "confidence": confidence,
        "text_preview": text[:100] + '...' if len(text) > 100 else text,
        "uncertain": uncertain
    }