import nltk

def download_nltk_data():
    print("Downloading NLTK data...")
    try:
        nltk.data.find('corpora/semcor')
        print("SemCor already downloaded.")
    except LookupError:
        print("Downloading SemCor...")
        nltk.download('semcor')
    
    try:
        nltk.data.find('corpora/wordnet')
        print("WordNet already downloaded.")
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')
    
    try:
        nltk.data.find('tokenizers/punkt')
        print("Punkt already downloaded.")
    except LookupError:
        print("Downloading Punkt...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/omw-1.4')
        print("OMW-1.4 already downloaded.")
    except LookupError:
        print("Downloading OMW-1.4...")
        nltk.download('omw-1.4') # Often needed for WordNet

if __name__ == "__main__":
    download_nltk_data()
