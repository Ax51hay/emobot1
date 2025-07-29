import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine
from nltk.stem.snowball import PorterStemmer
from nltk.corpus import words
import numpy as np
import warnings
import re

# Download words for name filtering
nltk.download('words')

# Surpresses warnings when calculating cosine similarities
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

english_words = set(words.words())

# Used to determine name from a users input
def is_english_word(word):
    return word.lower() in english_words

# Intent Training Data
moods = {
    "happy": [
        "I am feeling good today",
        "I am well thank you",
        "I'm good",
        "I'm feeling happy",
        "I'm so happy",
        "I'm so excited",
        "Today I am feeling happy",
        "I'm feeling joyful",
        "Things are going really well",
        "I'm feeling optimistic today",
        "I'm in a great mood!",
        "Life is good right now",
        "I'm quite cheerful",
        "I feel content and peaceful",
        "I'm full of energy today",
        "I'm feeling positive",
        "Everything feels right today"
    ],

    "sad": [
        "I'm feeling down",
        "Today hasn't been great",
        "I'm not doing so well",
        "I'm a bit sad",
        "I feel really low today",
        "I've been feeling blue",
        "I'm upset",
        "I'm having a hard day",
        "I just feel empty",
        "Things have been rough",
        "I feel like crying",
        "I'm emotionally drained",
        "I feel hopeless",
        "I'm feeling a little lost",
        "My mood is really low",
        "I'm feeling lonely",
        "I'm not in the best headspace right now"
    ],

    "angry": [
        "I'm really annoyed",
        "I'm frustrated today",
        "Things are making me angry",
        "I'm feeling irritated",
        "I'm mad right now",
        "I'm seriously pissed off",
        "I can't deal with this!",
        "Everything is getting on my nerves",
        "I'm so fed up",
        "I'm losing my patience",
        "People keep pushing my buttons",
        "I'm just in a bad mood",
        "I'm raging inside",
        "I feel so tense and angry",
        "This day has been infuriating",
        "I'm seething right now",
        "My temper is running thin"
    ]
}

name = ""
matched_mood = ""

# Preprocess input via tokenisation and stemming
def preprocess(user_input):
    p_stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(user_input.lower())
    stemmed_tokens = [p_stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# Process intents and dataset
processed_moods = {intent: [preprocess(phrase) for phrase in phrases] for intent, phrases in moods.items()}

def handle_confirmation(user_input):
    #Won't exit until correct input is found
    while True:
        if "yes" in user_input.lower() and "no" in user_input.lower():
            #Appropriate error handling
            print("Bot: I have detected both a yes and no as part of your response which would be contradictory, would you be able to confirm by typing 'Yes' OR 'No'? ")
            user_input = input("You: ")
        elif "yes" in user_input.lower():
            return True
        elif "no" in user_input.lower():
            return False
        else:
            #Re prompt the user with useful hint
            print("Bot: I'm sorry, I didn't understand that. Could you please confirm by typing 'Yes' or 'No'? ")
            user_input = input("You: ")

def get_name():
    global name
    while True:
        if name == "":
            print("Bot: May I ask what your name is?")
            user_input = input("You: ")
            words = user_input.lower().split()
            for word in words:
                #Finds the name by filtering out non english words
                if not is_english_word(word):
                    name = word.capitalize()
                    #Confirmation message in case name is wrongly selected
                    print(f"Bot: I have understood your name to be {name}, is that correct?")
                    user_input = input("You: ")
                    if handle_confirmation(user_input):
                        return f"Bot: Nice to meet you, {name}! How are you feeling today?"
                    else:
                        print("Bot: My apologies! Let's try again.")
                        name = ""

# Intent matching
def mood_matching(user_input):
    #Groups all phrases together
    all_phrases = [phrase for phrases in processed_moods.values() for phrase in phrases]
    
    count_vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    
    #Creates a count matrix/BOW representation of phrases
    count_matrix = count_vectorizer.fit_transform(all_phrases)
    #Converts count matrix to TF/IDF matrix of phrases
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
    
    #Preprocesses input and converts to fit same vector space as phrases
    input_count_vector = count_vectorizer.transform([preprocess(user_input)])
    #Converts to TF-IDF ready for comparisons
    input_tfidf_vector = tfidf_transformer.transform(input_count_vector)
    
    input_vector = input_tfidf_vector.toarray()[0]
    tfidf_vectors = tfidf_matrix.toarray()
    
    #Calculates cosine similarity inversely
    similarities = [1 - cosine(input_vector, phrase_vector) for phrase_vector in tfidf_vectors]

    #Checks for NaNs (Error Handling), ususally happens when input token isnt in vocab
    if np.isnan(max(similarities)):
        return None
    if max(similarities) < 0.4: #Threshold to catch similar phrases 
        return None
    best_match_index = np.argmax(similarities) #Finds index of phrase that matches the most
    
    intent_lengths = [len(phrases) for phrases in processed_moods.values()] #Matches phrase to intent
    cumulative_lengths = np.cumsum(intent_lengths)
    for i, length in enumerate(cumulative_lengths):
        if best_match_index < length:
            return list(processed_moods.keys())[i]

# Respond to user using intent matching
def emotion_q():
    global name
    global matched_mood
    user_input = input("You: ")
    matched_mood = mood_matching(user_input)
    if matched_mood is None:
        print("Bot: I'm sorry, I didn't quite understand that. If you are unsure of what to ask me, try asking for help!")
    elif matched_mood == "happy":
        print(f"Bot: I am so glad to hear that you are feeling happy today, {name}!")
    elif matched_mood == "sad":
        print(f"Bot: I'm sorry to hear that you're feeling upset {name}, maybe I can help.")
    elif matched_mood == "angry":
        print(f"Bot: It's ok to feel angry sometimes {name}, don't worry we will work through your emotions together.")

def activity_q():
    global matched_mood
    if matched_mood == "happy":
        print("Bot: Would you like to share what's brought on your good mood?")
    elif matched_mood == "sad":
        print("Bot: Sad days happen to all of us, would you like to share what has been on your mind lately?")
    elif matched_mood == "angry":
        print("Bot: If you don't mind sharing, what's been triggering these feelings today?")
    input("You: ")

def suggestion_q():
    global name
    global matched_mood
    if matched_mood == "happy":
        print("Bot: Wow! That's amazing!")
    elif matched_mood == "sad":
        print("Bot: Ah, I can imagine how that must make you feel")
    elif matched_mood == "angry":
        print("Bot: That sounds incredibly frustrating, I can see why it's caused your mood to worsen.")

# Main I/O Loop
def main():
    global name
    print("------------------------------------------------------------CHATBOT STARTED------------------------------------------------------------")
    print(get_name())
    emotion_q()
    activity_q()
    suggestion_q()
    print("------------------------------------------------------------CHATBOT ENDED--------------------------------------------------------------")

if __name__ == "__main__":
    main()