from flask import Flask, render_template, request, session
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.spatial.distance import cosine
from nltk.stem.snowball import PorterStemmer
import numpy as np
import re
import warnings
import string

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Needed for session support
warnings.filterwarnings("ignore", category=RuntimeWarning)
mood = None
reason = ""
plans = ""

# Mood dataset
moods = {
    "happy": [
    "I am feeling good today", "I am well thank you", "I'm good", "I'm feeling happy",
    "I'm so happy", "I'm so excited", "Today I am feeling happy", "I'm feeling joyful",
    "Things are going really well", "I'm feeling optimistic today", "I'm in a great mood!",
    "Life is good right now", "I'm quite cheerful", "I feel content and peaceful",
    "I'm full of energy today", "I'm feeling positive", "Everything feels right today",
    "I'm on top of the world", "Nothing can ruin my day", "I'm so grateful for today",
    "I'm smiling for no reason", "I'm feeling lucky", "Everything is falling into place",
    "Today feels special", "I feel refreshed and inspired", "I'm excited for what's next",
    "I'm buzzing with energy", "I feel like dancing", "I'm laughing a lot today",
    "This is one of the best days I've had in a while"
    ],

    "sad": [
    "I'm feeling down", "Today hasn't been great", "I'm not doing so well", "I'm a bit sad",
    "I feel really low today", "I've been feeling blue", "I'm upset", "I'm having a hard day",
    "I just feel empty", "Things have been rough", "I feel like crying", "I'm emotionally drained",
    "I feel hopeless", "I'm feeling a little lost", "My mood is really low", "I'm feeling lonely",
    "I'm not in the best headspace right now", "I'm feeling quite sad today", "I'm feeling terrible",
    "I don't have any motivation", "I just want to be alone", "I'm overwhelmed by everything",
    "I'm struggling to cope", "Nothing feels right", "I'm mentally exhausted",
    "I can't seem to shake this feeling", "I'm feeling heartbroken", "Today just feels heavy",
    "I feel like giving up", "I'm not okay", "I'm barely holding it together"
    ],

    "angry": [
    "I'm really annoyed", "I'm frustrated today", "Things are making me angry", "I'm feeling irritated",
    "I'm mad right now", "I'm seriously pissed off", "I can't deal with this!", "Everything is getting on my nerves",
    "I'm so fed up", "I'm losing my patience", "People keep pushing my buttons", "I'm just in a bad mood",
    "I'm raging inside", "I feel so tense and angry", "This day has been infuriating", "I'm seething right now",
    "My temper is running thin", "Why is everything so difficult?", "I'm boiling over with anger",
    "I'm furious right now", "I feel like screaming", "I'm not in the mood for anything",
    "Everyone's testing me today", "I can't take it anymore", "I'm about to explode",
    "I'm feeling really short-tempered", "I'm at my limit", "I'm furious and exhausted",
    "I want to lash out", "I'm done being nice today"
    ]
}

# Preprocessing
def preprocess(user_input):
    p_stemmer = PorterStemmer()
    tokens = re.findall(r'\b\w+\b', user_input.lower())
    stemmed_tokens = [p_stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# Prepare training data
processed_moods = {intent: [preprocess(p) for p in phrases] for intent, phrases in moods.items()}
all_phrases = [phrase for phrases in processed_moods.values() for phrase in phrases]
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
count_matrix = count_vectorizer.fit_transform(all_phrases)
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
tfidf_vectors = tfidf_matrix.toarray()

# Match user mood
def mood_matching(user_input):
    input_vector = tfidf_transformer.transform(count_vectorizer.transform([preprocess(user_input)])).toarray()[0]
    similarities = [1 - cosine(input_vector, vec) for vec in tfidf_vectors]
    if np.isnan(max(similarities)) or max(similarities) < 0.4:
        return None
    best_match_index = np.argmax(similarities)

    intent_lengths = [len(phrases) for phrases in processed_moods.values()]
    cumulative_lengths = np.cumsum(intent_lengths)
    for i, length in enumerate(cumulative_lengths):
        if best_match_index < length:
            return list(processed_moods.keys())[i]

def process_input(reason):
    reason = reason.strip().rstrip(string.punctuation)
    tokens = reason.lower().split()
    converted_tokens = []
    for token in tokens:
        if token == "i":
            converted_tokens.append("you")
        elif token == "we":
            converted_tokens.append("you")
        elif token == "my":
            converted_tokens.append("your")
        elif token == "i'm":
            converted_tokens.append("you're")
        elif token == "am":
            converted_tokens.append("are")
        else:
            converted_tokens.append(token)
    return " ".join(converted_tokens)


# Web route
@app.route("/", methods=["GET", "POST"])
def chat():
    global mood, reason, plans
    if request.method == "GET":
        session.clear()
        session["history"] = []
        session["step"] = 1
        session["name"] = ""
        bot_msg = "Hi! I am EmoBot, your emotional support chatbot - what is your name?"
        session["history"].append(("bot", bot_msg))


    if request.method == "POST":
        name_input = request.form.get("name", "").strip()
        message_input = request.form.get("message", "").strip()

        if session["step"] == 1 and name_input:
            session["name"] = name_input
            bot_msg = f"Nice to meet you {name_input.capitalize()}, how are you feeling today?"
            session["history"].append(("user", name_input))
            session["history"].append(("bot", bot_msg))
            session["step"] = 2

        elif session["step"] == 2 and message_input:
            session["history"].append(("user", message_input))
            mood = mood_matching(message_input)
            if mood != None:
                if mood == "happy":
                    bot_msg = f"I'm so glad to hear you're feeling happy today, {session['name'].capitalize()}!"
                elif mood == "sad":
                    bot_msg = f"I'm sorry to hear you're feeling sad, {session['name'].capitalize()}. Maybe I can help."
                elif mood == "angry":
                    bot_msg = f"It's okay to feel angry, {session['name'].capitalize()}. Let's talk about it."
                session["history"].append(("bot", bot_msg))

                if mood == "happy":
                    bot_msg = "Would you like to share what's brought on your good mood?"
                elif mood == "sad":
                    bot_msg = "Sad days happen to all of us, would you like to share what has been on your mind lately?"
                elif mood == "angry":
                    bot_msg = "If you don't mind sharing, what's been triggering these feelings today?"

                session["history"].append(("bot", bot_msg))
                session["step"] = 3
            else:
                bot_msg = f"I'm not quite sure I got how you're feeling, would you be able to describe your emotions to me in a different way?"
                session["history"].append(("bot", bot_msg))

        elif session["step"] == 3 and message_input:
            session["history"].append(("user", message_input))
            reason = message_input.lower()
            reason = process_input(reason)


            if mood == "happy":
                bot_msg = "Wow! That's amazing!"
            elif mood == "sad":
                bot_msg = "Ah, I can imagine how that must make you feel"
            elif mood == "angry":
                bot_msg = "That sounds incredibly frustrating, I can see why it's caused your mood to worsen."

            session["history"].append(("bot", bot_msg))

            if mood == "happy":
                bot_msg = "It is important to maintain a happy mood for as long as you can, have you got any plans for the rest of today that will keep you in good spirits?"
            elif mood == "sad":
                bot_msg = "Why don't we try to brighten your mood? Have you got anything planned today that will lift your spirits?"
            elif mood == "angry":
                bot_msg = "Maybe it would be good to take your mind off of things by doing something else, have you got anything planned for the rest of today?"

            session["history"].append(("bot", bot_msg))
            session["step"] = 4

        elif session["step"] == 4 and message_input:
            plan = message_input.lower()
            plan = process_input(plan)
            session["history"].append(("user", message_input))
            if mood == "happy":
                bot_msg = f"Sounds great! That combined with how {reason} is sure to keep you in a perfect mood!"
            elif mood == "sad":
                bot_msg = f"Sometimes when you are upset, talking through it with someone takes the burden off of you. If you feel comfortable disclosing how {reason}, then I recommend speaking to a friend about your worries."
            elif mood == "angry":
                bot_msg = f"Everyone feels anger and frustration sometimes, but it shows that we care about something."
            session["history"].append(("bot", bot_msg))

            if mood == "happy":
                bot_msg = f"Well I think that you have a great day ahead of you {session['name'].capitalize()}, feel free to update me about what you get up to tomorrow!"
            elif mood == "sad":
                bot_msg = f"I really hope that your day improves, {session['name'].capitalize()}. Please let me know how the rest of it goes tomorrow!"
            elif mood == "angry":
                bot_msg = f"I really hope that your mood lightens when {plan}, {session['name'].capitalize()}. I look forward to hearing about it in tomorrow's check in!"

            session["history"].append(("bot", bot_msg))
            session["step"] = 5

    return render_template("index.html", history=session["history"], step=session["step"], name=session["name"])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
