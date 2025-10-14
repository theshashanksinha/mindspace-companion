# 🧠 MindSpace: AI Mental Health Companion

<div align="center">

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Try_Now-success?style=for-the-badge)](https://thisisshashank00-mindspace-companion.hf.space/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-FF6F00?style=for-the-badge)](https://gradio.app/)

**Empathetic AI companion providing real-time emotional support with sentiment analysis and crisis detection**

[🚀 Try Live Demo](https://thisisshashank00-mindspace-companion.hf.space/)

</div>

---

## 📋 Overview

MindSpace is an AI-powered mental health companion built with Google Gemini 2.5 Flash. It analyzes emotional states in real-time, provides empathetic responses, and offers immediate crisis intervention when needed.

**Key Features:**
- 🎭 Real-time sentiment analysis (anxious, sad, positive, negative, neutral)
- 💬 Empathetic AI responses in CEFR C2 level English
- 🚨 Crisis detection with emergency resources
- 📊 Conversation analytics and mood tracking
- 🌍 Multilingual support
- 🌙 Dark/light theme toggle
- ⏱️ Timestamped conversations

---

## 🛠️ Tech Stack

```
Frontend:  Gradio 4.44.0
AI Model:  Google Gemini 2.5 Flash
NLP:       TextBlob (sentiment analysis)
Language:  Python 3.10+
Hosting:   Hugging Face Spaces
```

---

## 💻 Core Implementation

### **1. Gemini API Integration**

```
import os
from google import genai
from google.genai import types

# API Configuration
API_KEY = os.getenv("GEMINI_API_KEY", "")

# System Prompt - Core Personality
SYSTEM_PROMPT = """You are a friendly empathetic mental health companion and counsellor.
Listen carefully, detect emotional tone, respond kindly with motivational support and to turn the user to positivity and fuel with motivation.
Keep responses concise (3-5 sentences always replying in cefr c2 level english).
Automatically detect and respond in the user's language."""

# AI Response Function
def ai_reply(user_input, sentiment):
    try:
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\nUser mood:{sentiment}\nUser:{user_input}",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return response.text
    except Exception as e:
        return "⚠️ I'm having trouble connecting right now. Please try again in a moment."
```

**Key Points:**
- Uses environment variable for secure API key storage
- Injects user's emotional state into prompt for context-aware responses
- `thinking_budget=0` optimizes for faster response times
- Handles API failures gracefully

---

### **2. Sentiment Analysis Engine**

```
from textblob import TextBlob

def analyze_sentiment(text):
    # Calculate polarity score
    polarity = TextBlob(text).sentiment.polarity
    
    # Keyword-based emotion detection
    anxiety_keywords = ["anxious", "worried", "nervous", "panic", "stress"]
    sadness_keywords = ["sad", "depressed", "lonely", "empty", "hopeless"]
    
    text_lower = text.lower()
    
    # Priority-based classification
    if any(word in text_lower for word in anxiety_keywords):
        return "anxious", "😰"
    if any(word in text_lower for word in sadness_keywords):
        return "sad", "😔"
    if polarity < -0.3:
        return "negative", "😟"
    if polarity > 0.3:
        return "positive", "😊"
    
    return "neutral", "😐"
```

**How It Works:**
- Combines TextBlob's statistical polarity analysis with keyword matching
- Prioritizes specific emotions (anxiety/sadness) over general sentiment
- Returns both label and emoji for UI display

---

### **3. Crisis Detection System**

```
# Crisis keywords database
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", 
    "no reason to live", "self harm", "hurt myself", "can't go on"
]

def detect_crisis(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def crisis_response():
    return """🚨 **I'm concerned about your safety.** Please reach out for immediate help:

**Emergency Resources:**
- **India**: AASRA: 91-22-27546669, Vandrevala Foundation: 1860-2662-345
- **US**: 988 Suicide & Crisis Lifeline
- **UK**: Samaritans: 116 123
- **International**: findahelpline.com

You don't have to face this alone. Professional help is available 24/7. 💙"""
```

**Safety First:**
- Scans every message for crisis indicators
- Bypasses AI to provide immediate emergency resources
- Includes global helpline directory

---

### **4. Gradio Interface Setup**

```
import gradio as gr
from datetime import datetime

# Create Gradio app
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State({"messages": []})
    dark_mode_state = gr.State(False)
    
    with gr.Column(elem_id="main-container"):
        # Chat interface
        chat = gr.Chatbot(height=450, show_label=False, type="messages")
        msg_in = gr.Textbox(placeholder="Share what's on your mind...", show_label=False)
        
        # Buttons
        with gr.Row(elem_id="button-row"):
            send_btn = gr.Button("💬 Send Message", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            summary_btn = gr.Button("📊 Summary", variant="secondary")
            theme_btn = gr.Button("🌙/☀️ Theme", variant="secondary")
    
    # Event handlers
    send_btn.click(send, [msg_in, state, dark_mode_state], [chat, state, status, dark_mode_state])
    # ... other event bindings

# Launch
if __name__ == "__main__":
    demo.launch()
```

**Interface Features:**
- Clean, minimal UI with essential controls
- OpenAI-style message format for familiarity
- Dynamic theme switching
- Session-based state management

---

### **5. Security & Rate Limiting**

```
RATE_LIMIT = 20        # Max messages per session
MESSAGE_DELAY = 2      # Seconds between messages

def validate_input(text):
    if not text or not text.strip():
        return False, "Please enter a message."
    if len(text) > 2000:
        return False, "Message too long. Please keep it under 2000 characters."
    if len(set(text)) < 5 and len(text) > 20:
        return False, "Invalid message detected."
    return True, ""
```

**Protection Layers:**
- Volume limiting (20 messages/session)
- Temporal throttling (2-second cooldown)
- Input sanitization (2000 char limit)
- Spam pattern detection

---

## 📊 Architecture

```
User Input → Validation → Sentiment Analysis → Crisis Check
                                                      ↓
                                              [Crisis Detected?]
                                                   ↙     ↘
                                            YES: Resources    NO: Gemini API
                                                                    ↓
                                                        Formatted Response → UI
```

---

## 🚀 Deployment

**Platform:** Hugging Face Spaces  
**URL:** [https://thisisshashank00-mindspace-companion.hf.space/](https://thisisshashank00-mindspace-companion.hf.space/)

**Setup:**
1. Upload `app.py` to Hugging Face Space
2. Add `requirements.txt`:
   ```
   gradio>=4.0.0
   google-genai>=0.2.0
   textblob>=0.17.0
   ```
3. Set environment variable: `GEMINI_API_KEY`
4. Auto-deploys on commit

---

## 📝 Usage

```
# Local development
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
python app.py
```

Access at `http://localhost:7860`

---

## ⚠️ Disclaimer

**MindSpace is NOT a replacement for professional mental health services.**  
If you're in crisis, contact emergency services immediately.

---

## 📄 License

Apache License 2.0

---

## 👨‍💻 Author

**[Shashank Sinha](https://www.linkedin.com/in/theshashanksinha/)**

Data Science Enthusiast | Business Analyst
---

<div align="center">

[![Try MindSpace](https://img.shields.io/badge/🚀_TRY_MINDSPACE-Live_Demo-blueviolet?style=for-the-badge)](https://thisisshashank00-mindspace-companion.hf.space/)

*Made with ❤️ and Gemini 2.5 Flash*

</div>
