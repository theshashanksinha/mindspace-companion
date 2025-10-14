import gradio as gr
from google import genai
from google.genai import types
from textblob import TextBlob
from datetime import datetime
import time
import os


# ==================== CONFIGURATION ====================
API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment variables!")

SYSTEM_PROMPT = """You are a friendly empathetic mental health companion and counsellor.
Listen carefully, detect emotional tone, respond kindly with motivational support and to turn the user to positivity and fuel with motivation.
Keep responses concise (3-5 sentences always replying in cefr c2 level english).
Automatically detect and respond in the user's language."""

# Rate limiting configuration
RATE_LIMIT = 20  # Maximum messages per session
MESSAGE_DELAY = 2  # Minimum seconds between messages

# Crisis detection keywords
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "no reason to live", "self harm", "hurt myself", "can't go on"
]


# ==================== HELPER FUNCTIONS ====================

def validate_input(text):
    """
    Validate user input for length and content quality.

    Args:
        text (str): User input message

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    if not text or not text.strip():
        return False, "Please enter a message."

    if len(text) > 2000:
        return False, "Message too long. Please keep it under 2000 characters."

    # Basic spam detection
    if len(set(text)) < 5 and len(text) > 20:
        return False, "Invalid message detected."

    return True, ""


def detect_crisis(text):
    """
    Detect if user message contains crisis-related keywords.

    Args:
        text (str): User input message

    Returns:
        bool: True if crisis keywords detected, False otherwise
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)


def crisis_response():
    """
    Return formatted crisis resource information.

    Returns:
        str: Crisis helpline information with markdown formatting
    """
    return """🚨 **I'm concerned about your safety.** Please reach out for immediate help:

**Emergency Resources:**
- **India**: AASRA: 91-22-27546669, Vandrevala Foundation: 1860-2662-345
- **US**: 988 Suicide & Crisis Lifeline
- **UK**: Samaritans: 116 123
- **International**: findahelpline.com

You don't have to face this alone. Professional help is available 24/7. 💙"""


def analyze_sentiment(text):
    """
    Analyze sentiment of user message using TextBlob and keyword detection.

    Args:
        text (str): User input message

    Returns:
        tuple: (sentiment_label: str, emoji: str)
    """
    # Calculate polarity score
    polarity = TextBlob(text).sentiment.polarity

    # Define keyword lists for specific emotions
    anxiety_keywords = ["anxious", "worried", "nervous", "panic", "stress"]
    sadness_keywords = ["sad", "depressed", "lonely", "empty", "hopeless"]

    text_lower = text.lower()

    # Check for anxiety indicators
    if any(word in text_lower for word in anxiety_keywords):
        return "anxious", "😰"

    # Check for sadness indicators
    if any(word in text_lower for word in sadness_keywords):
        return "sad", "😔"

    # Use polarity for general sentiment
    if polarity < -0.3:
        return "negative", "😟"
    if polarity > 0.3:
        return "positive", "😊"

    return "neutral", "😐"


def ai_reply(user_input, sentiment):
    """
    Generate AI response using Gemini API.

    Args:
        user_input (str): User's message
        sentiment (str): Detected sentiment label

    Returns:
        str: AI-generated response or error message
    """
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=API_KEY)

        # Generate content with sentiment context
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\nUser mood:{sentiment}\nUser:{user_input}",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking mode
            )
        )

        return response.text

    except Exception as e:
        return "⚠️ I'm having trouble connecting right now. Please try again in a moment."


def format_messages_for_chatbot(msgs):
    """
    Convert internal message format to Gradio Chatbot display format.
    Adds timestamps, sentiment indicators, and proper formatting.

    Args:
        msgs (list): List of message dictionaries with role, content, timestamp, sentiment

    Returns:
        list: List of message dictionaries formatted for Gradio Chatbot (OpenAI style)
    """
    chatbot_msgs = []

    for msg in msgs:
        timestamp = msg.get("timestamp", "")
        time_str = f"<span style='font-size:0.75em; color:#888;'>{timestamp}</span><br>" if timestamp else ""

        if msg["role"] == "user":
            # Format user message with timestamp and sentiment
            sentiment = msg.get("sentiment", ("", ""))
            sentiment_str = f"<span style='font-size:0.9em'>{sentiment[1]} {sentiment[0]}</span><br>" if sentiment[0] else ""
            content = f"{time_str}{sentiment_str}{msg['content']}"
            chatbot_msgs.append({"role": "user", "content": content})

        else:
            # Format AI companion message with brain icon and name
            content = f"🧠 **Companion**<br>{time_str}{msg['content']}"
            chatbot_msgs.append({"role": "assistant", "content": content})

    return chatbot_msgs


def generate_summary(state):
    """
    Generate conversation summary with mood statistics.

    Args:
        state (dict): Application state containing messages

    Returns:
        str: Summary text in markdown format
    """
    msgs = state["messages"]

    # Check if enough messages exist for summary
    if not msgs or len(msgs) < 4:
        return "Not enough conversation data for summary. Chat more to generate insights!"

    # Extract user messages
    user_msgs = [m for m in msgs if m["role"] == "user"]

    # Count sentiment distribution
    sentiment_counts = {}
    for msg in user_msgs:
        if "sentiment" in msg:
            mood = msg["sentiment"][0]
            sentiment_counts[mood] = sentiment_counts.get(mood, 0) + 1

    # Determine dominant mood
    dominant_mood = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "neutral"

    # Build summary markdown
    summary = f"""### 📊 Conversation Summary

**Total Messages:** {len(msgs)} ({len(user_msgs)} from you)

**Emotional Journey:**
"""

    # Add sentiment breakdown
    for mood, count in sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(user_msgs)) * 100
        summary += f"\n- {mood.capitalize()}: {count} messages ({percentage:.1f}%)"

    summary += f"\n\n**Overall Mood:** {dominant_mood.capitalize()}"
    summary += f"\n\n💡 *Every conversation is a step toward better understanding yourself. Keep going!* 💙"

    return summary


# ==================== ACTION HANDLERS ====================

def send(msg, state, dark_mode):
    """
    Handle sending user message and generating AI response.
    Includes validation, rate limiting, crisis detection, and typing indicator.

    Args:
        msg (str): User's message
        state (dict): Application state
        dark_mode (bool): Current theme mode

    Yields:
        tuple: (chatbot_messages, state, status_message, dark_mode)
    """
    # Check rate limit
    user_message_count = len([m for m in state["messages"] if m["role"] == "user"])
    if user_message_count >= RATE_LIMIT:
        yield (
            format_messages_for_chatbot(state["messages"]),
            state,
            "⚠️ Rate limit reached. Please start a new session.",
            dark_mode
        )
        return

    # Validate input
    is_valid, error_msg = validate_input(msg)
    if not is_valid:
        yield (
            format_messages_for_chatbot(state["messages"]),
            state,
            f"⚠️ {error_msg}",
            dark_mode
        )
        return

    # Check message delay (rate limiting)
    if state["messages"]:
        last_msg = state["messages"][-1]
        if "timestamp" in last_msg:
            try:
                # Parse timestamps and calculate difference
                last_time = datetime.strptime(last_msg["timestamp"], "%I:%M:%S %p")
                current_time = datetime.now()

                last_seconds = last_time.hour * 3600 + last_time.minute * 60 + last_time.second
                current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
                time_diff = current_seconds - last_seconds

                if 0 < time_diff < MESSAGE_DELAY:
                    yield (
                        format_messages_for_chatbot(state["messages"]),
                        state,
                        "⚠️ Please wait a moment before sending another message.",
                        dark_mode
                    )
                    return
            except:
                pass

    # Get timestamp in 12-hour format with AM/PM
    timestamp = datetime.now().strftime("%I:%M:%S %p")

    # Analyze sentiment
    sentiment = analyze_sentiment(msg)

    # Add user message to state
    state["messages"].append({
        "role": "user",
        "content": msg,
        "sentiment": sentiment,
        "timestamp": timestamp
    })

    # Check for crisis keywords
    if detect_crisis(msg):
        crisis_msg = crisis_response()
        state["messages"].append({
            "role": "assistant",
            "content": crisis_msg,
            "timestamp": datetime.now().strftime("%I:%M:%S %p")
        })
        yield (
            format_messages_for_chatbot(state["messages"]),
            state,
            "",
            dark_mode
        )
        return

    # Show typing indicator
    yield (
        format_messages_for_chatbot(state["messages"]),
        state,
        "💭 Typing...",
        dark_mode
    )

    # Generate AI response
    ai_response = ai_reply(msg, sentiment[0])

    # Add AI response to state
    state["messages"].append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().strftime("%I:%M:%S %p")
    })

    # Return final state
    yield (
        format_messages_for_chatbot(state["messages"]),
        state,
        "",
        dark_mode
    )


def clear(state, dark_mode):
    """
    Clear all chat messages and reset state.

    Args:
        state (dict): Application state
        dark_mode (bool): Current theme mode

    Returns:
        tuple: Reset values for chat, state, status, summary, and visibility
    """
    state["messages"] = []
    return (
        format_messages_for_chatbot(state["messages"]),
        state,
        "",
        "",
        gr.update(visible=False),  # Hide summary section
        dark_mode
    )


def toggle_dark_mode(current_mode):
    """
    Toggle between light and dark theme.

    Args:
        current_mode (bool): Current dark mode state

    Returns:
        bool: Inverted mode state
    """
    return not current_mode


def show_summary_handler(state):
    """
    Handler to show summary section with generated content.

    Args:
        state (dict): Application state

    Returns:
        tuple: (summary_text, visibility_update)
    """
    summary_text = generate_summary(state)
    return summary_text, gr.update(visible=True)


def close_summary_handler():
    """
    Handler to close/hide summary section.

    Returns:
        tuple: (empty_text, visibility_update)
    """
    return "", gr.update(visible=False)


# ==================== STYLING ====================

def get_css(dark_mode):
    """Simple CSS."""
    bg_gradient = (
        "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" if dark_mode 
        else "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    )
    text_color = "rgba(255,255,255,0.95)"
    chat_bg = "#1e1e1e" if dark_mode else "white"
    chat_text = "#e0e0e0" if dark_mode else "#333"
    input_border = "#444" if dark_mode else "#e0e0e0"
    input_bg = "#2a2a2a" if dark_mode else "white"
    btn_secondary_bg = "#333" if dark_mode else "white"
    btn_secondary_hover = "#444" if dark_mode else "#f8f9ff"
    summary_bg = "#2a2a2a" if dark_mode else "#f8f9ff"
    summary_border = "#444" if dark_mode else "#e0e7ff"
    close_btn_bg = "#444" if dark_mode else "#e0e7ff"
    close_btn_hover = "#555" if dark_mode else "#d0d7ef"
    
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {{
    font-family: 'Inter', sans-serif;
}}

#main-container {{
    background: {bg_gradient};
    padding: 1.2rem 1.5rem 0.8rem 1.5rem;
}}

#header {{
    text-align: center;
    margin-bottom: 0.8rem;
}}

#title {{
    color: {text_color};
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}}

#subtitle {{
    color: {text_color};
    font-size: 0.85rem;
    font-weight: 400;
    font-style: italic;
    opacity: 0.9;
}}

#chat-container {{
    background: {chat_bg};
    color: {chat_text};
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    max-width: 1400px;
    margin: 0 auto;
}}

.input-box textarea {{
    border-radius: 12px !important;
    border: 2px solid {input_border} !important;
    padding: 12px !important;
    font-size: 1rem !important;
    min-height: 55px !important;
    background: {input_bg} !important;
    color: {chat_text} !important;
}}

.input-box textarea:focus {{
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.1) !important;
}}

.btn {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 11px 28px !important;
    transition: all 0.3s ease !important;
    font-size: 0.95rem !important;
}}

.btn-primary {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
}}

.btn-primary:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(102,126,234,0.4) !important;
}}

.btn-secondary {{
    background: {btn_secondary_bg} !important;
    border: 2px solid #667eea !important;
    color: {chat_text} !important;
}}

.btn-secondary:hover {{
    background: {btn_secondary_hover} !important;
    transform: translateY(-2px) !important;
}}

#status-text {{
    text-align: center;
    color: {text_color};
    font-style: italic;
    margin-top: 0.4rem;
    font-size: 0.85rem;
    min-height: 20px;
}}

.chatbot {{
    border-radius: 12px !important;
}}

#button-row {{
    display: flex;
    gap: 10px;
    margin-top: 1rem;
    flex-wrap: wrap;
    justify-content: center;
}}

#summary-section {{
    background: {summary_bg};
    border-radius: 12px;
    padding: 1.2rem 2.5rem 1.2rem 1.2rem;
    margin-top: 0.8rem;
    border: 2px solid {summary_border};
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    position: relative;
    max-width: 1400px;
    margin-left: auto;
    margin-right: auto;
}}

#close-summary-btn {{
    position: absolute;
    top: 0.8rem;
    right: 0.8rem;
    background: {close_btn_bg} !important;
    border: none !important;
    padding: 6px 12px !important;
    font-size: 0.9rem !important;
    border-radius: 6px !important;
    cursor: pointer;
    transition: all 0.2s ease !important;
    font-weight: 600 !important;
}}

#close-summary-btn:hover {{
    background: {close_btn_hover} !important;
}}

#footer {{
    text-align: center;
    color: {text_color};
    font-size: 0.85rem;
    margin-top: 0.6rem;
    padding: 0.3rem;
    line-height: 1.5;
}}

#footer a {{
    color: {text_color};
    text-decoration: none;
    font-weight: 600;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}}

#footer a:hover {{
    color: #ffd700;
    text-shadow: 0 0 10px rgba(255,215,0,0.5);
}}

.linkedin-icon {{
    width: 13px;
    height: 13px;
    display: inline-block;
    vertical-align: middle;
}}

/* Mobile Responsiveness */
@media (max-width: 768px) {{
    #main-container {{
        padding: 0.8rem 1rem;
    }}
    
    #chat-container {{
        padding: 1rem;
    }}
    
    #title {{
        font-size: 1.8rem;
    }}
    
    #subtitle {{
        font-size: 0.8rem;
    }}
    
    .btn {{
        padding: 9px 18px !important;
        font-size: 0.85rem !important;
    }}
    
    #button-row {{
        gap: 8px;
    }}
}}

@media (max-width: 480px) {{
    #title {{
        font-size: 1.5rem;
    }}
    
    .btn {{
        padding: 8px 14px !important;
        font-size: 0.8rem !important;
    }}
}}
"""


# ==================== GRADIO INTERFACE ====================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Initialize application state
    state = gr.State({"messages": []})
    dark_mode_state = gr.State(False)

    # Dynamic CSS component
    css_output = gr.HTML()

    with gr.Column(elem_id="main-container"):
        # Header section
        gr.HTML("""
            <div id='header'>
                <div id='title'>🧠 MindSpace</div>
                <div id='subtitle'>Made with love ❤️ by
                    <a href='https://www.linkedin.com/in/theshashanksinha/' target='_blank' style='color: rgba(255,255,255,0.95); text-decoration: none; font-weight: 600;'>
                        <svg style='width: 13px; height: 13px; display: inline-block; vertical-align: middle; margin: 0 2px;' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'>
                            <path d='M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z'/>
                        </svg>Shashank Sinha
                    </a>
                </div>
            </div>
        """)

        # Main chat container
        with gr.Column(elem_id="chat-container"):
            # Chatbot display
            chat = gr.Chatbot(height=450, show_label=False, type="messages")

            # Message input
            msg_in = gr.Textbox(
                placeholder="Share what's on your mind...",
                show_label=False,
                lines=2,
                elem_classes="input-box"
            )

            # Action buttons
            with gr.Row(elem_id="button-row"):
                send_btn = gr.Button("💬 Send Message", variant="primary", elem_classes="btn btn-primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", elem_classes="btn btn-secondary", scale=1)
                summary_btn = gr.Button("📊 Summary", variant="secondary", elem_classes="btn btn-secondary", scale=1)
                theme_btn = gr.Button("🌙/☀️ Theme", variant="secondary", elem_classes="btn btn-secondary", scale=1)

            # Status message
            status = gr.Markdown("", elem_id="status-text")

        # Summary section (hidden by default) - outside chat container
        with gr.Column(visible=False, elem_id="summary-section") as summary_section:
            summary_output = gr.Markdown("")
            close_btn = gr.Button("✕ Close", elem_id="close-summary-btn", size="sm")

    # ==================== EVENT HANDLERS ====================

    def update_theme(dark_mode):
        """Update CSS based on theme selection"""
        css = f"<style>{get_css(dark_mode)}</style>"
        return css, dark_mode

    # Load theme on startup
    demo.load(
        fn=lambda dm: update_theme(dm),
        inputs=[dark_mode_state],
        outputs=[css_output, dark_mode_state]
    )

    # Send message on button click
    send_btn.click(
        fn=send,
        inputs=[msg_in, state, dark_mode_state],
        outputs=[chat, state, status, dark_mode_state]
    ).then(
        fn=lambda: "",  # Clear input after sending
        inputs=None,
        outputs=msg_in
    )

    # Send message on Enter key
    msg_in.submit(
        fn=send,
        inputs=[msg_in, state, dark_mode_state],
        outputs=[chat, state, status, dark_mode_state]
    ).then(
        fn=lambda: "",  # Clear input after sending
        inputs=None,
        outputs=msg_in
    )

    # Clear chat
    clear_btn.click(
        fn=clear,
        inputs=[state, dark_mode_state],
        outputs=[chat, state, status, summary_output, summary_section, dark_mode_state]
    )

    # Show summary - Fixed to return proper outputs
    summary_btn.click(
        fn=show_summary_handler,
        inputs=[state],
        outputs=[summary_output, summary_section]
    )

    # Close summary - Fixed to return proper outputs
    close_btn.click(
        fn=close_summary_handler,
        inputs=None,
        outputs=[summary_output, summary_section]
    )

    # Toggle theme
    theme_btn.click(
        fn=toggle_dark_mode,
        inputs=[dark_mode_state],
        outputs=[dark_mode_state]
    ).then(
        fn=update_theme,
        inputs=[dark_mode_state],
        outputs=[css_output, dark_mode_state]
    )

# ==================== LAUNCH APPLICATION ====================
# demo.launch()
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        height=900,
        width="100%",
        prevent_thread_lock=False,
        inline=False
    )