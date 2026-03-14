"""
Streamlit ChatGPT Clone with Llama 3.3 70B on Groq
Run: streamlit run app.py

Make sure you have:
- A folder named `.streamlit` in the same directory as app.py
- Inside it, a file `secrets.toml` with:
    GROQ_API_KEY = "your-actual-api-key"
- Or set the environment variable GROQ_API_KEY
"""

import os
import streamlit as st

# ----------------------------
# Safe import of Groq
# ----------------------------
try:
    from groq import Groq, RateLimitError, APIError
except ImportError:
    st.error(
        "❌ The `groq` library is not installed.\n\n"
        "Please install it by running:\n"
        "```\npip install groq\n```"
    )
    st.stop()

# ----------------------------
# 1. Page Configuration
# ----------------------------
st.set_page_config(page_title="Llama Chat", page_icon="🦙", layout="wide")
st.title("🦙 Llama 3.3 ChatGPT Clone")
st.caption("Powered by Groq's Llama 3.3 70B model – streaming responses")

# ----------------------------
# 2. API Key Setup
# ----------------------------
# Try environment variable first, then Streamlit secrets
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "🚨 GROQ_API_KEY not found.\n\n"
        "Please set it in a Streamlit secret:\n"
        "1. Create a folder named `.streamlit` in the same directory as `app.py`.\n"
        "2. Inside it, create a file `secrets.toml` with the line:\n"
        "   GROQ_API_KEY = \"your-actual-api-key\"\n\n"
        "Or set the environment variable GROQ_API_KEY."
    )
    st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)

# ----------------------------
# 3. Session State Initialisation
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# ----------------------------
# 4. Display Chat History
# ----------------------------
for message in st.session_state.messages:
    if message["role"] != "system":  # don't show system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ----------------------------
# 5. Chat Input & Response Handling
# ----------------------------
if prompt := st.chat_input("Type your message..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for API (system message is included automatically)
    messages_for_api = st.session_state.messages

    # Call Groq with streaming
    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_api,
            temperature=0.7,
            max_tokens=1024,
            stream=True,
        )

        # Display assistant response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")  # cursor effect

            # Final response without cursor
            response_placeholder.markdown(full_response)

        # Append assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except RateLimitError:
        st.error("⚠️ Rate limit exceeded. Please wait a moment and try again.")
    except APIError as e:
        st.error(f"⚠️ API error: {e}")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")

    # No need for st.rerun() – Streamlit automatically re-runs after each interaction