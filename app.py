"""
Streamlit ChatGPT Clone with Llama 3.3 70B on Groq
Run: streamlit run app.py
"""

import os
import streamlit as st
from groq import Groq

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
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if not api_key:
    st.error("🚨 GROQ_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
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

    # Prepare messages for API (exclude any non-dict if present)
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

    # Force a rerun to update UI (optional, but streamlit handles it automatically)
    st.rerun()