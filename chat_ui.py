
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="Employee Helpdesk",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Employee Helpdesk")
st.caption("Ask questions about employee policies.")

# Store chat history during the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear conversation button
if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box at the bottom of the chat
question = st.chat_input("Ask about leave, attendance, or IT support...")

if question:
    # Display and save the user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching employee policies..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=180,
                )
                response.raise_for_status()

                answer = response.json()["answer"]
                st.markdown(answer)

            except requests.exceptions.ConnectionError:
                answer = (
                    "Cannot connect to the API. "
                    "Please start the FastAPI server first."
                )
                st.error(answer)

            except requests.exceptions.Timeout:
                answer = (
                    "The request took too long. "
                    "Please check that Ollama is running and try again."
                )
                st.error(answer)

            except requests.exceptions.RequestException as error:
                answer = f"API request failed: {error}"
                st.error(answer)

            except (ValueError, KeyError):
                answer = "The API returned an unexpected response."
                st.error(answer)

    # Save the assistant's answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })