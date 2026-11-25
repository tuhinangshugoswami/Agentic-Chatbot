import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

st.set_page_config(page_title="AI Agent & PDF Chatbot", page_icon="🤖", layout="wide")


# Utilities
def generate_thread_id():
    return str(uuid.uuid4())[:8]


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    if state and state.values:
        return state.values.get("messages", [])
    return []


# Session State Setup
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# Sidebar
st.sidebar.title("🤖 Chat Control")
st.sidebar.markdown(f"**Current Thread:** `{thread_key}`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📄 Document Ingestion (RAG)")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Active: `{latest_doc.get('filename')}`\n\n"
        f"({latest_doc.get('chunks')} chunks, {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed for this thread.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already indexed.")
    else:
        try:
            with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="✅ PDF indexed successfully", state="complete", expanded=False)
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Ingestion Error: {e}")

st.sidebar.divider()
st.sidebar.subheader("📜 Past Conversations")
if not threads:
    st.sidebar.write("No conversations yet.")
else:
    for tid in threads:
        is_active = (tid == thread_key)
        btn_label = f"👉 {tid}" if is_active else f"💬 {tid}"
        if st.sidebar.button(btn_label, key=f"side-thread-{tid}", use_container_width=True):
            selected_thread = tid


# Main Chat Interface
st.title("🤖 Multi-Utility Agentic Chatbot")
st.caption(f"Active Session: `{thread_key}` | Tools: Web Search, Stock Prices, Calculator, PDF RAG")

# Display message history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question, request a calculation, or inquire about your PDF...")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Running `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Running `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 Source: `{doc_meta.get('filename')}` "
            f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
        )

# Handle Thread Switching
if selected_thread and selected_thread != thread_key:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()