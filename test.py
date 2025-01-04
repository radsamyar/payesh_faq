import streamlit as st
import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="چت بات FAQ General", page_icon=":speech_balloon:", layout="centered")
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        import base64
        return base64.b64encode(image_file.read()).decode()

# Encode the image
logo_base64 = get_base64_encoded_image("payeshgaran_logo.jfif")

# Use HTML to center-align the logo and title
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{logo_base64}" alt="Logo" width="150">
        <h1 style="margin-top: 10px;">چت بات FAQ General</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.write()

# مقداردهی اولیه session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []  # لیست برای ذخیره تمام پیام‌های گفتگو
if 'last_retrieved_answer' not in st.session_state:
    st.session_state['last_retrieved_answer'] = None # ذخیره آخرین پاسخ بازیابی شده

SYSTEM_PROMPT = (
    "پاسخ‌های خود را بر اساس اطلاعات بازیابی‌شده از اسناد ارائه شده بنویسید. اگر اطلاعات کافی نیست، صریحاً اعلام کنید و سپس با دانش خود پاسخ دهید."
)



@st.cache_resource
def load_index_and_docs():
    index = faiss.read_index("faiss_questions (4).index")
    documents = pd.read_csv("questions (4).csv")
    return index, documents

index, documents = load_index_and_docs()

@st.cache_resource
def load_model():
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    return model

model = load_model()

@st.cache_resource
def load_llm():
    llm = ChatGroq(groq_api_key="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", model_name="Gemma2-9b-It")
    return llm

llm = load_llm()

def get_question_embeddings(question):
    sentences = [question]
    embeddings = model.encode(sentences, batch_size=12, max_length=512)['dense_vecs']
    return embeddings[0]

def search_questions(query, top_k=3):
    query_embedding = get_question_embeddings(query).astype('float16').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    if indices[0][0] == -1:
        return pd.DataFrame()
    results = documents.iloc[indices[0]]
    return results
def count_tokens(messages):
    tokens = 0
    for msg in messages:
        tokens += len(tokenizer.encode(msg.content))
    return tokens

def chatbot(user_question, conversation):
    relevant_questions = search_questions(user_question)
    if relevant_questions.empty:
        retrieved_answer = "متأسفم، پاسخ مناسبی در دیتابیس پیدا نشد."
        url = None
    else:
        retrieved_answers = [
            f"سند: {row['title']}\nلینک: {row['url']}"
            for _, row in relevant_questions.reset_index().iterrows()
        ]
        retrieved_answer = "\n---\n".join(retrieved_answers)
        url = relevant_questions.reset_index()['url'][0]
        st.session_state['last_retrieved_answer'] = retrieved_answer  # **نکته مهم:** فقط آخرین پاسخ بازیابی شده ذخیره می‌شود

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # **نکته مهم:** فقط از آخرین پاسخ بازیابی شده در پیام‌ها استفاده می‌شود
    if st.session_state['last_retrieved_answer'] and st.session_state['last_retrieved_answer'] != "متأسفم، پاسخ مناسبی در دیتابیس پیدا نشد.":
        messages.append(HumanMessage(content=f"اطلاعات بازیابی‌شده:\n{st.session_state['last_retrieved_answer']}"))

    # اضافه کردن تاریخچه گفتگو به پیام‌ها
    for msg in conversation:  # conversation همون st.session_state['messages'][:-1] هست
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))

    messages.append(HumanMessage(content=user_question))

    num_tokens = count_tokens(messages)
    print(f"تعداد توکن‌ها: {num_tokens}")
    st.write(f"تعداد توکن‌ها: {num_tokens}")

    if num_tokens > 8000:
        st.warning("تعداد توکن‌ها از 8K بیشتر شده است!")

    response = llm(messages=messages)
    return response.content, url, num_tokens

# نمایش پیام‌های قبلی
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.write(msg['content'])
    else:
        with st.chat_message("assistant"):
            st.write(msg['content'])

user_question = st.chat_input("سوال خود را وارد کنید:")

if user_question and user_question.strip():
    st.session_state['messages'].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("در حال پردازش..."):
            answer, url, num_tokens = chatbot(user_question, st.session_state['messages'][:-1])
            st.session_state['messages'].append({"role": "assistant", "content":url+"\n\n"+ answer})
            if url:
                st.write(f"[لینک مرتبط به پاسخ]({url})")
            st.write("\n\n")
            st.write(answer)
            st.write(f"تعداد توکن‌ها در این درخواست: {num_tokens}")
