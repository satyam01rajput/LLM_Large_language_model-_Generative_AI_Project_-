import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import requests
from io import StringIO, BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Functions to process various file types
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_csv_text(csv_files):
    text = ""
    for csv in csv_files:
        df = pd.read_csv(csv)
        text += df.to_string()
    return text

def get_excel_text(excel_files):
    text = ""
    for excel in excel_files:
        df = pd.read_excel(excel)
        text += df.to_string()
    return text

def get_url_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error("Failed to fetch data from the URL. Please check the link.")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


# Main App
def main():
    st.set_page_config("Chat Data", layout="wide")
    st.header("Ask Questions from Your Uploaded Data")

    user_question = st.text_input("Enter your question:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        file_type = st.radio("Choose File Type", options=["PDF", "CSV", "Excel", "URL"])
        uploaded_files = None

        if file_type == "PDF":
            uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        elif file_type == "CSV":
            uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True, type=["csv"])
        elif file_type == "Excel":
            uploaded_files = st.file_uploader("Upload Excel Files", accept_multiple_files=True, type=["xls", "xlsx"])
        elif file_type == "URL":
            url = st.text_input("Enter URL")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                if file_type == "PDF" and uploaded_files:
                    raw_text = get_pdf_text(uploaded_files)
                elif file_type == "CSV" and uploaded_files:
                    raw_text = get_csv_text(uploaded_files)
                elif file_type == "Excel" and uploaded_files:
                    raw_text = get_excel_text(uploaded_files)
                elif file_type == "URL" and url:
                    raw_text = get_url_text(url)
                else:
                    st.error("Please upload a valid file or enter a valid URL.")





# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     st.write("Reply: ", response["output_text"])

# # Main App
# def main():
#     st.set_page_config("Chat Data", layout="wide")
#     st.header("Ask Questions from Your Uploaded Data")

#     user_question = st.text_input("Enter your question:")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu")
#         file_type = st.radio("Choose File Type", options=["PDF", "CSV", "Excel", "URL"])
#         uploaded_files = None

#         if file_type == "PDF":
#             uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
#         elif file_type == "CSV":
#             uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True, type=["csv"])
#         elif file_type == "Excel":
#             uploaded_files = st.file_uploader("Upload Excel Files", accept_multiple_files=True, type=["xls", "xlsx"])
#         elif file_type == "URL":
#             url = st.text_input("Enter URL")

#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = ""
#                 if file_type == "PDF" and uploaded_files:
#                     raw_text = get_pdf_text(uploaded_files)
#                 elif file_type == "CSV" and uploaded_files:
#                     raw_text = get_csv_text(uploaded_files)
#                 elif file_type == "Excel" and uploaded_files:
#                     raw_text = get_excel_text(uploaded_files)
#                 elif file_type == "URL" and url:
#                     raw_text = get_url_text(url)
#                 else:






































                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Data processed successfully!")

if __name__ == "__main__":
    main()
