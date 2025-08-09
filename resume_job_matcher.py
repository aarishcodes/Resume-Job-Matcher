import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.title("📄 Resume Matcher & Custom Bullet Generator")

uploaded_pdf = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_url = st.text_input("Paste Job Description URL")

if st.button("Analysis Resume"):
    if not uploaded_pdf or not job_url:
        st.error("Please upload your resume and provide a job description URL.")
    else:
        with st.spinner("⏳ Analyzing and generating custom resume bullets..."):
            # Read PDF content
            pdf_reader = PdfReader(uploaded_pdf)
            resume_text = "\n".join([page.extract_text() for page in pdf_reader.pages])

            # Load job description
            loader = WebBaseLoader(job_url)
            job_description_docs = loader.load()
            jd_text = " ".join([job.page_content for job in job_description_docs])

            # Initialize models
            model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
            embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            jd_chunk = splitter.split_documents([Document(page_content=jd_text)])
            resume_chunk = splitter.split_documents([Document(page_content=resume_text)])

            # Create embeddings
            jd_embeddings = [embedding_model.embed_query(doc.page_content) for doc in jd_chunk]
            resume_embeddings = [embedding_model.embed_query(doc.page_content) for doc in resume_chunk]

            # Compare embeddings
            similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
            average_similarity = np.mean(similarity_matrix)

            st.write(f"### 🔍 Average Similarity Score: **{average_similarity:.2%}**")

            # Extract top relevant JD chunks
            top_jd_indices = np.argsort(-similarity_matrix.mean(axis=0))[:5]
            top_jd_content = "\n".join([jd_chunk[i].page_content for i in top_jd_indices])

            # Prompt for bullet generation
            template = """
            You are an expert resume writer. 
            Here is the job description for which I want to tailor my resume:
            {job_description}

            Here is my current resume content:
            {resume_content}

            Generate 5-7 strong, ATS-friendly bullet points that highlight my skills and achievements for this specific job.
            Each bullet should be concise and impactful.
            """
            prompt_template = PromptTemplate(
                input_variables=["job_description", "resume_content"],
                template=template
            )
            final_prompt = prompt_template.format(
                job_description=top_jd_content,
                resume_content=resume_text
            )

            # Generate custom bullets
            custom_bullets = model.invoke(final_prompt)

        st.subheader("📌 Customized Resume Bullets")
        st.write(custom_bullets.content)
