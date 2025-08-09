from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.schema import Document
from langchain.prompts import PromptTemplate


load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')


# PdF Load ---->
loader = PyPDFLoader("resume.pdf")
doc=loader.load()
resume_text = " ".join([page.page_content for page in doc])
# print(resume_text)
# print(doc[0].page_content)



#Web Site Data Loader
url='https://amazon.jobs/en/jobs/3055226/software-engineer-i'
loader = WebBaseLoader(url)
job_description = loader.load()
jd = " ".join([job.page_content for job in job_description])
# print(jd)
# print(job_des[0].page_content)


spliter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)


jd_chunk = spliter.split_documents([Document(page_content=jd)])
resume_chunk = spliter.split_documents([Document(page_content=resume_text)])
# jd_chunk = spliter.split_text(jd)
# resume_chunk = spliter.split_text(resume_text)
# print(jd_chunk[0])
# print(resume_chunk[0])



# Vector Store
# vector_store_jd = FAISS.from_documents(jd_chunk, embedding_model)
# vector_store_resume = FAISS.from_documents(resume_chunk, embedding_model)


# Retrievers 
# retriever_jd = vector_store_jd.as_retriever(search_type='similarity', search_kwargs={"k":3})
# retriever_resume = vector_store_resume.as_retriever(search_type='similarity', search_kwargs={"k":3})

# Get embeddings for all chunks
jd_embeddings = [embedding_model.embed_query(doc.page_content) for doc in jd_chunk]
resume_embeddings = [embedding_model.embed_query(doc.page_content) for doc in resume_chunk]

# Compare them
similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)

average_similarity = np.mean(similarity_matrix)
print(f"Average similarity score: {average_similarity:.2%}")


#creating a good suggestion for the Candidate
top_jd_indices = np.argsort(-similarity_matrix.mean(axis=0))[:5]
top_jd_content = "\n".join([jd_chunk[i].page_content for i in top_jd_indices])

# generate customized resume bullets
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

# Format with actual values
final_prompt = prompt_template.format(
    job_description=top_jd_content,
    resume_content=resume_text
)

custom_bullets = model.invoke(final_prompt)
print("\nCustomized Resume Bullets:\n")
print(custom_bullets.content)


