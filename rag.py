import os
import logging
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader  
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/academics"
COLLECTION_NAME = "vit_academics"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an **Academic Assistant** for VIT University. "
     "Your role is to help students by providing clear, structured, and accurate academic information "
     "about courses, subjects, credits, prerequisites, labs, and other program-related details. "
     "Always present answers in a structured format using tables when possible. "
     "The database context is loaded from a CSV file, so use the provided dataset context directly "
     "for accuracy and avoid assumptions. "
     "If information is missing or not available in the dataset, clearly reply with: 'I don’t know'. "
     "Do not hallucinate or guess. "
     "Always ensure responses are student-friendly, concise, and easy to understand."
    ),
    ("human", 
     "Question: {question}\n\nCSV Context:\n{context}\n\nAnswer in a structured format:")
])


def initialize_llm() -> OllamaLLM:
    """Initialize local Ollama LLM."""
    return OllamaLLM(model="llama2", temperature=0.3)

def initialize_vectorstore() -> Chroma:
    """Initialize Chroma vector store with embedding model."""
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(VECTORSTORE_DIR)
    )

def process_csv_to_vectorstore(csv_path: str, vector_store: Chroma, force_rebuild: bool = False):
    """Load academic data from CSV and store it in the vector DB."""
    existing_docs = vector_store._collection.count()
    if existing_docs > 0 and not force_rebuild:
        logger.info("Vector store already has %d documents. Skipping rebuild.", existing_docs)
        return

    logger.info("Loading CSV data from %s", csv_path)
    loader = CSVLoader(
        file_path=csv_path,
        csv_args={
            "delimiter": ",",
            "quotechar": '"'
        }
    )
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        logger.warning("No documents found in the CSV.")
        return

    logger.info("Adding %d documents to vector store...", len(docs))
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    logger.info("Data successfully added to vector store.")

def generate_answer(query: str, llm: OllamaLLM, vector_store: Chroma) -> str:
    """Run retrieval and LLM chain to answer query."""
    retriever = vector_store.as_retriever()

    chain = (
        retriever
        | (lambda docs: {
            "context": "\n\n".join([d.page_content for d in docs]),
            "question": query
        })
        | chat_prompt
        | llm
    )

    result = chain.invoke(query)
    return result.content if hasattr(result, "content") else str(result)

def academic_assistant(query: str) -> str:
    """Main RAG entrypoint."""
    llm = initialize_llm()
    vector_store = initialize_vectorstore()
    return generate_answer(query, llm, vector_store)

# def create_agent() -> Agent:
#     """Create the AGNO agent with academic assistant and reasoning tools."""
#     return Agent(
#         model=Gemini(id="gemini-2.0-flash"),
#         tools=[
#             ReasoningTools(add_instructions=True),
#             academic_assistant,
#         ],
#         instructions=[
#             "Use tables to display course data when helpful.",
#             "Only output the answer, no filler text.",
#         ],
#         markdown=True,
#         show_tool_calls=False,
#     )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Academic Assistant (CSV-based)")
    parser.add_argument("--csv", type=str, help="Path to academic CSV file", default="result.csv")
    parser.add_argument("--query", type=str, help="Academic question to answer", default="List electives related to AI")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector DB from CSV")

    args = parser.parse_args()

    llm = initialize_llm()
    vector_store = initialize_vectorstore()

    process_csv_to_vectorstore(args.csv, vector_store, force_rebuild=args.rebuild)

    # agent = create_agent()
    # logger.info("Running query: %s", args.query)
    # agent.print_response(args.query, stream=True)

    answer = generate_answer(args.query, llm, vector_store)
    print("\n=== Answer ===\n")
    print(answer)
