import os
import uuid
import pickle
import json
from typing import List, Dict, Any, Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader, JSONLoader
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

# Constants
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
COLLECTION_NAME = "course_documents"
EMBEDDING_MODEL = "NV-Embed-QA"
LLM_MODEL = "meta/llama-3.3-70b-instruct"
VECTOR_SIZE = 1024


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def setup_vector_db():
    """Set up Qdrant Cloud vector database connection."""
    print("🔗 Connecting to Qdrant Cloud...")
    try:
        QDRANT_URL = os.environ.get("QDRANT_URL")
        QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
        # Initialize Qdrant Cloud client
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        # Verify connection and collection
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            raise Exception(f"Collection '{COLLECTION_NAME}' not found in Qdrant Cloud!")

        # Get collection info
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"✅ Connected to Qdrant Cloud")
        print(f"📊 Collection '{COLLECTION_NAME}' has {collection_info.points_count} points")

        # Initialize embedder (for new queries, not for stored data)
        embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL)

        # Create vector store interface - IMPORTANT: Configure content_payload_key
        vector_db = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embedder,
            content_payload_key="text",  # This tells LangChain where to find the text
            metadata_payload_key="metadata"  # Optional: for additional metadata
        )

        return vector_db, qdrant_client, embedder

    except Exception as e:
        print(f"❌ Failed to connect to Qdrant Cloud: {str(e)}")
        raise

def test_qdrant_connection():
    """Test connection to Qdrant Cloud."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"✅ Qdrant Cloud connection successful")
        print(f"📊 Collection has {collection_info.points_count} points")
        return True
    except Exception as e:
        print(f"❌ Qdrant Cloud connection failed: {str(e)}")
        return False

class DirectQdrantRetriever(BaseRetriever):
    """Direct Qdrant retriever that properly handles payload extraction."""
    
    def __init__(self, qdrant_client, embedder, collection_name, k=5):
        super().__init__()
        self._client = qdrant_client
        self._embedder = embedder
        self._collection_name = collection_name
        self._k = k

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Get relevant documents using direct Qdrant search."""
        print(f"\n🔍 DIRECT QDRANT SEARCH DEBUG")
        print(f"📝 Query: {query}")
        print(f"🎯 Requested results (k): {self._k}")
        
        try:
            # Create embedding for the query
            query_embedding = self._embedder.embed_query(query)
            print(f"✅ Generated query embedding: {len(query_embedding)} dimensions")
            
            # Search Qdrant directly
            search_results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=self._k,
                with_payload=True,
                with_vectors=False
            )
            
            print(f"✅ Qdrant search returned {len(search_results)} results")
            
            # Convert Qdrant results to LangChain Documents
            documents = []
            for i, result in enumerate(search_results):
                try:
                    # Extract text from payload
                    text = result.payload.get('text', '')
                    
                    # Create metadata from payload
                    metadata = {
                        'chunk_id': result.payload.get('chunk_id', str(result.id)),
                        'source': result.payload.get('source', 'unknown'),
                        'page': result.payload.get('page', 0),
                        'score': result.score
                    }
                    
                    # Create Document object
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    
                    documents.append(doc)
                    
                    # Debug info for first few results
                    if i < 3:
                        print(f"🎯 Result {i+1}:")
                        print(f"   Score: {result.score:.4f}")
                        print(f"   Source: {metadata['source']}")
                        print(f"   Chunk ID: {metadata['chunk_id'][:20]}...")
                        print(f"   Text length: {len(text)} chars")
                        print(f"   Text preview: {text[:150]}...")
                        
                except Exception as e:
                    print(f"⚠ Error processing result {i}: {e}")
                    continue
            
            print(f"✅ Returning {len(documents)} valid documents")
            print(f"🔚 END DIRECT QDRANT SEARCH DEBUG\n")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error in direct Qdrant search: {str(e)}")
            print(f"🔚 END DIRECT QDRANT SEARCH DEBUG\n")
            return []

def format_docs(docs):
    """Format document chunks into a string."""
    print(f"\n📄 FORMATTING DOCS DEBUG")
    print(f"📊 Number of docs to format: {len(docs)}")
    
    if docs:
        # Show details of first doc
        print(f"🔍 First doc details:")
        print(f"   Source: {docs[0].metadata.get('source', 'unknown')}")
        print(f"   Score: {docs[0].metadata.get('score', 'N/A')}")
        print(f"   Content length: {len(docs[0].page_content)} chars")
        print(f"   Content preview: {docs[0].page_content[:200]}...")
        
        # Create formatted text
        formatted = "\n\n".join([doc.page_content for doc in docs])
        print(f"📝 Total formatted length: {len(formatted)} characters")
        print(f"🔚 END FORMATTING DEBUG\n")
        return formatted
    else:
        print(f"❌ No documents to format!")
        print(f"🔚 END FORMATTING DEBUG\n")
        return "No relevant documents found."

def setup_chatbot(vector_db):
    """Set up the chatbot with RAG pipeline using Qdrant Cloud - Direct Search."""
    print("🤖 Setting up chatbot with Qdrant Cloud integration...")
    print("⚡ Using Direct Qdrant search for reliable retrieval")
    
    # Create direct Qdrant retriever
    retriever = DirectQdrantRetriever(
        qdrant_client=vector_db.client,
        embedder=vector_db.embeddings,
        collection_name=COLLECTION_NAME,
        k=8
    )
    
    # Set up reranker
    print("📊 Setting up reranker...")
    reranker = NVIDIARerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever
    )

    # Set up LLM
    print("🧠 Setting up LLM...")
    llm = ChatNVIDIA(
        model=LLM_MODEL,
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )

    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create prompt template
    stateful_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a confident and helpful NTU expert that helps students find information about anything within NTU including courses, academic programs. You must answer based on your knowledge base, avoid making assumptions, yet answer confidently and add context to your answers. Do not mention the text, answer as if you are answering yourself not based on some text."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    # Create RAG chain
    conversational_rag_chain = (
        {
            "context": compression_retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
        }
        | stateful_prompt
        | llm
        | StrOutputParser()
    )

    print("✅ Chatbot setup complete - Direct Qdrant Search mode!")
    return conversational_rag_chain, memory

def chatbot_with_memory(query, conversational_rag_chain, memory):
    """Process a query with the chatbot and update memory."""
    try:
        print(f"\n🤖 CHATBOT PROCESSING DEBUG")
        print(f"📝 User query: {query}")
        
        response = conversational_rag_chain.invoke(query)
        
        print(f"🎯 Generated response length: {len(response)} characters")
        print(f"📄 Response preview: {response[:200]}...")
        
        # Update memory
        memory.save_context({"input": query}, {"output": response})
        print(f"💾 Memory updated")
        print(f"🔚 END CHATBOT DEBUG\n")
        
        return response
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"🔚 END CHATBOT DEBUG\n")
        return error_msg

def main():
    """Main execution flow for testing."""
    print("🚀 Starting Qdrant Cloud RAG system...")
    
    try:
        # Test Qdrant connection
        if not test_qdrant_connection():
            print("❌ Cannot proceed without Qdrant connection")
            return None, None

        # Set up vector database connection
        vector_db, client, _ = setup_vector_db()

        # Set up chatbot
        rag_chain, memory = setup_chatbot(vector_db)

        # Test with sample query
        print("\n🧪 Testing chatbot with sample query...")
        sample_query = "what courses are available for computer science?"
        response = chatbot_with_memory(sample_query, rag_chain, memory)
        print(f"📝 Query: {sample_query}")
        print(f"🤖 Response: {response}")

        print("\n✅ RAG system ready to process queries!")
        return rag_chain, memory

    except Exception as e:
        print(f"❌ Error in main setup: {str(e)}")
        return None, None

# Legacy functions for compatibility
def load_chunks_from_disk(filename="document_chunks.pkl"):
    """Return None since we're using Qdrant Cloud now."""
    print("📝 Note: Using Qdrant Cloud for document storage, not local files")
    return []

if __name__ == "__main__":
    main()
