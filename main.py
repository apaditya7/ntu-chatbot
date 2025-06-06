from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import time
import os
from rag import setup_vector_db, setup_chatbot, chatbot_with_memory, test_qdrant_connection

app = Flask(__name__)
# Configure CORS properly to allow requests from any origin
CORS(app, supports_credentials=True)

# Initialize RAG components
print("🚀 Starting Qdrant Cloud RAG system...")

# Global variables for RAG components
vector_db = None
rag_chain = None
memory = None
chatbot_initialized = False

def initialize_chatbot():
    """Initialize the chatbot with Qdrant Cloud."""
    global vector_db, rag_chain, memory, chatbot_initialized
    
    try:
        print("🔗 Testing Qdrant Cloud connection...")
        if not test_qdrant_connection():
            raise Exception("Cannot connect to Qdrant Cloud")
        
        print("📊 Setting up vector database...")
        vector_db, _, _ = setup_vector_db()
        
        print("🤖 Setting up chatbot...")
        rag_chain, memory = setup_chatbot(vector_db)
        
        chatbot_initialized = True
        print("✅ Chatbot initialized successfully with Qdrant Cloud!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full error traceback for debugging
        chatbot_initialized = False
        return False

# Initialize on startup
initialization_success = initialize_chatbot()

if not initialization_success:
    print("⚠️ Warning: Chatbot initialization failed. API will return errors.")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if chatbot_initialized:
        return jsonify({
            "status": "healthy",
            "qdrant_connected": True,
            "chatbot_ready": True
        }), 200
    else:
        return jsonify({
            "status": "unhealthy", 
            "qdrant_connected": False,
            "chatbot_ready": False
        }), 503

@app.route('/chat/stream', methods=['GET', 'OPTIONS'])
def chat_stream():
    """Streaming chat endpoint."""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
        
    user_message = request.args.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    if not chatbot_initialized:
        error_msg = "❌ Chatbot not initialized. Check server logs for details."
        return Response(
            f"data: {json.dumps({'text': error_msg})}\n\n",
            mimetype='text/event-stream',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
            }
        )
    
    # Set up the SSE response with explicit headers
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'X-Accel-Buffering': 'no'  # Disable proxy buffering for Nginx if used
    }
    
    return Response(
        generate_stream(user_message),
        mimetype='text/event-stream',
        headers=headers
    )

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Non-streaming chat endpoint."""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
        
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "No message provided"}), 400

    if not chatbot_initialized:
        return jsonify({
            "error": "Chatbot not initialized. Check server logs for details.",
            "details": "System failed to initialize properly on startup."
        }), 500

    try:
        print(f"📝 Processing query: {message}")
        response = chatbot_with_memory(message, rag_chain, memory)
        print(f"✅ Response generated successfully")
        return jsonify({"response": response})
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({"error": error_msg}), 500

def generate_stream(message):
    """Generate streaming response."""
    try:
        print(f"📝 Streaming query: {message}")
        full_response = chatbot_with_memory(message, rag_chain, memory)
        
        # Stream the response word by word
        words = full_response.split()
        
        for i, word in enumerate(words):
            text = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'text': text})}\n\n"
            time.sleep(0.05)  # Small delay for streaming effect
        
        # Send completion event
        yield f"event: complete\ndata: {json.dumps({'completed': True})}\n\n"
        print("✅ Streaming response completed")
        
    except Exception as e:
        error_message = f"❌ Error generating response: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'text': error_message})}\n\n"
        yield f"event: complete\ndata: {json.dumps({'completed': True, 'error': True})}\n\n"

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed system status."""
    try:
        if not chatbot_initialized:
            return jsonify({
                "chatbot_initialized": False,
                "qdrant_connected": False,
                "error": "System not properly initialized"
            }), 503
        
        # Test Qdrant connection
        qdrant_connected = test_qdrant_connection()
        
        # Get collection info if connected
        collection_info = {}
        if qdrant_connected and vector_db:
            try:
                collection_data = vector_db.client.get_collection("course_documents")
                collection_info = {
                    "points_count": collection_data.points_count,
                    "collection_name": "course_documents"
                }
            except Exception as e:
                collection_info = {"error": str(e)}
        
        return jsonify({
            "chatbot_initialized": chatbot_initialized,
            "qdrant_connected": qdrant_connected,
            "collection_info": collection_info,
            "status": "operational" if (chatbot_initialized and qdrant_connected) else "degraded"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Status check failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/reinitialize', methods=['POST'])
def reinitialize_system():
    """Reinitialize the chatbot system (useful for debugging)."""
    global chatbot_initialized
    
    print("🔄 Reinitializing chatbot system...")
    success = initialize_chatbot()
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Chatbot reinitialized successfully"
        }), 200
    else:
        return jsonify({
            "status": "failed",
            "message": "Failed to reinitialize chatbot"
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify the server is running."""
    return jsonify({
        "message": "Server is running!",
        "chatbot_status": "initialized" if chatbot_initialized else "not initialized"
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 5001))
    print(f"🌐 Starting Flask server on port {port}")
    print(f"🔗 Health check: http://localhost:{port}/health")
    print(f"📊 Status check: http://localhost:{port}/status")
    print(f"🧪 Test endpoint: http://localhost:{port}/test")
    
    app.run(host='0.0.0.0', debug=True, port=port)