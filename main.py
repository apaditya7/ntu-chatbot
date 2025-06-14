from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import time
import os
import logging
from rag import setup_vector_db, setup_chatbot, chatbot_with_memory, test_qdrant_connection

try:
    from drive_pipeline import DriveEmbeddingPipeline
    DRIVE_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Drive pipeline not available: {e}")
    DRIVE_PIPELINE_AVAILABLE = False

app = Flask(__name__)
CORS(app, supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 Starting Qdrant Cloud RAG system...")

vector_db = None
rag_chain = None
memory = None
chatbot_initialized = False
drive_pipeline = None

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
        traceback.print_exc()
        chatbot_initialized = False
        return False

def initialize_drive_pipeline():
    """Initialize the Google Drive embedding pipeline."""
    global drive_pipeline
    
    if not DRIVE_PIPELINE_AVAILABLE:
        print("⚠️ Drive pipeline not available")
        return False
    
    try:
        print("🔗 Initializing Google Drive pipeline...")
        drive_pipeline = DriveEmbeddingPipeline()
        print("✅ Drive pipeline initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing drive pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        drive_pipeline = None
        return False

# Initialize on startup
initialization_success = initialize_chatbot()
drive_pipeline_success = initialize_drive_pipeline()

if not initialization_success:
    print("⚠️ Warning: Chatbot initialization failed. API will return errors.")

if not drive_pipeline_success:
    print("⚠️ Warning: Drive pipeline initialization failed. Drive features disabled.")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if chatbot_initialized else "unhealthy",
        "qdrant_connected": chatbot_initialized,
        "chatbot_ready": chatbot_initialized,
        "drive_pipeline_ready": drive_pipeline is not None
    }), 200 if chatbot_initialized else 503

@app.route('/chat/stream', methods=['GET', 'OPTIONS'])
def chat_stream():
    """Streaming chat endpoint."""
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
    
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'X-Accel-Buffering': 'no'
    }
    
    return Response(
        generate_stream(user_message),
        mimetype='text/event-stream',
        headers=headers
    )

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Non-streaming chat endpoint."""
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
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

def generate_stream(message):
    """Generate streaming response."""
    try:
        print(f"📝 Streaming query: {message}")
        full_response = chatbot_with_memory(message, rag_chain, memory)
        
        words = full_response.split()
        
        for i, word in enumerate(words):
            text = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'text': text})}\n\n"
            time.sleep(0.05)
        
        yield f"event: complete\ndata: {json.dumps({'completed': True})}\n\n"
        print("✅ Streaming response completed")
        
    except Exception as e:
        error_message = f"❌ Error generating response: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'text': error_message})}\n\n"
        yield f"event: complete\ndata: {json.dumps({'completed': True, 'error': True})}\n\n"

# Drive Pipeline Endpoints
@app.route('/admin/process-file', methods=['POST'])
def process_single_file():
    """Manually process a single file by file ID or Drive URL."""
    if not drive_pipeline:
        return jsonify({"error": "Drive pipeline not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        file_input = data.get('file_id') or data.get('file_url')
        if not file_input:
            return jsonify({"error": "file_id or file_url required"}), 400
        
        # Extract file ID from URL if needed
        if 'drive.google.com' in file_input:
            try:
                file_id = file_input.split('/d/')[1].split('/')[0]
            except IndexError:
                return jsonify({"error": "Invalid Google Drive URL"}), 400
        else:
            file_id = file_input
        
        result = drive_pipeline.process_file(file_id)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in process_single_file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/sync-folder', methods=['POST'])
def sync_folder():
    """Sync all files in a folder (Courses or General)."""
    if not drive_pipeline:
        return jsonify({"error": "Drive pipeline not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        folder_name = data.get('folder_name')
        if not folder_name:
            return jsonify({"error": "folder_name required"}), 400
        
        if folder_name not in ['Courses', 'General']:
            return jsonify({"error": "folder_name must be 'Courses' or 'General'"}), 400
        
        result = drive_pipeline.sync_folder(folder_name)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in sync_folder: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/drive-status', methods=['GET'])
def drive_status():
    """Check drive pipeline status and connection."""
    if not drive_pipeline:
        return jsonify({
            "status": "error",
            "message": "Drive pipeline not initialized"
        }), 500
    
    try:
        # Test drive connection
        results = drive_pipeline.drive_service.files().list(pageSize=1).execute()
        
        # Test Qdrant connection
        collections = drive_pipeline.qdrant_client.get_collections()
        collection_info = None
        for col in collections.collections:
            if col.name == drive_pipeline.collection_name:
                collection_info = drive_pipeline.qdrant_client.get_collection(col.name)
                break
        
        return jsonify({
            "status": "healthy",
            "drive_connected": True,
            "qdrant_connected": True,
            "collection_name": drive_pipeline.collection_name,
            "total_embeddings": collection_info.points_count if collection_info else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking drive status: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/webhook/drive', methods=['POST', 'GET'])
def handle_drive_webhook():
    """Handle Google Drive webhook notifications."""
    try:
        if request.method == 'GET':
            challenge = request.args.get('challenge')
            if challenge:
                return challenge, 200
            return 'Webhook endpoint ready', 200
        
        headers = dict(request.headers)
        logger.info(f"Drive webhook received: {headers}")
        
        resource_state = headers.get('X-Goog-Resource-State')
        resource_id = headers.get('X-Goog-Resource-Id') 
        changed_fields = headers.get('X-Goog-Changed', '')
        
        logger.info(f"Drive notification: state={resource_state}, id={resource_id}, changed={changed_fields}")

        if resource_state == 'update' and 'children' in changed_fields and drive_pipeline:
            try:
                query = f"'{resource_id}' in parents and trashed=false"
                results = drive_pipeline.drive_service.files().list(
                    q=query,
                    orderBy='createdTime desc',
                    pageSize=5, 
                    fields='files(id,name,mimeType,createdTime)'
                ).execute()
                
                files = results.get('files', [])
                logger.info(f"Found {len(files)} files in folder")
                
                # Process each file
                for file_info in files:
                    file_id = file_info['id']
                    logger.info(f"Processing file from folder: {file_info['name']}")
                    result = drive_pipeline.process_file(file_id)
                    logger.info(f"Processed file {file_id}: {result.get('success', False)}")
                    
            except Exception as e:
                logger.error(f"Error processing folder change: {e}")
        
        return 'OK', 200
        
    except Exception as e:
        logger.error(f"Error handling drive webhook: {e}")
        return 'Error', 500

@app.route('/admin/setup-webhook', methods=['POST'])
def setup_drive_webhook():
    """Set up Google Drive webhook for a folder."""
    if not drive_pipeline:
        return jsonify({"error": "Drive pipeline not initialized"}), 500
    
    try:
        data = request.get_json()
        folder_name = data.get('folder_name', 'Courses') if data else 'Courses'
        
        # Get your Cloud Run URL (update this with your actual URL)
        webhook_url = f"https://ntu-chatbot-876229082962.us-central1.run.app/webhook/drive"
        
        # Find the folder ID
        folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        folder_results = drive_pipeline.drive_service.files().list(q=folder_query).execute()
        
        if not folder_results.get('files'):
            return jsonify({"error": f"Folder '{folder_name}' not found"}), 400
        
        folder_id = folder_results['files'][0]['id']
        
        # Set up webhook channel
        channel_body = {
            'id': f"channel-{folder_name}-{int(time.time())}",
            'type': 'web_hook',
            'address': webhook_url,
            'token': f'folder={folder_name}',
            'expiration': int((time.time() + 24*60*60) * 1000)
        }
        
        # Create the watch
        watch_response = drive_pipeline.drive_service.files().watch(
            fileId=folder_id,
            body=channel_body,
            supportsAllDrives=True
        ).execute()
        
        logger.info(f"Webhook created: {watch_response}")
        
        return jsonify({
            "success": True,
            "message": f"Webhook set up for {folder_name}",
            "channel_id": watch_response.get('id'),
            "resource_id": watch_response.get('resourceId'),
            "webhook_url": webhook_url,
            "expiration": watch_response.get('expiration')
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting up webhook: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/renew-webhooks', methods=['POST'])
def renew_webhooks():
    """Renew webhooks for all monitored folders."""
    if not drive_pipeline:
        return jsonify({"error": "Drive pipeline not initialized"}), 500
    
    try:
        folders_to_watch = ['Courses', 'General']
        results = []
        
        for folder_name in folders_to_watch:
            try:
                # Set up new webhook
                result = setup_drive_webhook_internal(folder_name)
                results.append({
                    "folder": folder_name,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "folder": folder_name,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "message": "Webhook renewal completed",
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def setup_drive_webhook_internal(folder_name):
    """Internal function to set up webhook."""
    if not drive_pipeline:
        raise Exception("Drive pipeline not initialized")
    
    webhook_url = f"https://ntu-chatbot-876229082962.us-central1.run.app/webhook/drive"
    
    # Find folder
    folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    folder_results = drive_pipeline.drive_service.files().list(q=folder_query).execute()
    
    if not folder_results.get('files'):
        raise Exception(f"Folder '{folder_name}' not found")
    
    folder_id = folder_results['files'][0]['id']
    
    # Create webhook
    channel_body = {
        'id': f"channel-{folder_name}-{int(time.time())}",
        'type': 'web_hook', 
        'address': webhook_url,
        'token': f'folder={folder_name}',
        'expiration': int((time.time() + 24*60*60) * 1000)
    }
    
    return drive_pipeline.drive_service.files().watch(
        fileId=folder_id,
        body=channel_body,
        supportsAllDrives=True
    ).execute()

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed system status."""
    try:
        if not chatbot_initialized:
            return jsonify({
                "chatbot_initialized": False,
                "qdrant_connected": False,
                "drive_pipeline_ready": drive_pipeline is not None,
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
            "drive_pipeline_ready": drive_pipeline is not None,
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
        "chatbot_status": "initialized" if chatbot_initialized else "not initialized",
        "drive_pipeline_status": "initialized" if drive_pipeline else "not initialized"
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
    
    if drive_pipeline:
        print(f"🔗 Drive admin: http://localhost:{port}/admin/drive-status")
        print(f"📤 Process file: http://localhost:{port}/admin/process-file")
        print(f"🔄 Sync folder: http://localhost:{port}/admin/sync-folder")
    
    app.run(host='0.0.0.0', debug=True, port=port)
