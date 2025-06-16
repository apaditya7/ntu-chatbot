# drive_pipeline.py - Updated version for your repository
import os
import io
import json
import hashlib
import time
from typing import Dict, List, Optional
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
import uuid
import tempfile
import logging

logger = logging.getLogger(__name__)

class DriveEmbeddingPipeline:
    def __init__(self):
        self.drive_service = self._init_drive_service()
        self.qdrant_client = self._init_qdrant_client()
        self.embedder = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            api_key=os.environ.get("NVIDIA_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        # Use the same collection as your existing chatbot
        self.collection_name = "university_materials"
        self._ensure_collection_exists()
        
    def _init_drive_service(self):
        """Initialize Google Drive service."""
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        
        if creds_json:
            try:
                creds_info = json.loads(creds_json)
            except json.JSONDecodeError:
                creds_info = None
        else:
            creds_info = None
        
        if not creds_info:
            # Load from file (for local testing)
            credentials = service_account.Credentials.from_service_account_file(
                "credentials.json",
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
        else:
            credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
        
        return build('drive', 'v3', credentials=credentials)
    
    def _init_qdrant_client(self):
        """Initialize Qdrant client."""
        return QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
    
    def _get_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content to detect duplicates."""
        return hashlib.md5(file_content).hexdigest()
    
    def extract_metadata_from_path(self, file_path: str, filename: str) -> Dict:
        """Extract metadata from Google Drive folder path."""
        path_parts = [part.strip() for part in file_path.split('/') if part.strip()]
        
        metadata = {
            "filename": filename,
            "upload_date": datetime.now().isoformat(),
            "file_path": file_path
        }
        
        if not path_parts:
            metadata["type"] = "unknown"
            return metadata
        
        # Remove 'NTUGPT' from path if it's the root folder
        if path_parts and path_parts[0].lower() == "ntugpt":
            path_parts = path_parts[1:]
        
        if not path_parts:
            metadata["type"] = "root"
            return metadata
        
        # Parse folder structure
        if path_parts[0].lower() == "general":
            metadata["type"] = "general"
            if len(path_parts) > 1:
                metadata["category"] = path_parts[1].lower().replace(" ", "_")
            if len(path_parts) > 2:
                metadata["subcategory"] = path_parts[2].lower().replace(" ", "_")
                
        elif path_parts[0].lower() == "courses":
            metadata["type"] = "course"
            if len(path_parts) > 1:
                course_name = path_parts[1].lower().replace(" ", "_").replace("-", "_")
                metadata["course"] = course_name
            if len(path_parts) > 2:
                year_folder = path_parts[2].lower()
                if year_folder.startswith("year"):
                    metadata["year"] = year_folder.replace("year", "")
                elif year_folder.isdigit():
                    metadata["year"] = year_folder
            if len(path_parts) > 3:
                metadata["subcategory"] = path_parts[3].lower().replace(" ", "_")
        else:
            metadata["type"] = "other"
            metadata["category"] = path_parts[0].lower().replace(" ", "_")
        
        return metadata
    
    def get_file_path_from_drive(self, file_id: str) -> str:
        """Get the full folder path for a file in Google Drive."""
        try:
            file_info = self.drive_service.files().get(
                fileId=file_id,
                fields='name,parents'
            ).execute()
            
            if not file_info.get('parents'):
                return ""
            
            path_parts = []
            current_id = file_info['parents'][0]
            
            while current_id:
                try:
                    folder_info = self.drive_service.files().get(
                        fileId=current_id,
                        fields='name,parents'
                    ).execute()
                    
                    folder_name = folder_info.get('name', '')
                    path_parts.insert(0, folder_name)
                    
                    parents = folder_info.get('parents')
                    current_id = parents[0] if parents else None
                    
                    if not parents or len(path_parts) > 10:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error getting parent folder: {e}")
                    break
            
            return '/'.join(path_parts)
            
        except Exception as e:
            logger.error(f"Error getting file path: {e}")
            return ""
    
    def download_and_extract_text(self, file_id: str, mime_type: str) -> tuple[Optional[str], Optional[str]]:
        """Download file and extract text content. Returns (text, file_hash)."""
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_content.seek(0)
            file_bytes = file_content.getvalue()
            file_hash = self._get_file_hash(file_bytes)
            
            file_extension = self._get_file_extension(mime_type)
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name
            
            try:
                if mime_type == 'application/pdf':
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    text = "\n".join([doc.page_content for doc in documents])
                elif mime_type in [
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/msword'
                ]:
                    loader = UnstructuredWordDocumentLoader(temp_file_path)
                    documents = loader.load()
                    text = "\n".join([doc.page_content for doc in documents])
                elif mime_type.startswith('text/'):
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    logger.warning(f"Unsupported file type: {mime_type}")
                    return None, None
                
                return text, file_hash
                
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error downloading/extracting file {file_id}: {e}")
            return None, None
    
    def _get_file_extension(self, mime_type: str) -> str:
        """Get file extension based on MIME type."""
        mime_to_ext = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/msword': '.doc',
            'text/plain': '.txt'
        }
        return mime_to_ext.get(mime_type, '.txt')
    
    def create_embeddings(self, text: str, metadata: Dict) -> List[PointStruct]:
        """Create embeddings for text chunks."""
        try:
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                return []
            
            points = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                embedding = self.embedder.embed_query(chunk)
                time.sleep(1.0)
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "text": chunk,
                    "chunk_index": i,
                    "chunk_id": f"{metadata.get('file_id', 'unknown')}_{i}",
                    "total_chunks": len(chunks)
                })
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=chunk_metadata
                )
                points.append(point)
            
            return points
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return []
    
    def process_file(self, file_id: str) -> Dict:
        """Process a single file from Google Drive."""
        try:
            logger.info(f"Processing file: {file_id}")
            
            file_info = self.drive_service.files().get(
                fileId=file_id,
                fields='name,mimeType,createdTime,modifiedTime,size'
            ).execute()
            
            filename = file_info['name']
            mime_type = file_info['mimeType']
            
            logger.info(f"File: {filename}, Type: {mime_type}")
            
            supported_types = [
                'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/msword',
                'text/plain'
            ]
            
            if mime_type not in supported_types:
                return {
                    "success": False,
                    "message": f"Unsupported file type: {mime_type}",
                    "file_id": file_id,
                    "filename": filename
                }
            
            text_content, file_hash = self.download_and_extract_text(file_id, mime_type)
            if not text_content:
                return {
                    "success": False,
                    "message": "Failed to extract text content",
                    "file_id": file_id,
                    "filename": filename
                }
            
            file_path = self.get_file_path_from_drive(file_id)
            metadata = self.extract_metadata_from_path(file_path, filename)
            metadata.update({
                'file_id': file_id,
                'file_hash': file_hash,
                'mime_type': mime_type,
                'file_size': file_info.get('size', 0),
                'created_time': file_info.get('createdTime'),
                'modified_time': file_info.get('modifiedTime')
            })
            
            logger.info(f"Extracted metadata: {metadata}")
            
            points = self.create_embeddings(text_content, metadata)
            if not points:
                return {
                    "success": False,
                    "message": "Failed to create embeddings",
                    "file_id": file_id,
                    "filename": filename
                }
            
            # Remove old embeddings for this file if they exist
            self._remove_old_embeddings(file_id)
            
            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully processed {filename}: {len(points)} chunks")
            
            return {
                "success": True,
                "message": f"Successfully processed {filename}",
                "file_id": file_id,
                "filename": filename,
                "chunks_created": len(points),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_id}: {e}")
            return {
                "success": False,
                "message": f"Error processing file: {str(e)}",
                "file_id": file_id,
                "filename": file_info.get('name', 'unknown') if 'file_info' in locals() else 'unknown'
            }
    
    def _remove_old_embeddings(self, file_id: str):
        """Remove old embeddings for a file before adding new ones."""
        try:
            # For now, skip the filtering since it needs indexing
            # Just log that we're processing the file
            logger.info(f"Processing file {file_id} - will add new embeddings")
        except Exception as e:
            logger.error(f"Error removing old embeddings: {e}")
    
    def sync_folder(self, folder_name: str) -> Dict:
        """Sync all files in a specific folder (Courses or General)."""
        try:
            folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            folder_results = self.drive_service.files().list(q=folder_query).execute()
            
            if not folder_results.get('files'):
                return {"success": False, "message": f"Folder '{folder_name}' not found"}
            
            folder_id = folder_results['files'][0]['id']
            all_files = self._get_files_recursively(folder_id)
            
            results = []
            for file_info in all_files:
                result = self.process_file(file_info['id'])
                results.append(result)
            
            success_count = sum(1 for r in results if r['success'])
            
            return {
                "success": True,
                "message": f"Processed {success_count}/{len(results)} files",
                "total_files": len(results),
                "successful": success_count,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error syncing folder {folder_name}: {e}")
            return {"success": False, "message": str(e)}
    
    def _get_files_recursively(self, folder_id: str) -> List[Dict]:
        """Get all files in a folder recursively."""
        files = []
        
        query = f"'{folder_id}' in parents and trashed=false"
        results = self.drive_service.files().list(
            q=query,
            fields='files(id,name,mimeType,parents,createdTime)'  # Added createdTime
        ).execute()
        
        for item in results.get('files', []):
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                files.extend(self._get_files_recursively(item['id']))
            else:
                files.append(item)
        
        return files