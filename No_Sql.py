import os
import json
import re
import time
import threading
from typing import Any, Dict, Tuple, Optional
from pymongo import MongoClient
import logging
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from tqdm import tqdm
import asyncio
from dotenv import load_dotenv
load_dotenv()
# LangChain core
from langchain_core.prompts import PromptTemplate

# Groq LangChain integration
from langchain_groq import ChatGroq

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Groq model options (much faster than Gemini)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")  # Default fast model
# Other options: "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"

class ProgressTracker:
    """Helper class to manage progress bars for different operations"""
    
    def __init__(self):
        self.bars = {}
    
    def create_bar(self, name: str, total: int, desc: str):
        """Create a new progress bar"""
        self.bars[name] = tqdm(
            total=total, 
            desc=desc, 
            unit="step",
            ncols=80,
            leave=True
        )
        return self.bars[name]
    
    def update_bar(self, name: str, n: int = 1):
        """Update progress bar"""
        if name in self.bars:
            self.bars[name].update(n)
    
    def close_bar(self, name: str):
        """Close and remove progress bar"""
        if name in self.bars:
            self.bars[name].close()
            del self.bars[name]
    
    def close_all(self):
        """Close all progress bars"""
        for bar in self.bars.values():
            bar.close()
        self.bars.clear()

class TimeoutHandler:
    """Handle timeouts with proper cleanup"""
    
    @staticmethod
    def run_with_timeout(func, timeout_seconds, *args, **kwargs):
        """Run function with timeout and proper error handling"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]

class NLPProcessor:
    def __init__(self):
        self._client = None
        self._llm = None
        self.progress = ProgressTracker()
        self._setup_prompt_template()
        logger.info("🚀 NLP Processor initialized with Groq API")

    def _setup_prompt_template(self):
        self.prompt_template = PromptTemplate.from_template("""
You are an expert MongoDB query generator. Convert natural language to MongoDB find queries.

DATABASE SCHEMA - EMPLOYEES:
- name (string): Employee's full name
- age (number): Employee's age in years  
- department (string): Department name (engineering, marketing, sales, hr)
- position (string): Job title/position
- salary (number): Annual salary in USD
- experience_years (number): Years of work experience
- location (string): Work location/office
- joining_date (string): Date in YYYY-MM-DD format

OUTPUT RULES:
1. Return ONLY valid JSON with "filter" and "projection" keys
2. Use MongoDB operators: $gt, $lt, $gte, $lte, $eq, $ne, $in, $regex
3. For text search: {{"$regex": "value", "$options": "i"}}
4. Empty projection {{}} if no specific fields requested
5. Be fast and concise

EXAMPLES:
Input: "Find engineers"
Output: {{"filter": {{"department": {{"$regex": "engineering", "$options": "i"}}}}, "projection": {{}}}}

Input: "employees earning over 60000"  
Output: {{"filter": {{"salary": {{"$gt": 60000}}}}, "projection": {{}}}}

Input: "names of marketing staff"
Output: {{"filter": {{"department": {{"$regex": "marketing", "$options": "i"}}}}, "projection": {{"name": 1}}}}

USER QUERY: {user_input}

JSON OUTPUT:""")

    def get_mongo_client(self, show_progress: bool = True):
        if self._client is None:
            if show_progress:
                pbar = self.progress.create_bar("mongo_connect", 3, "🔌 Connecting to MongoDB")
            
            try:
                if show_progress:
                    self.progress.update_bar("mongo_connect")
                
                self._client = MongoClient(
                    MONGODB_URI,
                    serverSelectionTimeoutMS=8000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=15000,
                    maxPoolSize=10,
                    retryWrites=True
                )
                
                if show_progress:
                    self.progress.update_bar("mongo_connect")
                
                # Test connection
                self._client.admin.command('ping', maxTimeMS=3000)
                
                if show_progress:
                    self.progress.update_bar("mongo_connect")
                    self.progress.close_bar("mongo_connect")
                
                logger.info("✅ MongoDB connection established")
                
            except (ServerSelectionTimeoutError, ConnectionFailure) as e:
                if show_progress:
                    self.progress.close_bar("mongo_connect")
                logger.error(f"❌ MongoDB connection failed: {e}")
                raise Exception("MongoDB connection failed. Make sure MongoDB is running on localhost:27017")
        return self._client

    def get_llm(self, show_progress: bool = True):
        if self._llm is None:
            if show_progress:
                pbar = self.progress.create_bar("llm_init", 2, "🤖 Initializing Groq LLM")
            
            if not GROQ_API_KEY:
                if show_progress:
                    self.progress.close_bar("llm_init")
                raise Exception("Please set GROQ_API_KEY environment variable. Get your free API key from https://console.groq.com/")
            
            try:
                if show_progress:
                    self.progress.update_bar("llm_init")
                
                self._llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=GROQ_MODEL,
                    temperature=0,
                    max_tokens=512,  # Limit for faster responses
                    timeout=15,  # Much shorter timeout - Groq is fast!
                    max_retries=2,
                )
                
                if show_progress:
                    self.progress.update_bar("llm_init")
                    self.progress.close_bar("llm_init")
                
                logger.info(f"✅ Groq LLM initialized successfully with model: {GROQ_MODEL}")
                
            except Exception as e:
                if show_progress:
                    self.progress.close_bar("llm_init")
                logger.error(f"❌ Groq LLM initialization failed: {e}")
                raise Exception(f"Groq LLM initialization failed: {str(e)}")
        return self._llm

    def generate_mongo_query(self, nl_text: str, show_progress: bool = True, max_timeout: int = 20) -> Dict[str, Any]:
        """Generate MongoDB query with Groq (much faster than Gemini!)"""
        if not nl_text or not nl_text.strip():
            raise Exception("Query text cannot be empty")
        
        if show_progress:
            pbar = self.progress.create_bar("query_gen", 4, "🧠 Generating MongoDB Query")
        
        try:
            if show_progress:
                self.progress.update_bar("query_gen")
            
            llm = self.get_llm(show_progress=False)
            
            if show_progress:
                self.progress.update_bar("query_gen")
            
            chain = self.prompt_template | llm
            
            # Groq is much faster, so we can use shorter timeout
            start_time = time.time()
            
            def _call_llm():
                return chain.invoke({"user_input": nl_text.strip()})
            
            # Use timeout handler for safety
            raw_msg = TimeoutHandler.run_with_timeout(_call_llm, max_timeout)
            
            elapsed = time.time() - start_time
            logger.info(f"⚡ Groq LLM processed query in {elapsed:.2f} seconds")
            
            if show_progress:
                self.progress.update_bar("query_gen")
            
            raw = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
            
            # Parse the response
            parsed_query = self._parse_llm_response(raw, nl_text)
            
            if show_progress:
                self.progress.update_bar("query_gen")
                self.progress.close_bar("query_gen")
            
            logger.info("✅ Query generation completed successfully")
            return parsed_query
            
        except TimeoutError:
            if show_progress:
                self.progress.close_bar("query_gen")
            logger.warning(f"⚠️ Query generation timed out after {max_timeout}s, using fallback")
            return self._create_fallback_query(nl_text)
            
        except Exception as e:
            if show_progress:
                self.progress.close_bar("query_gen")
            logger.error(f"❌ Query generation failed: {e}")
            return self._create_fallback_query(nl_text)

    def _parse_llm_response(self, raw_response: str, original_query: str) -> Dict[str, Any]:
        """Parse Groq LLM response with multiple fallback strategies"""
        # Clean the response
        cleaned = re.sub(r"```json|```|`", "", raw_response).strip()
        
        # Try direct JSON parsing
        try:
            obj = json.loads(cleaned)
            return self._validate_query_object(obj)
        except json.JSONDecodeError:
            pass
        
        # Try fixing common JSON issues
        try:
            # Fix single quotes and other common issues
            fixed = cleaned.replace("'", '"')
            fixed = re.sub(r'(\w+):', r'"\1":', fixed)  # Add quotes to keys
            obj = json.loads(fixed)
            return self._validate_query_object(obj)
        except json.JSONDecodeError:
            pass
        
        # Extract JSON from text using regex
        try:
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                obj = json.loads(json_match.group())
                return self._validate_query_object(obj)
        except:
            pass
        
        # If all parsing fails, create smart fallback
        logger.warning("⚠️ Could not parse Groq response, using intelligent fallback")
        return self._create_fallback_query(original_query)

    def _validate_query_object(self, obj) -> Dict[str, Any]:
        """Validate and fix query object structure"""
        if not isinstance(obj, dict):
            return {"filter": {}, "projection": {}}
        
        if "filter" not in obj:
            obj["filter"] = {}
        if "projection" not in obj:
            obj["projection"] = {}
        
        return obj

    def _create_fallback_query(self, nl_text: str) -> Dict[str, Any]:
        """Create intelligent fallback query based on keywords"""
        text_lower = nl_text.lower()
        logger.info(f"Creating fallback query for: {nl_text}")
        
        # Salary-based queries
        if any(word in text_lower for word in ['salary', 'earn', 'pay', 'income']):
            numbers = re.findall(r'\d+', nl_text)
            if numbers:
                salary_num = int(numbers[0])
                if any(word in text_lower for word in ['above', 'over', 'more', 'greater', '>']):
                    return {"filter": {"salary": {"$gt": salary_num}}, "projection": {}}
                elif any(word in text_lower for word in ['below', 'under', 'less', '<']):
                    return {"filter": {"salary": {"$lt": salary_num}}, "projection": {}}
                else:
                    return {"filter": {"salary": salary_num}, "projection": {}}
        
        # Age-based queries
        if 'age' in text_lower:
            numbers = re.findall(r'\d+', nl_text)
            if numbers:
                age_num = int(numbers[0])
                if any(word in text_lower for word in ['older', 'above', 'over']):
                    return {"filter": {"age": {"$gt": age_num}}, "projection": {}}
                elif any(word in text_lower for word in ['younger', 'below', 'under']):
                    return {"filter": {"age": {"$lt": age_num}}, "projection": {}}
        
        # Department-based queries
        departments = {
            'engineer': 'engineering',
            'engineering': 'engineering', 
            'developer': 'engineering',
            'dev': 'engineering',
            'marketing': 'marketing',
            'market': 'marketing',
            'sales': 'sales',
            'sale': 'sales',
            'hr': 'hr',
            'human': 'hr'
        }
        
        for keyword, dept in departments.items():
            if keyword in text_lower:
                return {"filter": {"department": {"$regex": dept, "$options": "i"}}, "projection": {}}
        
        # Name-specific queries
        if any(word in text_lower for word in ['name', 'names']):
            return {"filter": {}, "projection": {"name": 1}}
        
        # Experience-based queries
        if 'experience' in text_lower:
            numbers = re.findall(r'\d+', nl_text)
            if numbers:
                exp_num = int(numbers[0])
                if any(word in text_lower for word in ['more', 'over', 'above']):
                    return {"filter": {"experience_years": {"$gt": exp_num}}, "projection": {}}
                elif any(word in text_lower for word in ['less', 'under', 'below']):
                    return {"filter": {"experience_years": {"$lt": exp_num}}, "projection": {}}
        
        # Default fallback - return all documents
        logger.info("Using default fallback query (find all)")
        return {"filter": {}, "projection": {}}

    def run_mongo_query(self, db_name: str, coll_name: str, query: Dict[str, Any], 
                       limit: int = 50, show_progress: bool = True) -> Tuple[int, list]:
        if show_progress:
            pbar = self.progress.create_bar("db_query", 5, "💾 Executing Database Query")
        
        try:
            if show_progress:
                self.progress.update_bar("db_query")
            
            client = self.get_mongo_client(show_progress=False)
            db = client[db_name]
            coll = db[coll_name]
            
            if show_progress:
                self.progress.update_bar("db_query")
            
            _filter = query.get("filter", {})
            _projection = query.get("projection") or None
            limit = max(1, min(limit, 1000))
            
            if show_progress:
                self.progress.update_bar("db_query")
            
            # Execute query with timeout
            start_time = time.time()
            cursor = coll.find(_filter, _projection).limit(limit).max_time_ms(10000)
            results = list(cursor)
            query_elapsed = time.time() - start_time
            
            logger.info(f"💾 Database query executed in {query_elapsed:.2f} seconds, found {len(results)} results")
            
            if show_progress:
                self.progress.update_bar("db_query")
            
            # Count documents with timeout
            try:
                count_start = time.time()
                total = coll.count_documents(_filter, maxTimeMS=3000) if _filter else coll.estimated_document_count()
                count_elapsed = time.time() - count_start
                logger.info(f"📊 Document count took {count_elapsed:.2f} seconds")
            except Exception as e:
                logger.warning(f"⚠️ Count operation timed out: {e}")
                total = len(results)
            
            # Quick result sanitization
            sanitized_results = []
            for r in results:
                sanitized = {}
                for k, v in r.items():
                    try:
                        json.dumps(v)  # Test JSON serialization
                        sanitized[k] = v
                    except (TypeError, ValueError):
                        sanitized[k] = str(v)
                sanitized_results.append(sanitized)
            
            if show_progress:
                self.progress.update_bar("db_query")
                self.progress.close_bar("db_query")
            
            return total, sanitized_results
            
        except Exception as e:
            if show_progress:
                self.progress.close_bar("db_query")
            logger.error(f"❌ Database query failed: {e}")
            raise Exception(f"Database query failed: {str(e)}")

    def test_connections(self, show_progress: bool = True) -> Dict[str, bool]:
        results = {"mongodb": False, "llm": False, "overall": False}
        
        if show_progress:
            pbar = self.progress.create_bar("conn_test", 2, "🔍 Testing Connections")
        
        # Test MongoDB
        try:
            TimeoutHandler.run_with_timeout(lambda: self.get_mongo_client(show_progress=False), 8)
            results["mongodb"] = True
            logger.info("✅ MongoDB connection test passed")
            if show_progress:
                self.progress.update_bar("conn_test")
        except Exception as e:
            if show_progress:
                self.progress.update_bar("conn_test")
            logger.error(f"❌ MongoDB connection test failed: {e}")
        
        # Test Groq LLM
        try:
            TimeoutHandler.run_with_timeout(lambda: self.get_llm(show_progress=False), 10)
            results["llm"] = True
            logger.info("✅ Groq LLM connection test passed")
            if show_progress:
                self.progress.update_bar("conn_test")
        except Exception as e:
            if show_progress:
                self.progress.update_bar("conn_test")
            logger.error(f"❌ Groq LLM connection test failed: {e}")
        
        results["overall"] = results["mongodb"] and results["llm"]
        
        if show_progress:
            self.progress.close_bar("conn_test")
        
        return results

    def close_connections(self):
        """Clean up connections and progress bars"""
        self.progress.close_all()
        if self._client:
            self._client.close()
            self._client = None
            logger.info("🔌 MongoDB connections closed")

# Enhanced wrapper functions with timeout handling
def generate_mongo_query_with_timeout(nl_text: str, timeout: int = 15, show_progress: bool = True) -> Dict[str, Any]:
    """Generate MongoDB query with Groq (much faster timeout)"""
    def target():
        return processor.generate_mongo_query(nl_text, show_progress=show_progress, max_timeout=timeout-2)
    
    try:
        return TimeoutHandler.run_with_timeout(target, timeout)
    except TimeoutError:
        logger.error(f"Query generation timed out after {timeout} seconds")
        processor.progress.close_all()
        return processor._create_fallback_query(nl_text)
    except Exception as e:
        logger.error(f"Error in generate_mongo_query_with_timeout: {e}")
        processor.progress.close_all()
        return processor._create_fallback_query(nl_text)

# Global processor instance
processor = NLPProcessor()

# Public interface functions
def generate_mongo_query(nl_text: str, show_progress: bool = False) -> Dict[str, Any]:
    """Generate MongoDB query using Groq API"""
    return generate_mongo_query_with_timeout(nl_text, timeout=15, show_progress=show_progress)

def run_mongo_query(db_name: str, coll_name: str, query: Dict[str, Any], 
                   limit: int = 50, show_progress: bool = False) -> Tuple[int, list]:
    """Run MongoDB query with optional progress tracking"""
    return processor.run_mongo_query(db_name, coll_name, query, limit, show_progress=show_progress)

def test_connections(show_progress: bool = False) -> Dict[str, bool]:
    """Test connections with optional progress tracking"""
    return processor.test_connections(show_progress=show_progress)

# Cleanup function
def cleanup():
    """Clean up all resources"""
    processor.close_connections()