import streamlit as st
import requests
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import threading

# Config
API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300

st.set_page_config(page_title="NLP to NoSQL", page_icon="🔍", layout="wide")

class NLPQueryInterface:
    def __init__(self):
        self.query_history = []
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
    def check_api_health(self) -> Dict[str, Any]:
        """Check if the API server is healthy"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"status": "unreachable", "error": "Cannot connect to API server"}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "error": "Health check timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def send_query(self, query: str, db: str, collection: str, limit: int, 
                   timeout: int = DEFAULT_TIMEOUT, show_progress: bool = True) -> Optional[Dict[str, Any]]:
        """Send query to API with enhanced error handling and progress tracking"""
        try:
            payload = {
                "input": query, 
                "db": db, 
                "collection": collection, 
                "limit": limit,
                "timeout": timeout,
                "show_progress": show_progress
            }
            
            # Show progress bar while waiting for response
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress():
                """Update progress bar during request"""
                for i in range(timeout):
                    if not hasattr(update_progress, 'running'):
                        break
                    progress_bar.progress((i + 1) / timeout)
                    status_text.text(f"⏳ Processing query... ({i + 1}/{timeout}s)")
                    time.sleep(1)
            
            # Start progress tracking in background
            update_progress.running = True
            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query", 
                    json=payload, 
                    timeout=timeout + 5  # Add buffer to prevent client timeout before server
                )
                
                # Stop progress tracking
                update_progress.running = False
                progress_thread.join(timeout=0.1)
                
                elapsed_time = time.time() - start_time
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                try:
                    data = response.json()
                except Exception:
                    return {
                        "ok": False, 
                        "error": f"Non-JSON response (status {response.status_code})", 
                        "status_code": response.status_code,
                        "elapsed_time": elapsed_time
                    }

                if response.status_code == 200:
                    data["elapsed_time"] = elapsed_time
                    # Add to history
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "db": db,
                        "collection": collection,
                        "success": True,
                        "execution_time": data.get("execution_time", elapsed_time),
                        "result_count": data.get("result_count", 0)
                    })
                    return data
                else:
                    error_data = {
                        "ok": False,
                        "error": data.get("error", f"HTTP {response.status_code}"),
                        "error_type": data.get("error_type", "UNKNOWN"),
                        "status_code": response.status_code,
                        "elapsed_time": elapsed_time
                    }
                    
                    # Add failed query to history
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "db": db,
                        "collection": collection,
                        "success": False,
                        "error": error_data["error"],
                        "execution_time": elapsed_time
                    })
                    
                    return error_data
                    
            except requests.exceptions.Timeout:
                update_progress.running = False
                progress_bar.empty()
                status_text.empty()
                
                error_data = {
                    "ok": False, 
                    "error": f"Request timed out after {timeout} seconds", 
                    "status_code": 408,
                    "error_type": "TIMEOUT"
                }
                
                st.session_state.query_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "db": db,
                    "collection": collection,
                    "success": False,
                    "error": "Timeout",
                    "execution_time": timeout
                })
                
                return error_data
                
        except requests.exceptions.ConnectionError:
            return {
                "ok": False, 
                "error": "Cannot connect to API server. Make sure the server is running on localhost:8000", 
                "status_code": 503,
                "error_type": "CONNECTION_ERROR"
            }
        except Exception as e:
            return {
                "ok": False, 
                "error": str(e), 
                "status_code": 500,
                "error_type": "CLIENT_ERROR"
            }

    def render_results(self, result: Dict[str, Any]):
        """Enhanced result rendering with better formatting"""
        st.subheader("📊 Query Results")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if result.get("ok"):
            # Success metrics
            with col1:
                st.metric("Status", "✅ Success")
            with col2:
                execution_time = result.get("execution_time", result.get("elapsed_time", 0))
                st.metric("Execution Time", f"{execution_time:.3f}s")
            with col3:
                st.metric("Results Found", result.get("total_matching", 0))
            with col4:
                st.metric("Results Returned", result.get("result_count", 0))
            
            # Show timing breakdown if available
            if result.get("query_generation_time") and result.get("db_execution_time"):
                st.info(f"⏱️ **Timing Breakdown:** Query Generation: {result['query_generation_time']:.3f}s, Database: {result['db_execution_time']:.3f}s")
            
            # Show generated MongoDB query
            st.subheader("🔍 Generated MongoDB Query")
            st.code(json.dumps(result.get("mongo_query", {}), indent=2), language="json")
            
            # Show results
            if result.get("results"):
                st.subheader("📋 Results")
                
                # Convert to DataFrame for better display
                try:
                    df = pd.DataFrame(result["results"])
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("📭 No results returned")
                except Exception as e:
                    st.warning(f"Could not display as table: {e}")
                    st.json(result["results"])
            else:
                st.info("📭 No results found")
                
        else:
            # Error display
            with col1:
                st.metric("Status", "❌ Error")
            with col2:
                execution_time = result.get("execution_time", result.get("elapsed_time", 0))
                st.metric("Time Elapsed", f"{execution_time:.3f}s")
            with col3:
                st.metric("Error Type", result.get("error_type", "Unknown"))
            with col4:
                st.metric("Status Code", result.get("status_code", "N/A"))
            
            # Show error details
            st.error(f"❌ **Error:** {result.get('error', 'Unknown error')}")
            
            # Show suggestions based on error type
            error_type = result.get("error_type", "").upper()
            if "TIMEOUT" in error_type:
                st.warning("💡 **Suggestion:** Try increasing the timeout value or simplifying your query")
            elif "CONNECTION" in error_type:
                st.warning("💡 **Suggestion:** Make sure the API server is running on localhost:8000")
            elif result.get("status_code") == 500:
                st.warning("💡 **Suggestion:** Check the server logs for more details")

    def render_query_history(self):
        """Display query history"""
        if st.session_state.query_history:
            st.subheader("📈 Query History")
            
            # Convert history to DataFrame
            df = pd.DataFrame(st.session_state.query_history)
            
            # Show summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", len(df))
            with col2:
                success_rate = (df['success'].sum() / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col3:
                avg_time = df['execution_time'].mean() if len(df) > 0 else 0
                st.metric("Avg Execution Time", f"{avg_time:.3f}s")
            
            # Show history table
            st.dataframe(df, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.experimental_rerun()

    def render_api_status(self):
        """Display API server status"""
        st.sidebar.subheader("🔧 API Server Status")
        
        with st.sidebar:
            if st.button("🔄 Refresh Status"):
                st.experimental_rerun()
        
        health = self.check_api_health()
        
        if health.get("status") == "healthy":
            st.sidebar.success("✅ API Server: Healthy")
            if "uptime" in health:
                st.sidebar.info(f"⏰ Uptime: {health['uptime']:.0f}s")
        elif health.get("status") == "degraded":
            st.sidebar.warning("⚠️ API Server: Degraded")
        else:
            st.sidebar.error(f"❌ API Server: {health.get('status', 'Unknown').title()}")
            if "error" in health:
                st.sidebar.error(f"Error: {health['error']}")
        
        # Show connection details if available
        if "mongodb_connected" in health:
            mongodb_status = "✅" if health["mongodb_connected"] else "❌"
            st.sidebar.text(f"MongoDB: {mongodb_status}")
        
        if "llm_connected" in health:
            llm_status = "✅" if health["llm_connected"] else "❌"
            st.sidebar.text(f"LLM: {llm_status}")

    def run(self):
        """Main application interface"""
        st.title("🔍 NLP to NoSQL Query Interface")
        st.markdown("Convert natural language to MongoDB queries and execute them")
        
        # API Status in sidebar
        self.render_api_status()
        
        # Main query interface
        with st.form("query_form"):
            st.subheader("📝 Query Input")
            
            # Query input
            query = st.text_area(
                "Enter your natural language query:",
                value="Find all employees",
                help="Describe what you want to find in natural language"
            )
            
            # Database settings
            col1, col2 = st.columns(2)
            with col1:
                db = st.text_input("Database", value="testdb")
                limit = st.slider("Result Limit", 1, 1000, 50)
            with col2:
                collection = st.text_input("Collection", value="testcoll")
                timeout = st.slider("Timeout (seconds)", 5, MAX_TIMEOUT, DEFAULT_TIMEOUT)
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                show_progress = st.checkbox("Show detailed progress", value=False)
                
            # Submit button
            submitted = st.form_submit_button("🚀 Execute Query", type="primary")
        
        # Process query
        if submitted and query.strip():
            with st.spinner("Processing your query..."):
                result = self.send_query(query, db, collection, limit, timeout, show_progress)
            
            if result:
                self.render_results(result)
        
        # Show query history
        if st.session_state.query_history:
            with st.expander("📈 Query History", expanded=False):
                self.render_query_history()
        
        # Footer with tips
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips for Better Queries
        - Be specific about what you want to find
        - Use clear field names (name, age, department, salary, etc.)
        - Try queries like: "Find employees earning more than 50000", "Show all engineers", "List employees older than 30"
        - Increase timeout for complex queries
        """)

if __name__ == "__main__":
    app = NLPQueryInterface()
    app.run()