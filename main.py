"""
FastAPI SQL Agent Backend

This backend provides endpoints to:
1. Upload Excel files and convert them to SQLite
2. Upload SQLite database files directly  
3. Query databases using natural language via the SQL agent

To run this backend:
1. Install dependencies: pip install fastapi uvicorn pandas openpyxl sqlalchemy langchain langchain-community langgraph python-multipart
2. Run: uvicorn main:app --reload --port 8000
"""

import os
import tempfile
import sqlite3
from typing import Optional, Union
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

# Import your SQL agent components
from sql_agent import create_sql_agent

app = FastAPI(title="SQL Agent API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active database sessions
active_databases = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    result: str
    sql_query: Optional[str] = None
    session_id: str

def excel_to_sqlite(excel_file_path: str, db_path: str) -> None:
    """Convert Excel file to SQLite database"""
    try:
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(excel_file_path)
        
        # Create SQLite connection
        conn = sqlite3.connect(db_path)
        
        # Convert each sheet to a table
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            
            # Clean sheet name to be a valid table name
            table_name = sheet_name.replace(' ', '_').replace('-', '_')
            table_name = ''.join(c for c in table_name if c.isalnum() or c == '_')
            
            # Write DataFrame to SQLite table
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        conn.close()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting Excel to SQLite: {str(e)}")

@app.post("/upload-excel/")
async def upload_excel(file: UploadFile = File(...)):
    """Upload Excel file and convert to SQLite database"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
    session_id = str(uuid.uuid4())
    
    try:
        # Create temporary directory for this session
        temp_dir = tempfile.mkdtemp()
        excel_path = os.path.join(temp_dir, file.filename)
        db_path = os.path.join(temp_dir, f"{session_id}.db")
        
        # Save uploaded Excel file
        with open(excel_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Convert Excel to SQLite
        excel_to_sqlite(excel_path, db_path)
        
        # Create SQL agent for this database
        agent = create_sql_agent(db_path)
        
        # Store in active sessions
        active_databases[session_id] = {
            "agent": agent,
            "db_path": db_path,
            "temp_dir": temp_dir,
            "original_filename": file.filename
        }
        
        return {
            "session_id": session_id,
            "message": f"Excel file '{file.filename}' successfully converted to SQLite database",
            "tables_created": "Check available tables using a query"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")

@app.post("/upload-database/")
async def upload_database(file: UploadFile = File(...)):
    """Upload SQLite database file directly"""
    if not file.filename.endswith(('.db', '.sqlite', '.sqlite3')):
        raise HTTPException(status_code=400, detail="File must be a SQLite database (.db, .sqlite, .sqlite3)")
    
    session_id = str(uuid.uuid4())
    
    try:
        # Create temporary directory for this session
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded database file
        with open(db_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create SQL agent for this database
        agent = create_sql_agent(db_path)
        
        # Store in active sessions
        active_databases[session_id] = {
            "agent": agent,
            "db_path": db_path,
            "temp_dir": temp_dir,
            "original_filename": file.filename
        }
        
        return {
            "session_id": session_id,
            "message": f"Database '{file.filename}' successfully uploaded",
            "tables_available": "Check available tables using a query"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing database file: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Query the database using natural language"""
    if request.session_id not in active_databases:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a database first.")
    
    try:
        agent = active_databases[request.session_id]["agent"]
        
        # Process the query using the SQL agent
        result = agent.invoke({"messages": [{"role": "user", "content": request.query}]})
        
        # Extract the final response
        final_message = result["messages"][-1]
        response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Try to extract SQL query if available
        sql_query = None
        for message in result["messages"]:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get('name') == 'sql_db_query':
                        sql_query = tool_call.get('args', {}).get('query')
                        break
        
        return QueryResponse(
            result=response_content,
            sql_query=sql_query,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Get information about a database session"""
    if session_id not in active_databases:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_databases[session_id]
    return {
        "session_id": session_id,
        "original_filename": session_data["original_filename"],
        "status": "active"
    }

@app.delete("/sessions/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up a database session and remove temporary files"""
    if session_id not in active_databases:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = active_databases[session_id]
        temp_dir = session_data["temp_dir"]
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Remove from active sessions
        del active_databases[session_id]
        
        return {"message": f"Session {session_id} cleaned up successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up session: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SQL Agent API",
        "version": "1.0.0",
        "endpoints": {
            "upload_excel": "/upload-excel/",
            "upload_database": "/upload-database/",
            "query": "/query/",
            "session_info": "/sessions/{session_id}/info",
            "cleanup": "/sessions/{session_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)