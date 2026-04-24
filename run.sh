#!/bin/bash

echo "🚀 Starting ContextHub..."

# Start FastAPI backend in background
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to be ready
sleep 2

# Start Streamlit frontend
echo " Starting Frontend..."
streamlit run frontend/app.py

# When Streamlit is closed, also kill the backend
echo "🛑 Shutting down ContextHub..."
kill $BACKEND_PID
# all at once