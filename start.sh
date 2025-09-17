#!/bin/bash
# Startup script for production deployment

echo "Starting AI Agent Scraper Backend..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set"
fi

if [ -z "$DATAFORSEO_LOGIN" ]; then
    echo "Warning: DATAFORSEO_LOGIN not set"
fi

# Install dependencies if needed (for container deployments)
if [ "$INSTALL_DEPS" = "true" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Run database migrations if needed (future enhancement)
# python migrate.py

# Start the application
echo "Starting uvicorn server..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
