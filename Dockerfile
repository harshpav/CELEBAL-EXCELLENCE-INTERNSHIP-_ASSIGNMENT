# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + serve built frontend ────────────────────────────
FROM python:3.11-slim

# System deps for chromadb + matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Hugging Face Spaces runs on port 7860
ENV PORT=7860

# Create uploads directory
RUN mkdir -p backend/uploads

# Expose port
EXPOSE 7860

# Start gunicorn
CMD ["sh", "-c", "cd backend && gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --timeout 120"]
