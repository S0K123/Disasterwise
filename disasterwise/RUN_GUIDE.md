# 🚨 DISASTERWISE: End-to-End Disaster Management System

This project integrates satellite image analysis, damage prediction, alert generation, and result storage into a single working system.

## 📁 Project Structure
- `backend/`: FastAPI server for processing images, running model, and managing history.
- `frontend/`: React JS application for image upload and result display.
- `src/`: Core logic for data loading, preprocessing, and model inference (original project logic).
- `configs/`: YAML configuration files.

## 🚀 Step-by-Step Run Guide

### 1. Backend Setup (FastAPI)
- **Port:** 5000
- **Dependencies:** `fastapi`, `uvicorn`, `torch`, `numpy`, `sqlite3`
- **Steps:**
  1. Open a terminal in `backend/` directory.
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Start the server:
     ```bash
     python main.py
     ```
- **API URL:** `http://127.0.0.1:5000`

### 2. Frontend Setup (React JS)
- **Port:** 3000
- **Dependencies:** `react`, `axios`
- **Steps:**
  1. Open a terminal in `frontend/` directory.
  2. Install Node.js if you haven't (LTS version).
  3. Install dependencies:
     ```bash
     npm install
     ```
  4. Start the app:
     ```bash
     npm start
     ```
- **App URL:** `http://localhost:3000`

## 🛠️ Integration Layers Added
1. **`backend/main.py`**: A FastAPI application that connects the frontend to the model and database.
2. **`backend/alert_engine.py`**: Integrated the alert computation logic into the API flow.
3. **`backend/database.py`**: SQLite database implementation to store prediction history.
4. **`backend/model_wrapper.py`**: A dedicated layer for the model's `predict` function, allowing easy integration of trained weights.

## 🧪 Testing the Full System
1. **Start Backend**: Run `python backend/main.py`.
2. **Start Frontend**: Run `npm start` in `frontend/`.
3. **Upload Images**: Select a pre-disaster image and a post-disaster image from `data/raw/xview/train/images/`.
4. **Analyze**: Click "Analyze Damage".
5. **Verify**:
   - Check if the Damage % and Alert Level appear.
   - Check the message for damage severity.
   - Check `backend/disasterwise.db` for the stored entry.

## 🛡️ Error Handling
- **Invalid Images**: Backend checks for valid file uploads.
- **Model Failure**: Try-Except blocks catch inference errors and return a 500 status.
- **Database Failure**: History fetching and storage are wrapped in error handlers.
- **API Connection**: Frontend displays clear error messages if the backend is unreachable.
