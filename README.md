# 🤟 ISL Connect — Sign Language Assistant

An assistive communication platform for **Indian Sign Language (ISL)** recognition and translation. 

> [!WARNING]
> **Disclaimer:** This is a prototype under development. It is not a replacement for a qualified sign-language interpreter.

---

## 📌 Project Overview

**ISL Connect** is designed to bridge the communication gap between the Deaf and Hard-of-Hearing (DHH) community and the wider public in India. Indian Sign Language (ISL) is a rich visual-spatial language with its own distinct grammar, syntax, and vocabulary.

By leveraging accessible consumer hardware (standard webcams) paired with state-of-the-art computer vision and machine learning models, ISL Connect aims to deliver real-time, non-invasive sign recognition and textual translation for education, healthcare, and public services.

### Phase Status
- **Phase 1: Project Setup and Scaffolding** (Completed)
- **Phase 2: Frontend-to-Backend Connection** (Completed):
  - Streamlit frontend communicates with the FastAPI backend using Python `requests`.
  - Configurable via `API_BASE_URL` environment variable (default: `http://127.0.0.1:8000`).
  - Home page displays live backend status, health metrics, and troubleshooting instructions when offline.
- **Phase 3: Vision Pipeline & Hand Tracking** (Completed):
  - MediaPipe HandLandmarker with 21 3D coordinates per hand and OpenCV skeletal rendering.
  - Multi-hand tracking supporting both single-hand and dual-hand ISL gestures.
- **Phase 4: Dataset Preparation & Preprocessing** (Completed):
  - RealSign ISL dataset extracted (26,081 images across Training, Testing, Validation, Letters).
  - Robust ImageProcessor with aspect-ratio preserving padding, CLAHE/gamma brightness normalization, and bilateral denoising.
  - DatasetInspector detecting class balance, missing classes, and corrupted files.
- **Phase 5: Machine Learning Gesture Classification & Live Translation** (Completed):
  - Scale-, rotation-, and translation-invariant 151-dimensional LandmarkFeatureExtractor.
  - Trained Random Forest ensemble classifier on 26 ISL alphabet classes (A–Z) achieving **80.43% test accuracy** on held-out evaluation sets.
  - FastAPI recognition routes (`GET /recognition/supported-signs`, `POST /recognition/predict`, `POST /recognition/predict-landmarks`).
  - Streamlit Live Recognition UI with real-time gesture prediction hero card, top-k confidence bars, and interactive Word Builder with accessible Text-to-Speech (TTS).
- **Phase 6: Continuous Sentence Translation** (Completed):
  - `GlossBuffer`: Sliding temporal window that debounces and stabilises high-frequency frame predictions into discrete gloss events.
  - `SentenceDecoder`: Rule-based grammar lookup mapping ISL gloss sequences (SOV syntax) into natural English translations with fallback mechanics.
  - `ContinuousTranslator`: Orchestrator combining buffer + decoder for streaming and batch translation.
  - FastAPI `/continuous/*` endpoints (`/translate`, `/batch-translate`, `/session/push`, `/session/result`, `/session/reset`, `/grammar-mappings`).
  - Streamlit Sentence Translation section with preset selector, manual input, live sequence buffer, and Text-to-Speech playback.
  - 166 automated tests with 100% passing status across all modules.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) | Intuitive, interactive, and accessible web interface |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance asynchronous REST API |
| **Database** | [SQLite](https://www.sqlite.org/) + [SQLAlchemy](https://www.sqlalchemy.org/) | Lightweight local database and ORM mapping |
| **Language** | Python 3.10+ | Core programming language |
| **Config** | `python-dotenv` | Non-secret environment variable management |
| **Future CV** | OpenCV + MediaPipe | Real-time hand landmark and pose keypoint extraction |
| **Future ML** | Scikit-learn | Gesture classification and prediction pipeline |

---

## 📂 Project Structure

```text
ISL_Connect/
│
├── frontend/                     # Streamlit frontend application
│   ├── app.py                    # Main UI entrypoint with 4 navigation sections
│   └── api_client.py             # HTTP client communicating with FastAPI backend
│
├── backend/                      # FastAPI backend server
│   ├── __init__.py               # Package marker
│   ├── main.py                   # FastAPI app instance, CORS, /health route
│   ├── database.py               # SQLite connection, session factory, init_db()
│   ├── models.py                 # SQLAlchemy ORM models (SystemEvent, etc.)
│   ├── schemas.py                # Pydantic data schemas (HealthResponse, etc.)
│   └── routes/                   # Modular route handlers (for future endpoints)
│       └── __init__.py
│
├── src/                          # Core vision & ML pipeline modules (Phases 2-4)
│   ├── __init__.py
│   ├── vision/                   # Camera streaming and MediaPipe tracking
│   │   └── __init__.py
│   ├── preprocessing/            # Coordinate normalization and feature engineering
│   │   └── __init__.py
│   └── recognition/              # Classifier inference and language decoding
│       └── __init__.py
│
├── dataset/                      # Dataset directories (ignored by git except .gitkeep)
│   ├── raw/                      # Raw landmark recordings / video frames
│   └── processed/                # Normalized numpy arrays / tabular features
│
├── models/                       # Serialized trained model weights (e.g. .pkl)
├── database/                     # SQLite database file (isl_connect.db)
├── tests/                        # Automated test suite
│   ├── __init__.py
│   └── test_basic.py             # Phase 1 import, health, and database tests
│
├── .env.example                  # Template configuration file
├── .gitignore                    # Git exclusion rules (.venv, *.db, *.pkl, caches)
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Documentation and setup instructions
```

---

## 🚀 Getting Started (Windows PowerShell)

Follow these step-by-step instructions to set up and run ISL Connect locally.

### 1. Open Windows PowerShell and Navigate to the Project

```powershell
cd "c:\Users\R Sanjushree\OneDrive\Desktop\sign speak\ISL_Connect"
```

### 2. Create and Activate a Virtual Environment

```powershell
# Create a dedicated virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> **Note on PowerShell Execution Policy:**  
> If you encounter an execution policy error when activating, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

### 3. Install Dependencies

```powershell
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```powershell
# Copy template to .env
Copy-Item .env.example .env
```

---

## 💻 Running the Application

ISL Connect requires running the **Backend** and **Frontend** in two separate PowerShell terminals.

### Terminal 1: Start the FastAPI Backend

```powershell
cd "c:\Users\R Sanjushree\OneDrive\Desktop\sign speak\ISL_Connect"
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- **Backend Base URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Terminal 2: Start the Streamlit Frontend

```powershell
cd "c:\Users\R Sanjushree\OneDrive\Desktop\sign speak\ISL_Connect"
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

- **Frontend Web Application:** [http://localhost:8501](http://localhost:8501)

---

## 🧪 Running Automated Tests

To run the complete test suite (backend API tests, database operations, requests-based API client tests, and Streamlit AppTests):

```powershell
pytest -v tests/
```

---

## 🧭 Navigation Sections in Streamlit

The frontend provides seven dedicated sections accessible via the sidebar:
1. **Home:** Introduction to ISL Connect, societal purpose, architecture, backend connection health status, and 4-step workflow.
2. **Live Recognition:** Live webcam feed or uploaded image processing with 21-point MediaPipe hand landmark tracking, real-time ISL alphabet prediction hero card, top-k probabilities, and interactive Word Builder with text-to-speech.
3. **Learn ISL (A–Z):** Displays 26 verified ISL alphabet signs (A–Z) with individual status cards, model test accuracy benchmark (80.4%), and detailed per-class classification reports.
4. **Dataset & Preprocessing:** Dataset inspection metrics, missing class alerts, format distribution, side-by-side visual image preprocessing comparison, and batch preprocessing tools.
5. **Model Evaluation:** Full diagnostics suite, held-out test set accuracy, confusion matrices, per-letter classification metrics, and top misclassification pairs.
6. **Sentence Translation:** Interactive continuous ISL sentence decoder from gloss streams into natural English with sample presets and Text-to-Speech synthesis.
7. **About:** Outlines full technology stack, completed development phases, and system disclaimers.

---

## 🔮 Future Development Roadmap

- **Phase 7 — Learned Sequence Modeling:** Hidden Markov Models (HMM) or Temporal Convolutional Networks (TCN) trained on spatio-temporal landmark trajectories for learned sign transitions.
- **Phase 8 — Multi-Modal Accessibility:** Two-way translation (Speech-to-Sign animated avatar display and Sign-to-Speech audio synthesis).
- **Phase 9 — Offline Mobile Edge Deployment:** ONNX / TFLite quantization for mobile edge devices and offline tablets.
