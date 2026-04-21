## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` (comes with Python)
- Git

---

## 🐍 Setting Up a Virtual Environment

It is strongly recommended to use a virtual environment to isolate project dependencies.

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd genai-rag-agenticai
```

### Step 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv

# Windows
python -m venv venv
```

### Step 3 — Activate the virtual environment

```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

> You should see `(venv)` appear at the start of your terminal prompt, indicating the environment is active.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Deactivate when done

```bash
deactivate
```

---

## 🔑 Environment Variables (API Keys)

Create a `.env` file in the project root (same level as `requirements.txt`) and add your API keys:

```env
GOOGLE_API_KEY=ABCXXXDBE
TAVILY_API_KEY=tvly-xyabced
```

> ⚠️ **Never commit your `.env` file to Git.** Make sure `.env` is listed in your `.gitignore`.

### How to get API keys

| Key | Source |
|-----|--------|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `TAVILY_API_KEY` | [Tavily](https://app.tavily.com/) |

---

## 📦 System-level Dependencies

Some packages require system binaries to be installed separately:

```bash
# macOS (using Homebrew)
brew install tesseract   # For OCR
brew install poppler     # For PDF rendering

# Ubuntu / Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows
# Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 🗂️ Project Structure

```
genai-rag-agenticai/
├── .env                  # API keys (do NOT commit)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── ...                   # Your notebooks and scripts
```

---

## 🛠️ Troubleshooting

- **`ModuleNotFoundError`** — Make sure the virtual environment is activated and dependencies are installed.
- **`GOOGLE_API_KEY` not found** — Ensure `.env` exists in the project root and `python-dotenv` is installed.
- **Tesseract errors** — Install the Tesseract binary for your OS (see System Dependencies above).
