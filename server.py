import os
import json
import sys
import re
import base64
import subprocess
import threading
import io
import time
import tempfile
import sqlite3
import uuid
import shutil
import traceback
import urllib.request
import urllib.parse
from http.server import SimpleHTTPRequestHandler, HTTPServer
import onnxruntime as ort
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np
import pypdf
try:
    import inflect
    _inflect_engine = inflect.engine()
except ImportError:
    _inflect_engine = None
try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None
# ------------------ GOOGLE DRIVE API IMPORTS & CONFIG ------------------
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

GOOGLE_DRIVE_FOLDER_ID = ""
drive_service = None

creds_path = os.path.join(os.path.dirname(__file__), "google_creds.json")
folder_path = os.path.join(os.path.dirname(__file__), "google_drive_folder.txt")

if GOOGLE_DRIVE_AVAILABLE:
    try:
        env_creds = os.environ.get("GOOGLE_CREDS_JSON")
        env_folder = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        if env_creds and env_folder:
            GOOGLE_DRIVE_FOLDER_ID = env_folder.strip()
            creds_info = json.loads(env_creds)
            creds = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            drive_service = build('drive', 'v3', credentials=creds)
            print(f"Google Drive Storage initialized via environment variables. Folder ID: {GOOGLE_DRIVE_FOLDER_ID}")
        elif os.path.exists(creds_path) and os.path.exists(folder_path):
            with open(folder_path, 'r') as f:
                GOOGLE_DRIVE_FOLDER_ID = f.read().strip()
            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            drive_service = build('drive', 'v3', credentials=creds)
            print(f"Google Drive Storage initialized via files. Folder ID: {GOOGLE_DRIVE_FOLDER_ID}")
    except Exception as e:
        print(f"Warning: Failed to initialize Google Drive: {e}")

# Global lock to ensure only one audiobook/TTS project runs at a time
synthesis_lock = threading.Lock()
cancelled_projects = set()

PORT = int(os.environ.get("PORT", 8000))

# Configure espeak-ng path
ESPEAK_PATH = r"C:\Program Files\eSpeak NG"
if os.path.exists(ESPEAK_PATH):
    os.environ["PATH"] = ESPEAK_PATH + ";" + os.environ.get("PATH", "")

# Verify model files exist
if not os.path.exists("kokoro-v1.0.onnx") or not os.path.exists("voices-v1.0.bin"):
    print("Error: Model weights not found. Please ensure kokoro-v1.0.onnx and voices-v1.0.bin are in the directory.")
    sys.exit(1)

print("Loading Kokoro model (this might take a moment)...")
try:
    _ort_options = ort.SessionOptions()
    _ort_options.intra_op_num_threads = 4
    _ort_options.inter_op_num_threads = 4
    _ort_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ort_session = ort.InferenceSession("kokoro-v1.0.onnx", _ort_options)
    kokoro = Kokoro.from_session(_ort_session, "voices-v1.0.bin")
    print("Kokoro model loaded successfully (4-thread optimized)!")
except Exception as e:
    print(f"Error loading Kokoro model: {e}")
    sys.exit(1)

# ------------------ PERSISTENT SQLITE DATABASE & STORAGE ------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(DATA_DIR, "narrator.db"))
AUDIO_DIR = os.path.join(DATA_DIR, "audiobooks")
os.makedirs(AUDIO_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'pending',
            voice TEXT DEFAULT 'af_heart',
            speed REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            project_name TEXT NOT NULL,
            chapter_number INTEGER NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            audio_url TEXT DEFAULT '',
            section_type TEXT DEFAULT 'Narrative',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()

init_db()

def db_list_projects():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

def db_get_project(name):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

def db_create_project(name, voice='af_heart', speed=1.0):
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO projects (name, voice, speed) VALUES (?, ?, ?)", (name, voice, speed))
        conn.commit()

def db_update_project(name, fields):
    if not fields:
        return
    keys = list(fields.keys())
    values = [fields[k] for k in keys]
    set_clause = ", ".join(f"{k} = ?" for k in keys)
    with get_db() as conn:
        conn.execute(f"UPDATE projects SET {set_clause} WHERE name = ?", (*values, name))
        conn.commit()

def db_delete_project(name):
    with get_db() as conn:
        conn.execute("DELETE FROM chapters WHERE project_name = ?", (name,))
        conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        conn.commit()

def db_list_chapters(project_name):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM chapters WHERE project_name = ? ORDER BY chapter_number ASC", (project_name,)).fetchall()
        return [dict(r) for r in rows]

def db_insert_chapters(project_name, chapters_list):
    with get_db() as conn:
        for idx, ch in enumerate(chapters_list):
            ch_id = ch.get('id') or str(uuid.uuid4())
            ch_num = ch.get('chapter_number', idx + 1)
            title = ch.get('title', '')
            text = ch.get('text', '')
            section_type = ch.get('section_type') or ch.get('type') or 'Narrative'
            status = ch.get('status', 'pending')
            audio_url = ch.get('audio_url', '')
            conn.execute("""
            INSERT OR REPLACE INTO chapters (id, project_name, chapter_number, title, text, status, audio_url, section_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (ch_id, project_name, ch_num, title, text, status, audio_url, section_type))
        conn.commit()

def db_add_single_chapter(project_name, title, text):
    with get_db() as conn:
        cnt_row = conn.execute("SELECT COUNT(*) FROM chapters WHERE project_name = ?", (project_name,)).fetchone()
        num = (cnt_row[0] if cnt_row else 0) + 1
        ch_id = str(uuid.uuid4())
        conn.execute("""
        INSERT INTO chapters (id, project_name, chapter_number, title, text, status)
        VALUES (?, ?, ?, ?, ?, 'pending')
        """, (ch_id, project_name, num, title, text))
        conn.commit()
        return {"id": ch_id, "project_name": project_name, "chapter_number": num, "title": title, "text": text, "status": "pending"}

def db_update_chapter(ch_id, fields):
    if not fields:
        return
    keys = list(fields.keys())
    values = [fields[k] for k in keys]
    set_clause = ", ".join(f"{k} = ?" for k in keys)
    with get_db() as conn:
        conn.execute(f"UPDATE chapters SET {set_clause} WHERE id = ?", (*values, ch_id))
        conn.commit()

def db_delete_project_chapters(project_name):
    with get_db() as conn:
        conn.execute("DELETE FROM chapters WHERE project_name = ?", (project_name,))
        conn.commit()

# Optional Supabase credentials from environment (if configured by user)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def supabase_request(path, method="GET", body=None, headers=None, content_type="application/json", timeout=10):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    url = f"{SUPABASE_URL}{path}"
    req_headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    if body is not None and content_type:
        req_headers["Content-Type"] = content_type
    if headers:
        req_headers.update(headers)
    data = None
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode('utf-8')
        else:
            data = body
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = resp.read()
            if not resp_data:
                return None
            try:
                return json.loads(resp_data.decode('utf-8'))
            except:
                return resp_data.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Supabase request notice ({method} {path}): {e}")
        return None

def upload_to_storage(project_name, filename, file_bytes):
    # 1. Always save locally for instant, zero-latency streaming & synthesis compilation
    clean_proj = re.sub(r'[\\/*?:"<>|]', "", project_name)
    proj_dir = os.path.join(AUDIO_DIR, clean_proj)
    os.makedirs(proj_dir, exist_ok=True)
    local_path = os.path.join(proj_dir, filename)
    with open(local_path, 'wb') as f:
        f.write(file_bytes)
    
    # 2. If Google Drive is configured, also upload a cloud backup copy
    if drive_service and GOOGLE_DRIVE_FOLDER_ID:
        try:
            print(f"Backing up {filename} to Google Drive...")
            query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and name = '{filename}' and trashed = false"
            response = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            files = response.get('files', [])
            media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype="audio/mpeg" if filename.endswith(".mp3") else "audio/wav")
            if files:
                drive_service.files().update(fileId=files[0]['id'], media_body=media).execute()
            else:
                drive_service.files().create(body={'name': filename, 'parents': [GOOGLE_DRIVE_FOLDER_ID]}, media_body=media, fields='id').execute()
        except Exception as e:
            print(f"Warning: Google Drive backup notice: {e}")
            
    # Return local relative public streaming URL
    return f"/storage/v1/object/public/audiobooks/{urllib.parse.quote(clean_proj)}/{urllib.parse.quote(filename)}"

def delete_from_storage(project_name, filenames):
    if not filenames:
        return
    clean_proj = re.sub(r'[\\/*?:"<>|]', "", project_name)
    proj_dir = os.path.join(AUDIO_DIR, clean_proj)
    for fn in filenames:
        fp = os.path.join(proj_dir, fn)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception as e:
                print(f"Warning deleting local file {fp}: {e}")
                
    if drive_service and GOOGLE_DRIVE_FOLDER_ID:
        try:
            for fn in filenames:
                query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and name = '{fn}' and trashed = false"
                response = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
                for f in response.get('files', []):
                    drive_service.files().delete(fileId=f['id']).execute()
        except Exception as e:
            print(f"Warning: Google Drive delete notice: {e}")

def _num_to_words(n_str):
    """Convert a numeric string to spoken words using inflect, or fall back to digits."""
    try:
        n = int(n_str.replace(',', ''))
        if _inflect_engine:
            return _inflect_engine.number_to_words(n)
        return n_str
    except:
        return n_str

def expand_numbers_and_abbreviations(text):
    """SYSTEM 4 — Expand numbers, currency, abbreviations for natural TTS pronunciation."""
    # Deduplicate repeated-phrase artifacts from PDF column layout (Starman issue)
    # "Deciding on a business Deciding on a business" → "Deciding on a business"
    text = re.sub(r'\b(.{10,80}?)\s+\1', r'\1', text)

    # Strip page-number artifacts: "4   |   Page", "17| Page", "3535 36" etc.
    text = re.sub(r'\b\d{1,3}\s*\|\s*Page\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)(\d{1,3})\s{2,}(\d{1,3})(?!\w)', '', text)  # e.g. "3535 36"

    # Currency: $2,500 → "two thousand five hundred dollars"
    def replace_currency(m):
        return _num_to_words(m.group(1)) + ' dollars'
    text = re.sub(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', replace_currency, text)

    # Percentages: 15% → "fifteen percent"
    def replace_pct(m):
        return _num_to_words(m.group(1)) + ' percent'
    text = re.sub(r'(\d+(?:\.\d+)?)%', replace_pct, text)

    # Ordinals: 1st → "first", 21st → "twenty-first"
    ordinals = {'1':'first','2':'second','3':'third','4':'fourth','5':'fifth',
                '6':'sixth','7':'seventh','8':'eighth','9':'ninth','10':'tenth',
                '11':'eleventh','12':'twelfth'}
    def replace_ordinal(m):
        n = m.group(1)
        if n in ordinals:
            return ordinals[n]
        return _num_to_words(n) + 'th'
    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', replace_ordinal, text, flags=re.IGNORECASE)

    # Common title/formal abbreviations
    abbreviations = [
        (r'\bDr\.', 'Doctor'), (r'\bMr\.', 'Mister'), (r'\bMrs\.', 'Missus'),
        (r'\bMs\.', 'Miss'), (r'\bProf\.', 'Professor'), (r'\bSt\.(?=\s+[A-Z])', 'Saint'),
        (r'\bSt\.', 'Street'), (r'\bAve\.', 'Avenue'), (r'\bBlvd\.', 'Boulevard'),
        (r'\bvs\.', 'versus'), (r'\betc\.', 'and so on'), (r'\be\.g\.', 'for example'),
        (r'\bi\.e\.', 'that is'), (r'\bMBA\b', 'M B A'), (r'\bLLC\b', 'L L C'),
        (r'\bUSA\b', 'U S A'), (r'\bU\.S\.A\.', 'U S A'), (r'\bU\.S\.', 'U S'),
        (r'\bCEO\b', 'C E O'), (r'\bCFO\b', 'C F O'), (r'\bCTO\b', 'C T O'),
    ]
    for pattern, replacement in abbreviations:
        text = re.sub(pattern, replacement, text)

    return text

def chunk_sentence(text, max_len=150):
    """Split a text chunk into sub-chunks of max_len chars at natural boundaries.
    Splits at: comma, semicolon, colon, em-dash area. Falls back to word boundary."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while len(text) > max_len:
        # Find best split point: comma/semicolon/colon before max_len
        split_pos = -1
        for delim in [', ', '; ', ': ', ' - ', ' ']:
            pos = text.rfind(delim, 0, max_len)
            if pos > max_len // 2:  # at least halfway through
                split_pos = pos + len(delim)
                break
        if split_pos == -1:
            # fallback: split at last space before max_len
            split_pos = text.rfind(' ', 0, max_len)
            if split_pos == -1:
                split_pos = max_len
        chunk = text[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        text = text[split_pos:].strip()
    if text:
        chunks.append(text)
    return chunks

# ─────────────────────────────────────────────
# SYSTEM 1 + 4: Full Text Cleaning Pipeline
# ─────────────────────────────────────────────
def clean_text_for_synthesis(text):
    """Full text normalization pipeline: ligatures, OCR fixes, prosody, numbers."""
    if not text:
        return ""

    # 1. Normalize PDF ligatures
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'ﬆ': 'st', 'œ': 'oe', 'æ': 'ae', '\u0027': "'", '\u2019': "'",
        '\u2018': "'", '\u201c': '"', '\u201d': '"', '…': '...',
        '\u2013': ',', '\u2014': ',',  # en-dash and em-dash → comma pause
    }
    for lig, rep in ligatures.items():
        text = text.replace(lig, rep)

    # 2. Fix common PDF/OCR broken word spacing (hyphenated line-wraps)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # fix hyphenated line breaks
    text = re.sub(r'\b(specif)\s+(ics)\b', r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(organiza)\s+(tion)\b', r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(adminis)\s+(tration)\b', r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(informa)\s+(tion)\b', r'\1\2', text, flags=re.IGNORECASE)

    # 3. Remove bullet and list symbols from line starts
    text = re.sub(r'^\s*[\u2022\u00b7\u25cf\u25fe\u25aa\u25ab\u2015●•]\s*', '', text, flags=re.MULTILINE)

    # 4. Prosody shaping
    text = re.sub(r'--+', ', ', text)          # double-hyphen → pause comma
    text = re.sub(r'\.\.\.', '... ', text)     # ellipsis → add trailing space for breath
    text = re.sub(r'!{2,}', '!', text)         # collapse multiple exclamation marks

    # 5. Apply number and abbreviation expansion
    text = expand_numbers_and_abbreviations(text)

    # 6. Strip trailing whitespace lines and normalize blank lines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove pure page-number lines like "17| Page" or "4   " or "3535 36"
        if re.fullmatch(r'\d{1,3}\s*\|?\s*(?:Page)?\s*', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)
    text = '\n'.join(cleaned_lines)

    return text.strip()

# ─────────────────────────────────────────────
# SYSTEM 3: Section Type Classifier
# ─────────────────────────────────────────────
def classify_section_type(lines, title=''):
    """Classify a section as Poem, Dialogue, Narrative, Activity, or Front/Back Matter."""
    if not lines:
        return 'Narrative'

    lower_title = title.lower()
    front_matter_keywords = [
        'copyright', 'dedication', 'table of contents', 'contents',
        'about the author', 'acknowledgements', 'appendix', 'references',
        'bibliography', 'index', 'epigraph', 'also by', 'other books'
    ]
    back_matter_keywords = [
        'about the author', 'about the illustrator', 'author note',
        'from the author', 'author letter'
    ]
    activity_keywords = [
        'book discussion', 'discussion questions', 'activity', 'exercise',
        'worksheet', 'questions for', 'talking about', 'what to do'
    ]
    for kw in front_matter_keywords + back_matter_keywords:
        if kw in lower_title:
            return 'Front/Back Matter'
    for kw in activity_keywords:
        if kw in lower_title:
            return 'Activity'

    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 'Narrative'

    avg_len = sum(len(l) for l in non_empty) / len(non_empty)
    total_lines = len(non_empty)

    # Dialogue detection: 25%+ of lines start with a quote or contain "said/asked/replied"
    dialogue_lines = sum(1 for l in non_empty
                         if l.strip().startswith('"') or l.strip().startswith('\u201c')
                         or re.search(r'\b(said|asked|replied|whispered|shouted|murmured|cried)\b', l))
    dialogue_ratio = dialogue_lines / max(total_lines, 1)

    # Activity/Q&A detection
    question_lines = sum(1 for l in non_empty if l.strip().endswith('?'))
    question_ratio = question_lines / max(total_lines, 1)
    if question_ratio > 0.35:
        return 'Activity'

    # Poem detection: short avg line length + no long prose sentences + multiple lines
    if avg_len < 45 and total_lines >= 4:
        # Make sure it's not just a bullet-list (bullet lists have numbering)
        numbered = sum(1 for l in non_empty if re.match(r'^\d+[.)]\s', l.strip()))
        if numbered / max(total_lines, 1) < 0.4:
            return 'Poem'

    # Dialogue-heavy narrative
    if dialogue_ratio > 0.25:
        return 'Dialogue'

    return 'Narrative'

# ─────────────────────────────────────────────
# SYSTEM 8: ACX Audio Mastering
# ─────────────────────────────────────────────
def master_to_acx(audio_array, sample_rate):
    """Normalize audio to ACX/Audible standard: -19 LUFS, peak ≤ -3 dB."""
    if pyln is None:
        return audio_array
    try:
        # Ensure float64 for pyloudnorm
        audio = audio_array.astype(np.float64)
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        # Avoid normalizing silence (very quiet audio has -inf loudness)
        if loudness < -70:
            return audio_array
        normalized = pyln.normalize.loudness(audio, loudness, -19.0)
        # Hard-limit peaks to -3 dB (0.708 linear)
        peak = np.max(np.abs(normalized))
        if peak > 0.708:
            normalized = normalized * (0.708 / peak)
        return normalized.astype(np.float32)
    except Exception as e:
        print(f"Warning: ACX mastering failed ({e}), using raw audio.")
        return audio_array

# ─────────────────────────────────────────────
# SYSTEM 5–7: Prosody Engine & Synthesis
# ─────────────────────────────────────────────
def synthesize_section(text, section_type, voice, speed, sample_rate, project_name=None):
    """Synthesize a section with pacing rules appropriate to its content type."""
    all_samples = []

    # Pacing profiles per section type
    if section_type == 'Poem':
        speed_mult = 0.84
        line_silence = np.zeros(int(sample_rate * 0.65))
        stanza_silence = np.zeros(int(sample_rate * 1.4))
        lines = text.split('\n')
        for line in lines:
            if project_name and project_name in cancelled_projects:
                print(f"Project '{project_name}' was cancelled. Aborting Poem synthesis.", flush=True)
                return []
            stripped = line.strip()
            if stripped:
                for chunk in chunk_sentence(stripped, max_len=150):
                    try:
                        samples, _ = kokoro.create(chunk, voice=voice, speed=speed * speed_mult, lang="en-us")
                        all_samples.append(samples)
                        all_samples.append(line_silence)
                    except Exception as e:
                        print(f"Poem line synthesis error: {e}")
            else:
                if all_samples:
                    all_samples.append(stanza_silence)
        if all_samples and len(all_samples) > 1:
            all_samples.pop()  # remove trailing silence
        return all_samples

    elif section_type == 'Dialogue':
        speed_mult = 0.95
        sentence_silence = np.zeros(int(sample_rate * 0.4))
        paragraph_silence = np.zeros(int(sample_rate * 1.0))
    else:  # Narrative, Activity
        speed_mult = 1.00
        sentence_silence = np.zeros(int(sample_rate * 0.3))
        paragraph_silence = np.zeros(int(sample_rate * 1.0))

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    for p_idx, paragraph in enumerate(paragraphs):
        # Merge single newlines inside paragraph
        paragraph = re.sub(r'\s*\n\s*', ' ', paragraph).strip()
        if not paragraph:
            continue
        p_samples = []

        # Split into sentences, then chunk each to <=150 chars for fast TTS
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', paragraph) if s.strip()]
        if not raw_sentences:
            raw_sentences = [paragraph]

        chunks = []
        for raw_s in raw_sentences:
            chunks.extend(chunk_sentence(raw_s, max_len=150))

        total_chunks = len(chunks)
        print(f"  Para {p_idx + 1}/{len(paragraphs)}: {total_chunks} chunks", flush=True)

        for c_idx, chunk in enumerate(chunks):
            if project_name and project_name in cancelled_projects:
                print(f"Project '{project_name}' was cancelled. Aborting synthesis.", flush=True)
                return []
            if not chunk:
                continue
            try:
                t0 = time.time()
                samples, _ = kokoro.create(chunk, voice=voice, speed=speed * speed_mult, lang="en-us")
                p_samples.append(samples)
                p_samples.append(sentence_silence)
                dur = time.time() - t0
                print(f"    Chunk [{c_idx + 1}/{total_chunks}] completed in {dur:.2f}s", flush=True)
            except Exception as e:
                print(f"  Chunk [{c_idx + 1}/{total_chunks}] error: {e}", flush=True)

        # Remove trailing silence from last chunk
        if p_samples and isinstance(p_samples[-1], np.ndarray) and len(p_samples[-1]) == len(sentence_silence):
            p_samples.pop()

        if p_samples:
            all_samples.extend(p_samples)
            all_samples.append(paragraph_silence)

    if all_samples:
        all_samples.pop()  # remove final paragraph silence

    return all_samples


# ─────────────────────────────────────────────
# Asynchronous Background Synthesis Loop
# (Systems 1–8 applied)
# ─────────────────────────────────────────────

def run_automated_synthesis(project_name, voice, speed):
    print(f"Project '{project_name}' queued for synthesis...")
    synthesis_lock.acquire()
    try:
        print(f"Starting automated synthesis for project: {project_name}")

        chapters = db_list_chapters(project_name)
        total_chapters = len(chapters)

        if total_chapters == 0:
            print(f"No chapters found for project {project_name}.")
            db_update_project(project_name, {"status": "failed"})
            return

        sample_rate = 24000
        chapter_gap_silence = np.zeros(int(sample_rate * 1.5))

        for idx, ch in enumerate(chapters):
            if project_name in cancelled_projects:
                print(f"Project '{project_name}' was cancelled. Aborting synthesis loop.", flush=True)
                return
            ch_num = ch['chapter_number']
            ch_id = ch['id']
            ch_title = ch.get('title', '')
            text = ch.get('text', '')
            section_type = ch.get('section_type', 'Narrative')

            if ch.get('status') == 'completed':
                print(f"Chapter {ch_num} is already completed. Skipping.")
                continue

            # SYSTEM 1+4: Full text cleaning pipeline
            cleaned_text = clean_text_for_synthesis(text)
            paragraphs_for_type = [l.strip() for l in cleaned_text.split('\n') if l.strip()]

            # SYSTEM 3: Classify section type if not stored
            if not section_type or section_type == 'Narrative':
                section_type = classify_section_type(paragraphs_for_type, ch_title)

            if not cleaned_text.strip():
                db_update_chapter(ch_id, {"status": "completed"})
                continue

            db_update_chapter(ch_id, {"status": "synthesizing"})

            status_text = f"Generating Chapter {ch_num} of {total_chapters} ({section_type})"
            db_update_project(project_name, {"status": status_text})

            all_samples = []

            # SYSTEM 6: Narrate the chapter title before body text
            if ch_title and ch_title.lower() not in ('preface / front matter', 'front matter'):
                try:
                    title_text = f"Chapter {ch_num}. {ch_title}."
                    title_samples, _ = kokoro.create(title_text, voice=voice, speed=speed * 0.88, lang="en-us")
                    title_silence = np.zeros(int(sample_rate * 1.8))
                    all_samples.append(title_samples)
                    all_samples.append(title_silence)
                except Exception as e:
                    print(f"Chapter title narration error: {e}")

            # SYSTEM 5–7: Synthesize body with section-appropriate pacing
            body_samples = synthesize_section(cleaned_text, section_type, voice, speed, sample_rate, project_name=project_name)
            if project_name in cancelled_projects or not body_samples:
                print(f"Project '{project_name}' was cancelled. Aborting synthesis.", flush=True)
                return
            all_samples.extend(body_samples)

            if all_samples:
                combined_samples = np.concatenate(all_samples)

                # SYSTEM 8: ACX audio mastering
                combined_samples = master_to_acx(combined_samples, sample_rate)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                    temp_path = temp_mp3.name
                    sf.write(temp_path, combined_samples, sample_rate, format='MP3')

                with open(temp_path, 'rb') as f:
                    mp3_bytes = f.read()
                os.remove(temp_path)

                print(f"Uploading Chapter {ch_num} ({section_type}) audio to Storage...")
                audio_url = upload_to_storage(project_name, f"chapter_{ch_num}.mp3", mp3_bytes)

                db_update_chapter(ch_id, {
                    "status": "completed",
                    "audio_url": audio_url
                })

        # Compile full audiobook
        print(f"Compilation phase: merging all {total_chapters} chapters for project {project_name}...")
        db_update_project(project_name, {"status": "combining"})

        chapters = db_list_chapters(project_name)

        all_audio = []
        for ch in chapters:
            url = ch.get('audio_url')
            if url:
                print(f"Reading chapter {ch['chapter_number']} for compilation: {url}")
                audio_bytes = None
                if url.startswith('/storage/v1/object/public/audiobooks/'):
                    rel_p = urllib.parse.unquote(url[len('/storage/v1/object/public/audiobooks/'):])
                    local_p = os.path.join(AUDIO_DIR, os.path.normpath(rel_p).lstrip(r'\/'))
                    if os.path.exists(local_p):
                        with open(local_p, 'rb') as f:
                            audio_bytes = f.read()
                if not audio_bytes:
                    try:
                        with urllib.request.urlopen(url, timeout=600) as response:
                            audio_bytes = response.read()
                    except Exception as err:
                        print(f"Warning: Could not fetch {url}: {err}")
                if audio_bytes:
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    all_audio.append(data)
                    all_audio.append(np.zeros(int(sr * 1.5)))

        if all_audio:
            all_audio.pop()
            combined_audio = np.concatenate(all_audio)
            combined_audio = master_to_acx(combined_audio, sample_rate)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_full:
                full_temp_path = temp_full.name
                sf.write(full_temp_path, combined_audio, sample_rate, format='MP3')

            with open(full_temp_path, 'rb') as f:
                full_bytes = f.read()
            os.remove(full_temp_path)

            print("Uploading full combined MP3 audiobook to Storage...")
            upload_to_storage(project_name, "full_audiobook.mp3", full_bytes)

            # Clean up temporary chapter audio files to save local storage
            try:
                ch_filenames = [f"chapter_{ch['chapter_number']}.mp3" for ch in chapters]
                delete_from_storage(project_name, ch_filenames)
            except Exception as clean_err:
                print(f"Warning: Could not clean up temporary chapters: {clean_err}")

            db_update_project(project_name, {"status": "completed"})
            print("Automated synthesis and compilation finished successfully!")
        else:
            db_update_project(project_name, {"status": "failed"})

    except Exception as e:
        print(f"Automated synthesis thread failed: {e}")
        try:
            db_update_project(project_name, {"status": f"failed: {str(e)}"})
        except:
            pass
    finally:
        if project_name in cancelled_projects:
            cancelled_projects.remove(project_name)
        synthesis_lock.release()
        print(f"Project '{project_name}' released from synthesis lock.")


class TTSRequestHandler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.end_headers()

    def do_POST(self):
        origin = self.headers.get('Origin', '*')
        
        if self.path == '/api/tts':
            # Single quick text generation
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode('utf-8'))
                
                text = params.get('text', '').strip()
                voice = params.get('voice', 'af_bella')
                speed = float(params.get('speed', 1.0))
                
                if not text:
                    self.send_json_response(400, {'error': 'Text is required'}, origin)
                    return
                
                samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
                
                wav_io = io.BytesIO()
                sf.write(wav_io, samples, sample_rate, format='WAV', subtype='PCM_16')
                wav_bytes = wav_io.getvalue()
                
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(len(wav_bytes)))
                self.send_header('Access-Control-Allow-Origin', origin)
                self.end_headers()
                self.wfile.write(wav_bytes)
                
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)

        elif self.path == '/api/projects':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode('utf-8'))
                name = params.get('name', '').strip()
                name = re.sub(r'\s+', ' ', name).strip()
                name = re.sub(r'[\\/*?:"<>|]', "", name)
                
                if not name:
                    self.send_json_response(400, {'error': 'Project name is required'}, origin)
                    return
                
                check_proj = db_get_project(name)
                if check_proj:
                    self.send_json_response(400, {'error': 'Project already exists'}, origin)
                    return
                
                db_create_project(name)
                self.send_json_response(200, {'success': True, 'name': name}, origin)
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)

        elif self.path == '/api/projects/parse-manuscript':
            # SYSTEMS 1, 2, 3, 9, 10 — Intelligent manuscript parser supporting PDF and TXT/Markdown
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode('utf-8'))

                filename = params.get('filename', '')
                base64_data = params.get('data', '')

                if not filename or not base64_data:
                    self.send_json_response(400, {'error': 'Filename and data are required'}, origin)
                    return

                if ',' in base64_data:
                    base64_data = base64_data.split(',', 1)[1]
                file_bytes = base64.b64decode(base64_data)

                # SYSTEM 2: Expanded Chapter Header Pattern Library
                chapter_patterns = [
                    # Standard: "Chapter 1", "Chapter I", "Chapter One", "Chapter 1: Title", "Chapter 1. Title"
                    r'^chapter\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|\w+)\s*(?:[:\-—\.].*)?$',
                    # Acts and Parts: "Act 1", "Part One: Subtitle", "Part III"
                    r'^(?:act|part|section|book)\s+(?:\d+|[ivxlcdm]+)\s*(?:[:\-—\.].*)?$',
                    # Front/back matter special words (standalone lines)
                    r'^(?:prologue|epilogue|introduction|foreword|afterword|conclusion|interlude|coda|about the author|about the illustrator|about the author/illustrator|dedication|copyright)$',
                    # Short numbered sections: "1. Title" but NOT bullet list items
                    r'^\d{1,2}\.\s+[A-Z][a-zA-Z\s]{4,50}$',
                ]

                exclude_keywords = [
                    'copyright', 'all rights reserved', 'dedication', 'table of contents',
                    'contents', 'index', 'bibliography', 'acknowledgements', 'about the author',
                    'about the illustrator', 'about the author/illustrator', 'foreword', 'appendix',
                    'references', 'epigraph', 'also by', 'other books', 'author note',
                    'from the author', 'author letter', 'printed in', 'no part of', 'isbn',
                    'published by', 'cover design',
                ]

                activity_keywords = [
                    'book discussion', 'discussion questions', 'activity', 'exercise',
                    'worksheet', 'questions for', 'talking about', "what you will need",
                    "what to do", "children's activity", "kids activity", "discussion guide"
                ]

                def is_valid_chapter_header(line_str):
                    s = line_str.strip()
                    if not s:
                        return False
                    # Must start with an uppercase letter or digit
                    if not (s[0].isupper() or s[0].isdigit()):
                        return False
                    # Match against expanded patterns
                    matched = False
                    for pat in chapter_patterns:
                        if re.match(pat, s, re.IGNORECASE):
                            matched = True
                            break
                    if not matched:
                        return False
                    # Filter: TOC lines with dot leaders
                    if re.search(r'\.{3,}', s):
                        return False
                    # Filter: lines ending with a page number suffix separated by dots or wide spaces (TOC artifact)
                    if re.search(r'(?:\.{2,}|\s{3,})\d+\s*$', s):
                        return False
                    # Filter: horizontal bullet lists "1. Product 2. Service 3. Matrix"
                    if re.search(r'\b\d+\.\s+\w', s[5:]):
                        return False
                    # Filter: lines that are too long to be a title
                    if len(s) > 90:
                        return False
                    # Filter: lines that are Q&A style starts
                    lower_line = re.sub(r'^\d+\.\s+', '', s.lower())
                    if lower_line.startswith(('what ', 'how ', 'why ', 'who ', 'which ', 'when ')):
                        return False
                    return True

                cleaned_sections = []

                if filename.lower().endswith('.pdf'):
                    # ─────────────────────────────────────────────
                    # DUAL MODE A: PDF Page-level Segmentation
                    # ─────────────────────────────────────────────
                    pdf_file = io.BytesIO(file_bytes)
                    reader = pypdf.PdfReader(pdf_file)
                    pages_data = []
                    for idx, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ''
                        # Apply ligature normalization
                        ligatures = {'ﬁ':'fi','ﬂ':'fl','ﬀ':'ff','ﬃ':'ffi','ﬄ':'ffl',
                                     'ﬆ':'st','œ':'oe','æ':'ae','\u2019':"'",'\u2018':"'",
                                     '\u201c':'"','\u201d':'"','…':'...'}
                        for lig, rep in ligatures.items():
                            page_text = page_text.replace(lig, rep)

                        lines = [l.strip() for l in page_text.split('\n') if l.strip()]

                        # Deduplicate repeated lines (column layout artifact)
                        deduped = []
                        for line in lines:
                            m = re.match(r'^(.{10,60})\s+\1\s*$', line)
                            if m:
                                deduped.append(m.group(1))
                            else:
                                deduped.append(line)
                        pages_data.append((idx + 1, deduped))

                    # Page Classifier
                    def classify_page(page_num, lines):
                        text = "\n".join(lines).strip()
                        text_lower = text.lower()
                        if len(text) < 20:
                            return 'Blank/Image'
                        if "table of contents" in text_lower or "index of chapters" in text_lower or (
                            "contents" in text_lower and re.search(r'\.{3,}\s*\d+', text_lower)
                        ) or len(re.findall(r'(?:\.{3,}|\s{4,})\d+', text_lower)) >= 3:
                            return 'TOC'
                        for kw in exclude_keywords:
                            if kw in text_lower:
                                return 'Metadata'
                        if len(text) < 150 and ('author' in text_lower or 'illustrator' in text_lower or 'maria' in text_lower):
                            return 'Metadata'
                        for kw in activity_keywords:
                            if kw in text_lower:
                                return 'Activity'
                        
                        back_matter_kws = [
                            'about the author', 'about the illustrator', 'about the author/illustrator',
                            'author note', 'from the author', 'author letter', 'simple kindness',
                            'simplekindness', 'biography', 'illustrator', 'standing bear',
                            'kristin schmidt larson', 'carmen maria', 'simple kindness, llc'
                        ]
                        for kw in back_matter_kws:
                            if kw in text_lower:
                                return 'BackMatter'

                        for line in lines:
                            if is_valid_chapter_header(line):
                                return 'ChapterHeader'
                        return 'Content'

                    classified_pages = []
                    for page_num, p_lines in pages_data:
                        ptype = classify_page(page_num, p_lines)
                        classified_pages.append({
                            'page': page_num,
                            'type': ptype,
                            'lines': p_lines,
                            'text': '\n'.join(p_lines)
                        })

                    sections = []
                    current_section_title = "Preface / Front Matter"
                    current_section_page = 1
                    current_section_lines = []

                    def flush_section():
                        nonlocal current_section_lines
                        if not current_section_lines:
                            return
                        sect_text = "\n".join(current_section_lines).strip()
                        if not sect_text:
                            current_section_lines = []
                            return

                        # Strip isolated page numbers
                        clean_sect_lines = []
                        for l in sect_text.split('\n'):
                            if re.fullmatch(r'\s*\d{1,3}\s*\|?\s*(?:Page)?\s*', l, re.IGNORECASE):
                                continue
                            clean_sect_lines.append(l)
                        sect_text = "\n".join(clean_sect_lines).strip()
                        if not sect_text:
                            current_section_lines = []
                            return

                        word_count = len(sect_text.split())
                        lower_title = current_section_title.lower()
                        text_preview_start = sect_text[:120].lower()

                        section_type = 'Narrative'
                        should_include = True

                        for kw in exclude_keywords:
                            if kw in lower_title or kw in text_preview_start:
                                should_include = False
                                section_type = 'Front/Back Matter'
                                break

                        if should_include:
                            for kw in activity_keywords:
                                if kw in lower_title or kw in text_preview_start:
                                    should_include = False
                                    section_type = 'Activity'
                                    break

                        if should_include:
                            content_lines = [l for l in sect_text.split('\n') if l.strip()]
                            section_type = classify_section_type(content_lines, current_section_title)
                            if section_type in ('Front/Back Matter', 'Activity'):
                                should_include = False

                        preview_lines = [l.strip() for l in sect_text.split('\n') if l.strip()]
                        preview = ' '.join(preview_lines[:2])[:120]

                        sections.append({
                            'title': current_section_title,
                            'text': sect_text,
                            'word_count': word_count,
                            'should_include': should_include,
                            'type': section_type,
                            'section_type': section_type,
                            'page_number': current_section_page,
                            'preview': preview,
                        })
                        current_section_lines = []

                    for p in classified_pages:
                        ptype = p['type']
                        if ptype in ('TOC', 'Metadata', 'Activity', 'BackMatter', 'Blank/Image'):
                            flush_section()
                            label_map = {
                                'TOC': 'TOC',
                                'Metadata': 'Front/Back Matter',
                                'Activity': 'Activity',
                                'BackMatter': 'Front/Back Matter',
                                'Blank/Image': 'Blank/Image'
                            }
                            title = p['lines'][0][:50] if p['lines'] else f"Page {p['page']} {ptype}"
                            sections.append({
                                'title': title,
                                'text': p['text'],
                                'word_count': len(p['text'].split()),
                                'should_include': False,
                                'type': label_map.get(ptype, 'Front/Back Matter'),
                                'section_type': label_map.get(ptype, 'Front/Back Matter'),
                                'page_number': p['page'],
                                'preview': p['text'][:120].replace('\n', ' '),
                            })
                            current_section_title = "Preface / Front Matter"
                            current_section_page = p['page'] + 1
                            continue

                        if ptype == 'ChapterHeader':
                            flush_section()
                            header = "Chapter"
                            for line in p['lines']:
                                if is_valid_chapter_header(line):
                                    header = line
                                    break
                            current_section_title = header
                            current_section_page = p['page']

                        for line in p['lines']:
                            if ptype == 'ChapterHeader' and line == current_section_title:
                                continue
                            current_section_lines.append(line)
                    flush_section()
                    cleaned_sections = sections

                else:
                    # ─────────────────────────────────────────────
                    # DUAL MODE B: TXT/Markdown Line-level Parser
                    # ─────────────────────────────────────────────
                    text = file_bytes.decode('utf-8', errors='ignore')
                    lines = [l.strip() for l in text.split('\n')]
                    raw_sections = []

                    current_title = "Preface / Front Matter"
                    current_lines = []

                    for idx, line in enumerate(lines):
                        if is_valid_chapter_header(line):
                            if current_lines:
                                raw_sections.append({
                                    'title': current_title,
                                    'lines': current_lines,
                                    'text': '\n'.join(current_lines),
                                    'page': 1
                                })
                            current_title = line
                            current_lines = []
                        else:
                            if line:
                                current_lines.append(line)

                    if current_lines:
                        raw_sections.append({
                            'title': current_title,
                            'lines': current_lines,
                            'text': '\n'.join(current_lines),
                            'page': 1
                        })

                    for s in raw_sections:
                        stype = classify_section_type(s['lines'], s['title'])
                        should_include = stype not in ('Front/Back Matter', 'Activity')

                        clean_lines = [l for l in s['lines'] if l.strip() != s['title'].strip()]
                        clean_text = "\n".join(clean_lines).strip()
                        word_count = len(clean_text.split())

                        if not clean_text:
                            continue

                        preview = clean_text.replace('\n', ' ')[:60]
                        cleaned_sections.append({
                            'title': s['title'],
                            'text': clean_text,
                            'word_count': word_count,
                            'should_include': should_include,
                            'type': stype,
                            'section_type': stype,
                            'page_number': s['page'],
                            'preview': preview
                        })

                # ─────────────────────────────────────────────
                # Post-Processing: Short Section Merging (Common)
                # ─────────────────────────────────────────────
                final_sections = []
                for sec in cleaned_sections:
                    if sec['word_count'] >= 30 or sec['type'] in ('Poem', 'Dialogue'):
                        final_sections.append(sec)
                    else:
                        if final_sections:
                            final_sections[-1]['text'] += '\n\n' + sec['text']
                            final_sections[-1]['word_count'] = len(final_sections[-1]['text'].split())

                # Re-classify poem sections that were merged into front matter
                for sec in final_sections:
                    if sec['title'] == 'Preface / Front Matter' or sec['type'] == 'Narrative':
                        content_lines = [l.strip() for l in sec['text'].split('\n') if l.strip()]
                        reclassified = classify_section_type(content_lines, sec['title'])
                        if reclassified == 'Poem':
                            sec['type'] = 'Poem'
                            sec['section_type'] = 'Poem'
                            sec['should_include'] = True
                            sec['title'] = 'Opening Poem / Verse'

                self.send_json_response(200, final_sections, origin)

            except Exception as e:
                import traceback
                print(f"Parse manuscript error: {traceback.format_exc()}")
                self.send_json_response(500, {'error': str(e)}, origin)


        
        elif self.path == '/api/projects/import-chapters':
            # Bulk import chapters and trigger synthesis
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode('utf-8'))
                
                project = params.get('project', '').strip()
                chapters = params.get('chapters', [])
                voice = params.get('voice', 'af_heart')
                speed = float(params.get('speed', 1.0))
                
                if not project or not chapters:
                    self.send_json_response(400, {'error': 'Project and chapters list are required'}, origin)
                    return
                
                db_delete_project_chapters(project)
                
                db_chapters = []
                for idx, ch in enumerate(chapters):
                    db_chapters.append({
                        "id": str(uuid.uuid4()),
                        "project_name": project,
                        "chapter_number": idx + 1,
                        "title": ch.get('title', ''),
                        "text": ch.get('text', ''),
                        "section_type": ch.get('section_type') or ch.get('type') or 'Narrative',
                        "status": "pending"
                    })
                
                db_insert_chapters(project, db_chapters)
                db_update_project(project, {
                    "voice": voice,
                    "speed": speed,
                    "status": "pending"
                })
                
                # Start automated sequential synthesis loop on a background thread!
                thread = threading.Thread(target=run_automated_synthesis, args=(project, voice, speed))
                thread.daemon = True
                thread.start()
                
                self.send_json_response(200, {'success': True, 'message': 'Automated audiobook synthesis started.'}, origin)
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)

        elif self.path == '/api/projects/chapters':
            # Add manual single chapter to project
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode('utf-8'))
                
                project = params.get('project', '').strip()
                title = params.get('title', '').strip()
                text = params.get('text', '').strip()
                
                if not project or not title or not text:
                    self.send_json_response(400, {'error': 'Project, title, and text are required'}, origin)
                    return
                
                new_ch = db_add_single_chapter(project, title, text)
                self.send_json_response(200, new_ch, origin)
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)

        elif self.path == '/api/projects/delete':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode('utf-8'))
                project = params.get('project', '').strip()
                
                if not project:
                    self.send_json_response(400, {'error': 'Project name is required'}, origin)
                    return
                
                print(f"Deleting project {project} records and audio files...")
                cancelled_projects.add(project)
                
                try:
                    clean_proj = re.sub(r'[\\/*?:"<>|]', "", project)
                    proj_dir = os.path.join(AUDIO_DIR, clean_proj)
                    if os.path.exists(proj_dir):
                        shutil.rmtree(proj_dir, ignore_errors=True)
                except Exception as storage_err:
                    print(f"Warning: Could not clean up storage files: {storage_err}")

                db_delete_project(project)
                self.send_json_response(200, {'success': True, 'message': f'Project {project} deleted successfully.'}, origin)
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        origin = self.headers.get('Origin', '*')
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        
        # Audio Streaming & Static File Serving
        if path.startswith('/storage/v1/object/public/audiobooks/') or path.startswith('/api/audio/'):
            prefix = '/storage/v1/object/public/audiobooks/' if path.startswith('/storage/v1/object/public/audiobooks/') else '/api/audio/'
            rel_path = urllib.parse.unquote(path[len(prefix):])
            clean_rel = os.path.normpath(rel_path).lstrip(r'\/')
            abs_path = os.path.abspath(os.path.join(AUDIO_DIR, clean_rel))
            
            if abs_path.startswith(os.path.abspath(AUDIO_DIR)) and os.path.isfile(abs_path):
                file_size = os.path.getsize(abs_path)
                content_type = "audio/mpeg" if abs_path.lower().endswith(".mp3") else "audio/wav"
                
                range_header = self.headers.get('Range')
                if range_header:
                    range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
                    if range_match:
                        start = int(range_match.group(1))
                        end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                        length = end - start + 1
                        self.send_response(206)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                        self.send_header('Content-Length', str(length))
                        self.send_header('Accept-Ranges', 'bytes')
                        self.send_header('Access-Control-Allow-Origin', origin)
                        self.end_headers()
                        with open(abs_path, 'rb') as f:
                            f.seek(start)
                            self.wfile.write(f.read(length))
                        return

                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(file_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', origin)
                self.end_headers()
                with open(abs_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        if path == '/api/projects':
            try:
                projs = db_list_projects()
                names = [p['name'] for p in projs]
                self.send_json_response(200, names, origin)
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)

        elif path == '/api/projects/chapters':
            try:
                project = query.get('project', [''])[0]
                chapters = db_list_chapters(project)
                
                ui_chapters = []
                for ch in chapters:
                    ui_chapters.append({
                        'id': ch['id'],
                        'title': f"{ch['chapter_number']}. {ch['title']}",
                        'text': ch['text'],
                        'status': ch['status'],
                        'audio_path': ch.get('audio_url', '')
                    })
                self.send_json_response(200, ui_chapters, origin)
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)

        elif path == '/api/projects/status':
            try:
                project = query.get('project', [''])[0]
                proj = db_get_project(project)
                if proj:
                    self.send_json_response(200, proj, origin)
                else:
                    self.send_json_response(404, {'error': 'Project not found'}, origin)
            except Exception as e:
                self.send_json_response(500, {'error': str(e)}, origin)

        elif path == '/api/system-stats':
            free_mb = 1024
            total_mb = 8192
            mem_percent = 25
            try:
                # 1. Linux container: inspect /proc/meminfo
                if os.path.exists('/proc/meminfo'):
                    mem_info = {}
                    with open('/proc/meminfo', 'r') as mf:
                        for line in mf:
                            parts = line.split(':')
                            if len(parts) == 2:
                                mem_info[parts[0].strip()] = int(parts[1].strip().split()[0])
                    if 'MemTotal' in mem_info:
                        total_mb = mem_info['MemTotal'] // 1024
                        avail = mem_info.get('MemAvailable', mem_info.get('MemFree', 0))
                        free_mb = avail // 1024
                        mem_percent = int(((total_mb - free_mb) / total_mb) * 100) if total_mb > 0 else 25
                elif sys.platform == 'win32':
                    out = subprocess.check_output("wmic os get freephysicalmemory, totalvisiblememorysize /format:list", shell=True).decode('utf-8')
                    lines = [l.strip() for l in out.split('\n') if l.strip()]
                    stats = {}
                    for l in lines:
                        if '=' in l:
                            k, v = l.split('=', 1)
                            stats[k.strip()] = int(v.strip())
                    if 'FreePhysicalMemory' in stats and 'TotalVisibleMemorySize' in stats:
                        free_mb = int(stats['FreePhysicalMemory'] / 1024)
                        total_mb = int(stats['TotalVisibleMemorySize'] / 1024)
                        mem_percent = int((1 - (free_mb / total_mb)) * 100) if total_mb > 0 else 25
            except Exception as e:
                pass

            self.send_json_response(200, {
                'free_ram_mb': free_mb,
                'total_ram_mb': total_mb,
                'ram_usage_percent': mem_percent,
                'cpu_status': 'Optimized',
                'model_loaded': 'Kokoro-82M ONNX'
            }, origin)

        elif path == '/favicon.ico':
            self.send_response(200)
            self.send_header('Content-Type', 'image/x-icon')
            self.end_headers()
        else:
            if path == '/' or path == '/index.html':
                self.path = '/index.html'
            return super().do_GET()

    def send_json_response(self, status, data, origin):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', origin)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

def run():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, TTSRequestHandler)
    print(f"Web interface running at: http://localhost:{PORT}")
    
    # Auto-reset any unfinished projects from prior unexpected container exit
    try:
        with get_db() as conn:
            conn.execute("UPDATE projects SET status = 'failed' WHERE status LIKE 'Generating%' OR status = 'combining'")
            conn.commit()
    except Exception as e:
        print("Notice on project cleanup:", e)
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    run()
