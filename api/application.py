import os
import sys
import uuid
import shutil
import json
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
import jwt
import gradio as gr
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.llm.graph import ask

JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

USERS_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

app = FastAPI(
    title="A/B Test Assistant",
    description="LLM-powered A/B testing assistant",
    version="1.0.0",
)



bearer_scheme = HTTPBearer(auto_error=False)


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


def _load_users() -> dict:
    if not os.path.exists(USERS_DB_PATH):
        return {}
    with open(USERS_DB_PATH, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_DB_PATH, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def authenticate_or_create_user(username: str, password: str) -> str:
    username = username.strip().lower()
    users = _load_users()

    if username in users:
        record = users[username]
        given_hash = _hash_password(password, record["salt"])
        if not hmac.compare_digest(given_hash, record["password_hash"]):
            raise ValueError("Incorrect password.")
        return record["user_id"]

    salt = secrets.token_hex(16)
    user_id = str(uuid.uuid4())
    users[username] = {
        "user_id": user_id,
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    return user_id


def issue_token(username: str, password: str) -> TokenResponse:
    user_id = authenticate_or_create_user(username, password)
    payload = {
        "sub": user_id,
        "username": username.strip().lower(),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return TokenResponse(access_token=token, user_id=user_id)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return decode_token(creds.credentials)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/auth/token", response_model=TokenResponse)
def login(req: TokenRequest):
    
    if not req.username.strip():
        raise HTTPException(status_code=400, detail="username is required")
    if not req.password:
        raise HTTPException(status_code=400, detail="password is required")
    try:
        return issue_token(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return {"user_id": user["sub"], "username": user.get("username")}

def do_login(username: str, password: str):
    if not username or not username.strip():
        return None, None, "⚠️ Enter a username first."
    if not password:
        return None, None, "⚠️ Enter a password first."

    try:
        token_resp = issue_token(username, password)
    except ValueError as e:
        return None, None, f"❌ {e}"

    user_id = token_resp.user_id
    thread_id = str(uuid.uuid4())

    status_msg = f"✅ Logged in as **{username}** (user_id: `{user_id[:8]}...`)"
    return user_id, thread_id, status_msg

@app.on_event("shutdown")
def cleanup_uploads():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def upload_csv(file, user_id):
    if file is None:
        return None, "No file uploaded."
    if not user_id:
        return None, "⚠️ Please log in before uploading a CSV."

    user_dir = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    filename = os.path.basename(file)
    dest_path = os.path.join(user_dir, filename)
    shutil.copy(file, dest_path)

    return dest_path, f"Uploaded `{filename}` — ready for analysis."

def chat(message, history, user_id, thread_id, csv_path):
    if not user_id or not thread_id:
        return "Please log in first (enter a username above and click Login)."

    return ask(
        user_id=user_id,
        user_message=message,
        thread_id=thread_id,
        csv_path=csv_path,
    )


with gr.Blocks(title="A/B Test Assistant") as demo:
    gr.Markdown("# A/B Test Assistant")

    user_id_state = gr.State(None)
    thread_id_state = gr.State(None)
    csv_path_state = gr.State(None)

    with gr.Group():
        with gr.Row():
            username_box = gr.Textbox(label="Username", placeholder="e.g. jane.doe", scale=2)
            password_box = gr.Textbox(label="Password", type="password", scale=2)
            login_btn = gr.Button("Login / Sign up", scale=1, variant="primary")
        login_status = gr.Markdown("🔒 Not logged in.")

    with gr.Group(visible=False) as upload_group:
        with gr.Row():
            csv_file = gr.File(label="Upload CSV for A/B testing", file_types=[".csv"])
        upload_status = gr.Markdown("")
        csv_preview = gr.Dataframe(label="Preview", visible=False, interactive=False)

    with gr.Group(visible=False) as chat_group:
        chat_ui = gr.ChatInterface(
            fn=chat,
            additional_inputs=[user_id_state, thread_id_state, csv_path_state],
            type="messages",
            title=None,
            description="Ask questions, run A/B tests on your uploaded CSV, or retrieve past results.",
        )

    def handle_login(username, password):
        user_id, thread_id, status = do_login(username, password)
        ok = user_id is not None
        status_msg = f"✅ {status}" if ok else f"❌ {status}"
        return (
            user_id,
            thread_id,
            status_msg,
            gr.update(visible=ok),
            gr.update(visible=ok),
        )

    login_btn.click(
        fn=handle_login,
        inputs=[username_box, password_box],
        outputs=[user_id_state, thread_id_state, login_status, upload_group, chat_group],
    )
    password_box.submit(
        fn=handle_login,
        inputs=[username_box, password_box],
        outputs=[user_id_state, thread_id_state, login_status, upload_group, chat_group],
    )

    def handle_upload(file, user_id):
        if file is None:
            return None, "No file uploaded.", gr.update(visible=False)
        try:
            path, status = upload_csv(file, user_id)
            df_preview = pd.read_csv(path).head(10)
            return path, f"✅ {status}", gr.update(value=df_preview, visible=True)
        except Exception as e:
            return None, f"❌ Failed to process CSV: {e}", gr.update(visible=False)

    csv_file.upload(
        fn=handle_upload,
        inputs=[csv_file, user_id_state],
        outputs=[csv_path_state, upload_status, csv_preview],
    )

app = gr.mount_gradio_app(app, demo, path="/ui")