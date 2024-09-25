import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import random
import argon2
from st_supabase_connection import SupabaseConnection
from supabase import Client
import jwt
from datetime import datetime, timedelta
import extra_streamlit_components as stx
from BEST_PDF_STUDY_APP import Authenticator, set_auth_cookie, get_auth_cookie, verify_jwt_token

# Page config should be the very first Streamlit command
st.set_page_config(
    page_title="Master Your Studies - Create Your Exam",
    page_icon="🧠",  
    layout="centered",
    initial_sidebar_state="expanded",
)

dotenv.load_dotenv()

openai_models = [
    "gpt-4o-mini", 
    "gpt-4-turbo", 
    "gpt-3.5-turbo-16k", 
]

# Authentication Utilities
def validate_email(username: str) -> bool:
    """Validates that the username contains an @ symbol, indicating it's an email."""
    return "@" in username

def create_jwt_token(username: str, expiration_days: int = 30) -> str:
    expiration = datetime.utcnow() + timedelta(days=expiration_days)
    payload = {
        "sub": username,
        "exp": expiration
    }
    return jwt.encode(payload, st.secrets["JWT_SECRET"], algorithm="HS256")

def verify_jwt_token(token: str) -> tuple[bool, str | None]:
    try:
        payload = jwt.decode(token, st.secrets["JWT_SECRET"], algorithms=["HS256"])
        return True, payload["sub"]
    except jwt.ExpiredSignatureError:
        return False, None
    except jwt.InvalidTokenError:
        return False, None

def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

def set_auth_cookie(username: str):
    token = create_jwt_token(username)
    cookie_manager.set("auth_token", token, expires_at=datetime.now() + timedelta(days=30))

def get_auth_cookie():
    return cookie_manager.get("auth_token")

def clear_auth_cookie():
    cookie_manager.delete("auth_token")

# Supabase Login Functionality
def login_form(
    *,
    title: str = "Authentication",
    user_tablename: str = "users",
    username_col: str = "username",
    password_col: str = "password",
    create_title: str = "Create new account :baby: ",
    login_title: str = "Login to existing account :prince: ",
    allow_guest: bool = False,
    allow_create: bool = True,
    create_username_label: str = "Create an email username",
    create_username_placeholder: str = None,
    create_username_help: str = None,
    create_password_label: str = "Create a password",
    create_password_placeholder: str = None,
    create_password_help: str = "Password cannot be recovered if lost",
    create_submit_label: str = "Create account",
    create_success_message: str = "Account created and logged-in :tada:",
    login_username_label: str = "Enter your email username",
    login_username_placeholder: str = None,
    login_username_help: str = None,
    login_password_label: str = "Enter your password",
    login_password_placeholder: str = None,
    login_password_help: str = None,
    login_submit_label: str = "Login",
    login_success_message: str = "Login succeeded :tada:",
    login_error_message: str = "Wrong username/password :x: ",
    email_constraint_fail_message: str = "Please sign up with a valid email address (must contain @).",
) -> None:
    auth = Authenticator()

    # Check for existing auth token
    auth_token = get_auth_cookie()
    if auth_token:
        is_valid, username = verify_jwt_token(auth_token)
        if is_valid:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            return

    if not st.session_state.get("authenticated", False):
        with st.expander(title, expanded=True):
            if allow_create:
                create_tab, login_tab = st.tabs([create_title, login_title])
            else:
                login_tab = st.container()

            if allow_create:
                with create_tab:
                    with st.form(key="create"):
                        username = st.text_input(label=create_username_label, placeholder=create_username_placeholder, help=create_username_help)
                        password = st.text_input(label=create_password_label, placeholder=create_password_placeholder, help=create_password_help, type="password")
                        if st.form_submit_button(label=create_submit_label, type="primary"):
                            if "@" not in username:
                                st.error(email_constraint_fail_message)
                            else:
                                hashed_password = auth.generate_pwd_hash(password)
                                try:
                                    # Here you would typically insert the new user into your database
                                    # For now, we'll just set the session state
                                    st.session_state["authenticated"] = True
                                    st.session_state["username"] = username
                                    set_auth_cookie(username)
                                    st.success(create_success_message)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

            with login_tab:
                with st.form(key="login"):
                    username = st.text_input(label=login_username_label, placeholder=login_username_placeholder, help=login_username_help)
                    password = st.text_input(label=login_password_label, placeholder=login_password_placeholder, help=login_password_help, type="password")
                    if st.form_submit_button(label=login_submit_label, type="primary"):
                        try:
                            # Here you would typically verify the user credentials against your database
                            # For now, we'll just set the session state if the username is not empty
                            if username:
                                st.session_state["authenticated"] = True
                                st.session_state["username"] = username
                                set_auth_cookie(username)
                                st.success(login_success_message)
                                st.rerun()
                            else:
                                st.error(login_error_message)
                        except Exception as e:
                            st.error(f"Login error: {str(e)}")

# Function to query and stream the response from the LLM
def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""
    if model_type == "openai":
        client = OpenAI(api_key=api_key)
        for chunk in client.chat.completions.create(
            model=model_params["model"] if "model" in model_params else "gpt-4o",
            messages=st.session_state.messages,
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
            stream=True,
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def query_image(base64_image: str, question: str) -> str:
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    
    answer = response.choices[0].message.content
    
    # Store the Q&A in chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"question": question, "answer": answer})
    
    return answer

def main():
    login_form()

    if st.session_state.get("authenticated", False):
        st.sidebar.title("Chat with Images")
        
        # Add logout button
        if st.sidebar.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            clear_auth_cookie()
            st.experimental_rerun()

        # Main app content
        st.title("🖼️ Chat with Images")
        st.write("Upload an image and ask questions about it!")

        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Encode image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode()

            # User input
            user_question = st.text_input("Ask a question about the image:")
            if user_question:
                with st.spinner("Analyzing the image..."):
                    response = query_image(encoded_image, user_question)
                st.write("Answer:", response)

        # Display chat history
        if "chat_history" in st.session_state:
            st.subheader("Chat History")
            for entry in st.session_state.chat_history:
                st.write(f"Q: {entry['question']}")
                st.write(f"A: {entry['answer']}")

    else:
        st.warning("Please log in to access the application.")

if __name__ == "__main__":
    main()
