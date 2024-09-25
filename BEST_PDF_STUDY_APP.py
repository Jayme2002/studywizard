import streamlit as st
import time  # Import the time module to use sleep

# This must be the first Streamlit command
st.set_page_config(page_title="SmartExam Creator", page_icon="📝")

import argon2
from st_supabase_connection import SupabaseConnection
from stqdm import stqdm
from supabase import Client
from openai import OpenAI
import dotenv
import os
import json
from PyPDF2 import PdfReader
from PIL import Image
from io import BytesIO

from fpdf import FPDF
import base64
import jwt
from datetime import datetime, timedelta
import extra_streamlit_components as stx

__version__ = "1.1.0"

# Add these lines for debugging
print("API Key from secrets:", st.secrets.get("OPENAI_API_KEY"))
print("API Key from env:", os.getenv("OPENAI_API_KEY"))

# Authentication Utilities
def validate_email(username: str) -> bool:
    """Validates that the username contains an @ symbol, indicating it's an email."""
    return "@" in username

class Authenticator(argon2.PasswordHasher):
    def generate_pwd_hash(self, password: str):
        return self.hash(password)

    def verify_password(self, hashed_password, plain_password):
        try:
            return self.verify(hashed_password, plain_password)
        except argon2.exceptions.VerificationError:
            return False

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
    if "cookie_manager" not in st.session_state:
        st.session_state.cookie_manager = stx.CookieManager()
    return st.session_state.cookie_manager

def set_auth_cookie(username: str):
    token = create_jwt_token(username)
    cookie_manager = get_manager()
    cookie_manager.set("auth_token", token, expires_at=datetime.now() + timedelta(days=30))

def get_auth_cookie():
    cookie_manager = get_manager()
    return cookie_manager.get("auth_token")

def clear_auth_cookie():
    cookie_manager = get_manager()
    cookie_manager.delete("auth_token")

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
) -> Client:
    client = st.connection(name="supabase", type=SupabaseConnection)
    auth = Authenticator()

    # Check for existing auth token
    auth_token = get_auth_cookie()
    if auth_token:
        is_valid, username = verify_jwt_token(auth_token)
        if is_valid:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            return client

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
                                    client.table(user_tablename).insert({username_col: username, password_col: hashed_password}).execute()
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
                            response = client.table(user_tablename).select(f"{username_col}, {password_col}").eq(username_col, username).execute()
                            if response.data:
                                db_password = response.data[0][password_col]
                                if auth.verify_password(db_password, password):
                                    st.session_state["authenticated"] = True
                                    st.session_state["username"] = username
                                    set_auth_cookie(username)
                                    st.success(login_success_message)
                                    st.experimental_rerun()
                                else:
                                    st.error(login_error_message)
                            else:
                                st.error("User not found")
                        except Exception as e:
                            st.error(f"Login error: {str(e)}")

    return client

def logout():
    # Clear all session state except for the cookie manager
    for key in list(st.session_state.keys()):
        if key != "cookie_manager":
            del st.session_state[key]
    
    # Clear the authentication cookie
    clear_auth_cookie()
    
    # Ensure we set authenticated to False
    st.session_state.authenticated = False
    
    # Use st.experimental_rerun() instead of st.rerun()
    st.experimental_rerun()

# Function to reset quiz state when a new exam is uploaded
def reset_quiz_state():
    """Resets the session state for a new quiz."""
    st.session_state.answers = []
    st.session_state.feedback = []
    st.session_state.correct_answers = 0
    st.session_state.mc_test_generated = False
    st.session_state.generated_questions = []
    st.session_state.content_text = None

# Main app functions
def stream_llm_response(messages, model_params, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_params["model"] if "model" in model_params else "gpt-4o",
        messages=messages,
        temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def summarize_text(text, api_key=None):
    if api_key is None:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    print("API Key used in summarize_text:", api_key)  # Add this line for debugging
    client = OpenAI(api_key=api_key)
    prompt = (
        "Please summarize the following text to be concise and to the point:\n\n" + text
    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    summary = stream_llm_response(messages, model_params={"model": "gpt-4o-mini", "temperature": 0.3}, api_key=api_key)
    return summary

def chunk_text(text, max_tokens=3000):
    sentences = text.split('. ')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_tokens:
            chunks.append(chunk)
            chunk = sentence + ". "
        else:
            chunk += sentence + ". "
    if chunk:
        chunks.append(chunk)
    return chunks

def generate_mc_questions(content_text, api_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
    prompt = (
        "Based on the following content from a PDF, create a comprehensive multiple-choice exam. "
        "The exam should contain both multiple-choice and single-choice questions, "
        "clearly indicating which type each question is. Create 10 realistic exam questions covering the key points of the content. "
        "Provide the output in JSON format with the following structure: "
        "[{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}, ...]. "
        "Ensure the JSON is valid and properly formatted."
    )
    messages = [
        {"role": "system", "content": "You are an expert exam creator, capable of generating high-quality multiple-choice questions based on provided content."},
        {"role": "user", "content": f"Content: {content_text}\n\n{prompt}"}
    ]
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=4000
    )
    return parse_generated_questions(response.choices[0].message.content)

def parse_generated_questions(response):
    try:
        # Try to find JSON in the response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in the response")
        json_str = response[json_start:json_end]

        questions = json.loads(json_str)
        
        # Validate the structure of each question
        for q in questions:
            if not all(key in q for key in ('question', 'choices', 'correct_answer', 'explanation')):
                raise ValueError(f"Invalid question structure: {q}")
        
        return questions
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
    except ValueError as e:
        st.error(f"Validation error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    
    st.error("Response from OpenAI:")
    st.text(response)
    return None

def get_question(index, questions):
    return questions[index]

def initialize_session_state(questions):
    session_state = st.session_state
    session_state.current_question_index = 0
    session_state.quiz_data = get_question(session_state.current_question_index, questions)
    session_state.correct_answers = 0

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Generated Exam', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 10, title)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def generate_pdf(questions):
    pdf = PDF()
    pdf.add_page()

    for i, q in enumerate(questions):
        question = f"Q{i+1}: {q['question']}"

        # Avoid encoding errors by replacing problematic characters with alternatives
        question = question.replace("—", "-").encode('latin1', 'replace').decode('latin1')
        pdf.chapter_title(question)

        choices = "\n".join(q['choices'])
        choices = choices.replace("—", "-").encode('latin1', 'replace').decode('latin1')
        pdf.chapter_body(choices)

        correct_answer = f"Correct answer: {q['correct_answer']}"
        correct_answer = correct_answer.replace("—", "-").encode('latin1', 'replace').decode('latin1')
        pdf.chapter_body(correct_answer)

        explanation = f"Explanation: {q['explanation']}"
        explanation = explanation.replace("—", "-").encode('latin1', 'replace').decode('latin1')
        pdf.chapter_body(explanation)

    return pdf.output(dest="S").encode("latin1")

# Integration with the main app
def pdf_upload_app():
    st.title("Upload PDF & Generate Questions")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        content_text = extract_text_from_pdf(uploaded_file)
        st.session_state.content_text = content_text
        st.success("PDF uploaded successfully!")
        
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                questions = generate_mc_questions(content_text)
                st.session_state.generated_questions = questions
                st.session_state.mc_test_generated = True
            st.success("Questions generated successfully!")

def mc_quiz_app():
    st.title("Take the Quiz")
    if 'generated_questions' in st.session_state and st.session_state.generated_questions:
        questions = st.session_state.generated_questions
        for i, q in enumerate(questions):
            st.subheader(f"Question {i+1}")
            st.write(q['question'])
            answer = st.radio("Choose your answer:", q['choices'], key=f"q_{i}")
            if st.button("Submit", key=f"submit_{i}"):
                if answer == q['correct_answer']:
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect. The correct answer is: {q['correct_answer']}")
                st.write(f"Explanation: {q['explanation']}")
    else:
        st.warning("No questions generated yet. Please upload a PDF and generate questions first.")

def download_pdf_app():
    st.title("Download as PDF")
    if 'generated_questions' in st.session_state and st.session_state.generated_questions:
        pdf_bytes = generate_pdf(st.session_state.generated_questions)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="generated_exam.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("No questions generated yet. Please upload a PDF and generate questions first.")

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Upload PDF & Generate Questions"

    # Check for existing auth token
    auth_token = get_auth_cookie()
    if auth_token:
        is_valid, username = verify_jwt_token(auth_token)
        if is_valid:
            st.session_state.authenticated = True
            st.session_state.username = username

    if not st.session_state.authenticated:
        login_form()
    
    if st.session_state.authenticated:
        st.sidebar.title("SmartExam Creator")
        
        # Add logout button at the top of the sidebar
        if st.sidebar.button("Logout", key="logout_button"):
            logout()
        else:
            # Main app content
            dotenv.load_dotenv()
            OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

            app_mode_options = ["Upload PDF & Generate Questions", "Take the Quiz", "Download as PDF"]
            st.session_state.app_mode = st.sidebar.selectbox(
                "Choose the app mode", 
                app_mode_options, 
                index=app_mode_options.index(st.session_state.app_mode), 
                key="app_mode_select"
            )
            
            if st.session_state.app_mode == "Upload PDF & Generate Questions":
                pdf_upload_app()
            elif st.session_state.app_mode == "Take the Quiz":
                mc_quiz_app()
            elif st.session_state.app_mode == "Download as PDF":
                download_pdf_app()

    else:
        st.warning("Please log in to access the application.")

if __name__ == "__main__":
    main()
