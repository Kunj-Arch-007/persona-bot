import re
import streamlit as st
from typing import TypedDict, Literal, Optional, Dict
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
import time
import os
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Personal Details Collection Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define the state structure
class PersonalDetailsState(TypedDict):
    current_question: int  # 0-7 (0=start, 1-7=questions, >7=complete)
    questions: list[str]
    answers: list[str]
    conversation_history: list[dict]
    is_complete: bool
    summary: str
    retries: Dict[int, int]
    last_validation_error: Optional[str]

# Initialize LLM
@st.cache_resource
def load_llm():
    """Load and cache the LLM model"""
    try:
        return OllamaLLM(model="llama3.1:8b-instruct-q4_K_M")
    except Exception as e:
    # Defer showing error until runtime UI; return None so callers can fallback
        return None

# Define the 7 questions in order
QUESTIONS = [
    "Hi! I'm here to collect your personal details. Let's start - what's your full name?",
    "Great! Now, how old are you?",
    "Perfect! What's your gender?",
    "Excellent! Could you please provide your email address?",
    "Wonderful! What's your mobile number?",
    "Great! Which country are you from?",
    "Finally, what's your profession or occupation?"
]

QUESTION_LABELS = [
    "Full Name",
    "Age", 
    "Gender",
    "Email Address",
    "Mobile Number",
    "Country",
    "Profession"
]

# LangGraph Node Functions
def start_conversation(state: PersonalDetailsState) -> PersonalDetailsState:
    """Initialize the conversation"""
    return {
        **state,
        "current_question": 1,
        "questions": QUESTIONS,
        "answers": [],
        "conversation_history": [],
        "is_complete": False,
        "summary": ""
    }

# Remove unused functions
def ask_question(state: PersonalDetailsState) -> PersonalDetailsState:
    """Ask the current question"""
    if state["current_question"] > 7:
        return state  # Don't ask more questions
        
    question_index = state["current_question"] - 1
    question = QUESTIONS[question_index]
    
    # Add to conversation history
    new_history = state["conversation_history"].copy()
    new_history.append({
        "role": "bot",
        "content": question,
        "question_number": state["current_question"]
    })
    
    return {
        **state,
        "conversation_history": new_history
    }


def add_history(state: PersonalDetailsState, role: str, content: str, qnum: int) -> PersonalDetailsState:
    new_history = state["conversation_history"].copy()
    new_history.append({"role": role, "content": content, "question_number": qnum})
    return {**state, "conversation_history": new_history}


def llm_normalize(field: str, value: str) -> str:
    """Attempt to normalize an answer using the LLM; fall back to original value on failure."""
    llm = load_llm()
    if not llm:
        return value.strip()

    prompt = f"Normalize the following user answer for the field '{field}'. Return only the normalized value, nothing else.\n\nInput: {value}\n\nNormalized:"
    try:
        out = llm.invoke(prompt)
        if isinstance(out, str) and out.strip():
            return out.strip()
    except Exception:
        pass
    return value.strip()


def validate_name(value: str) -> tuple[bool, str, Optional[str]]:
    v = value.strip()
    if len(v) == 0:
        return False, v, "Name cannot be empty. Please provide your full name."
    # require at least two words (first + last) containing alphabetic characters
    parts = [p for p in re.split(r"\s+", v) if p]
    if len(parts) < 2:
        return False, v, "Please enter your full name (first and last name)."
    # basic check: letters and common name characters
    if not all(re.search(r"[A-Za-z]", p) for p in parts):
        return False, v, "Name should contain alphabetic characters."
    normalized = llm_normalize("Full Name", v)
    return True, normalized, None


def validate_age(value: str) -> tuple[bool, str, Optional[str]]:
    v = value.strip()
    # direct int parse
    try:
        n = int(re.sub(r"[^0-9]", "", v))
        if n <= 0 or n > 120:
            return False, v, "Please provide a valid age between 1 and 120."
        return True, str(n), None
    except Exception:
        return False, v, "Please enter your age as a number (e.g., 29)."


def validate_gender(value: str) -> tuple[bool, str, Optional[str]]:
    v = value.strip().lower()
    if not v:
        return False, v, "Please specify your gender (e.g., Male, Female, Non-binary, Prefer not to say)."
    mapping = {
        "m": "Male", "male": "Male",
        "f": "Female", "female": "Female",
        "non-binary": "Non-binary", "nonbinary": "Non-binary",
        "nb": "Non-binary", "prefer not to say": "Prefer not to say",
        "pn": "Prefer not to say"
    }
    norm = mapping.get(v, None)
    if not norm:
        # fallback to normalized string from LLM
        norm = llm_normalize("Gender", value)
    return True, norm, None


def validate_email(value: str) -> tuple[bool, str, Optional[str]]:
    v = value.strip()
    if not v:
        return False, v, "Email cannot be empty."
    # simple regex
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
        return False, v, "Please enter a valid email address (e.g., user@example.com)."
    return True, v, None


def validate_mobile(value: str) -> tuple[bool, str, Optional[str]]:
    v = re.sub(r"[^0-9+]", "", value.strip())
    # allow leading + and digits
    digits = re.sub(r"[^0-9]", "", v)
    if len(digits) < 7 or len(digits) > 15:
        return False, v, "Please provide a valid mobile number (7 to 15 digits, include country code if possible)."
    # basic normalization: keep + if present
    if v.startswith("+"):
        norm = "+" + digits
    else:
        norm = digits
    return True, norm, None


def validate_country(value: str) -> tuple[bool, str, Optional[str]]:
    v = value.strip()
    if not v:
        return False, v, "Country cannot be empty."
    # basic alpha check (allow spaces)
    if not re.search(r"[A-Za-z]", v):
        return False, v, "Please enter a valid country name."
    norm = llm_normalize("Country", v)
    return True, norm, None


def validate_profession(value: str) -> tuple[bool, str, Optional[str]]:
    v = value.strip()
    if not v:
        return False, v, "Profession cannot be empty. Please provide your occupation."
    norm = llm_normalize("Profession", v)
    return True, norm, None

def generate_summary(state: PersonalDetailsState) -> PersonalDetailsState:
    """Generate final summary using LLM"""
    llm = load_llm()
    # Build a clear prompt to create a 7-line professional introduction (not a bulleted list)
    qa_pairs = []
    for i, (label, answer) in enumerate(zip(QUESTION_LABELS, state["answers"])):
        qa_pairs.append(f"{label}: {answer}")

    qa_text = "\n".join(qa_pairs)

    if not llm:
        # Fallback: create a simple local summary (brief, one-liner per field)
        lines = [f"{label}: {answer}" for label, answer in zip(QUESTION_LABELS, state.get("answers", []))]
        # Create up to 7 short lines joined as paragraph
        summary = "\n".join(lines[:7])
    else:
        prompt = f"""Using the following collected personal details, write a concise, professional introduction of about 7 lines (each line should be a short sentence). Do not add sections or headings — just the introduction text.

{qa_text}

Write exactly 7 lines and make the language formal and friendly. Introduction:"""
        try:
            summary = llm.invoke(prompt).strip()
        except Exception as e:
            summary = f"Error generating summary: {e}"
    
    # Add summary to conversation
    new_history = state["conversation_history"].copy()
    new_history.append({
        "role": "bot", 
        "content": f"@Summary:\n\n{summary}",
        "question_number": 8
    })
    
    return {
        **state,
        "is_complete": True,
        "summary": summary,
        "conversation_history": new_history
    }

def route_conversation(state: PersonalDetailsState) -> Literal["continue", "generate_summary", "end"]:
    """Route based on current state"""
    current_q = state["current_question"]
    
    if current_q > 7:
        if not state["is_complete"]:
            return "generate_summary"
        else:
            return "end"
    else:
        return "continue"

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'graph_state' not in st.session_state:
        st.session_state.graph_state = {
            "current_question": 0,
            "questions": [],
            "answers": [],
            "conversation_history": [],
            "is_complete": False,
            "summary": ""
        }
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None  # We're not using workflow anymore
    if 'waiting_for_answer' not in st.session_state:
        st.session_state.waiting_for_answer = False

def create_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(PersonalDetailsState)
    
    # Add nodes
    workflow.add_node("start", start_conversation)
    workflow.add_node("ask_question", ask_question)
    workflow.add_node("generate_summary", generate_summary)
    
    # Set entry point
    workflow.set_entry_point("start")
    
    # Simple linear flow
    workflow.add_edge("start", "ask_question")
    workflow.add_conditional_edges(
        "ask_question",
        route_conversation,
        {
            "continue": "ask_question",
            "generate_summary": "generate_summary",
            "end": END
        }
    )
    workflow.add_edge("generate_summary", END)
    
    # Compile with increased recursion limit
    compiled = workflow.compile(config={"recursion_limit": 50})

    # Attempt to save the compiled workflow to the same directory as this file
    try:
        save_path = Path(__file__).parent / "langgraph_workflow.pkl"
        try:
            with open(save_path, "wb") as f:
                pickle.dump(compiled, f)
            # store path on workflow object if possible
            try:
                setattr(compiled, "_saved_path", str(save_path))
            except Exception:
                pass
            # If running under Streamlit, give a small info message
            try:
                st.info(f"Saved compiled workflow to: {save_path}")
            except Exception:
                pass
        except Exception:
            # If pickling fails, fall back to writing a text representation
            try:
                text_path = Path(__file__).parent / "langgraph_workflow.txt"
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(str(compiled))
                try:
                    st.warning(f"Could not pickle workflow; wrote text representation to: {text_path}")
                except Exception:
                    pass
            except Exception:
                # give up silently; the app should still run
                pass
    except Exception:
        # non-fatal; do not crash the app
        pass

    return compiled

def process_user_input(user_input: str):
    """Process user input and update state manually"""
    state = st.session_state.graph_state.copy()
    q = state["current_question"]

    # Append user message
    state = add_history(state, "user", user_input, q)

    # Select validator based on current question
    validators = {
        1: validate_name,
        2: validate_age,
        3: validate_gender,
        4: validate_email,
        5: validate_mobile,
        6: validate_country,
        7: validate_profession
    }

    validator = validators.get(q)
    if validator is None:
        # Nothing to validate; move forward
        state["answers"].append(user_input)
        state["current_question"] = q + 1
    else:
        valid, normalized, error = validator(user_input)

        # initialize retries for this question
        retries = state.get("retries", {})
        retries[q] = retries.get(q, 0)

        if valid:
            # accept normalized answer
            state["answers"].append(normalized)
            state["last_validation_error"] = None
            # reset retries for this q
            retries[q] = 0
            state["retries"] = retries
            state["current_question"] = q + 1
            # Ask next question or generate summary
            if state["current_question"] <= 7:
                state = ask_question(state)
            else:
                state = generate_summary(state)
        else:
            # invalid answer: increment retry and add bot error message
            retries[q] += 1
            state["retries"] = retries
            state["last_validation_error"] = error
            # Add bot clarification message
            state = add_history(state, "bot", error + " Please try again.", q)

            # If retries exceeded, accept as-is with a note and move on
            if retries[q] >= 3:
                note = f"I've accepted your response after {retries[q]} attempts."
                state = add_history(state, "bot", note, q)
                state["answers"].append(user_input.strip())
                state["current_question"] = q + 1
                if state["current_question"] <= 7:
                    state = ask_question(state)
                else:
                    state = generate_summary(state)

    st.session_state.graph_state = state
    st.session_state.waiting_for_answer = False

def start_new_session():
    """Start a new collection session"""
    # Initialize state manually instead of using workflow
    initial_state = {
        "current_question": 1,  # Start with question 1
        "questions": QUESTIONS,
        "answers": [],
        "conversation_history": [],
        "is_complete": False,
        "summary": ""
    }
    
    # Ask the first question
    updated_state = ask_question(initial_state)
    st.session_state.graph_state = updated_state
    st.session_state.waiting_for_answer = True

def reset_session():
    """Reset the entire session"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    st.rerun()

def main():
    """Main Streamlit application"""
    # -------------  SAVE/LOAD WORKFLOW GRAPH  -------------
    # Initialize session state
    initialize_session_state()

    # Try to load an already-compiled workflow saved next to this file.
    try:
        wf_path = Path(__file__).parent / "langgraph_workflow.pkl"
        if wf_path.exists():
            try:
                with open(wf_path, "rb") as f:
                    compiled_wf = pickle.load(f)
                st.session_state.compiled_workflow = compiled_wf
                try:
                    st.info(f"Loaded compiled workflow from: {wf_path}")
                except Exception:
                    pass
            except Exception:
                # If loading fails, fall back to creating a fresh compiled workflow
                try:
                    compiled_wf = create_workflow()
                    st.session_state.compiled_workflow = compiled_wf
                except Exception:
                    st.session_state.compiled_workflow = None
        else:
            # No saved workflow; create and save one
            try:
                compiled_wf = create_workflow()
                st.session_state.compiled_workflow = compiled_wf
            except Exception:
                st.session_state.compiled_workflow = None
    except Exception:
        # non-fatal: continue without compiled workflow
        st.session_state.compiled_workflow = None
    
    # Header
    st.title("🤖 Personal Details Collection Bot")
    st.markdown("*Powered by LangGraph - Exactly 7 Questions, Then Summary*")
    st.markdown("---")
    
    # Sidebar with progress
    with st.sidebar:
        st.header("📋 Collection Progress")
        
        current_q = st.session_state.graph_state["current_question"]
        total_questions = 7
        
        # Progress indicators
        for i, label in enumerate(QUESTION_LABELS, 1):
            if i < current_q:
                st.write(f"✅ {i}. {label}")
            elif i == current_q and not st.session_state.graph_state["is_complete"]:
                st.write(f"🔄 {i}. {label} *(current)*")
            else:
                st.write(f"⏳ {i}. {label}")
        
        st.markdown("---")
        
        # Progress bar
        if st.session_state.graph_state["is_complete"]:
            progress = 1.0
            st.progress(progress)
            st.write("**Status:** ✅ Complete!")
        else:
            progress = max(0, (current_q - 1) / total_questions)
            st.progress(progress)
            st.write(f"**Progress:** {max(0, current_q - 1)}/{total_questions}")
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", type="secondary"):
                reset_session()
        
        with col2:
            if current_q == 0:
                if st.button("🚀 Start", type="primary"):
                    start_new_session()
                    st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Conversation")
        
        # Display conversation history
        if st.session_state.graph_state["conversation_history"]:
            for message in st.session_state.graph_state["conversation_history"]:
                if message["role"] == "bot":
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                else:
                    with st.chat_message("user"):
                        st.write(message["content"])
        
        # User input area
        current_state = st.session_state.graph_state
        
        if current_state["current_question"] == 0:
            st.info("👈 Click 'Start' in the sidebar to begin the collection process!")
        elif not current_state["is_complete"] and current_state["current_question"] <= 7:
            qidx = current_state["current_question"] - 1
            question_text = QUESTIONS[qidx]

            # If there was a validation error, display it as a bot message
            if current_state.get("last_validation_error"):
                with st.chat_message("assistant"):
                    st.error(current_state.get("last_validation_error"))

            # Show input for current question with the exact prompt
            user_input = st.chat_input(question_text)

            if user_input and user_input.strip():
                # Process the answer
                process_user_input(user_input.strip())
                st.rerun()
                
        elif current_state["is_complete"]:
            st.success("🎉 Collection Complete!")
            if st.button("Start New Collection", type="primary"):
                reset_session()
    
    with col2:
        st.header("📊 Session Info")
        
        current_state = st.session_state.graph_state
        
        if current_state["is_complete"]:
            st.success("🎉 All details collected!")
            
            # Show collected answers
            st.subheader("📝 Collected Data")
            for i, (label, answer) in enumerate(zip(QUESTION_LABELS, current_state["answers"]), 1):
                st.write(f"**{i}. {label}:** {answer}")
                
        elif current_state["current_question"] > 0:
            st.info("🔄 Collection in progress...")
            st.write(f"Current question: {current_state['current_question']}/7")
            
            # Show answers so far
            if current_state["answers"]:
                st.subheader("✅ Answered So Far")
                for i, answer in enumerate(current_state["answers"], 1):
                    st.write(f"**{QUESTION_LABELS[i-1]}:** {answer}")
        else:
            st.warning("⏳ Ready to start collection")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>LangGraph-Powered Collection Bot | 7 Questions → Auto Summary</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        # Check dependencies
        missing_packages = []
        try:
            import streamlit
        except ImportError:
            missing_packages.append("streamlit")
        
        try:
            from langchain_ollama import OllamaLLM
            from langgraph.graph import StateGraph, END
        except ImportError:
            missing_packages.append("langchain-ollama langgraph")
        
        if missing_packages:
            st.error(f"Missing packages: {', '.join(missing_packages)}")
            st.code(f"pip install {' '.join(missing_packages)}")
            st.stop()
            
        main()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.write("Ensure Ollama is running with llama3.1:8b-instruct-q4_K_M model")