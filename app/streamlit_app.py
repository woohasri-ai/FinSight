
import re
import time
import requests
import streamlit as st


st.set_page_config(
    page_title="FinSight | Intelligence Workspace",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://127.0.0.1:8000"


CAREER_SECTIONS = {
    "overview": {
        "title": "Career Guide",
        "subtitle": "Plan your career, build the right skills, and prepare for opportunities.",
        "prompts": [
            "I want to become an AI/ML Engineer. Guide me from beginner to job-ready.",
            "Create a learning roadmap for Python, SQL, Machine Learning and Deep Learning.",
            "How should I prepare for a fresher technical interview?",
            "Review the skills I need to build a strong AI/ML portfolio.",
        ],
    },
    "explore": {
        "title": "Explore Careers",
        "subtitle": "Understand job roles, responsibilities, required skills and career paths.",
        "prompts": [
            "What does an AI/ML Engineer do in a company?",
            "Compare AI Engineer, ML Engineer and Data Scientist roles.",
            "Which entry-level technology roles should I explore?",
        ],
    },
    "roadmaps": {
        "title": "Skills & Learning Roadmaps",
        "subtitle": "Learn what to study, in what order, and how to practise each skill.",
        "prompts": [
            "Create a complete AI/ML Engineer learning roadmap.",
            "List the exact Python topics I should learn for AI/ML.",
            "Explain the Machine Learning topics I need to master.",
        ],
    },
    "preparation": {
        "title": "Job Preparation",
        "subtitle": "Prepare for aptitude tests, technical rounds and recruitment processes.",
        "prompts": [
            "Create a fresher AI/ML Engineer interview preparation plan.",
            "What SQL topics are commonly tested in technical interviews?",
            "How should I prepare for coding rounds?",
        ],
    },
    "interview": {
        "title": "Interview Practice",
        "subtitle": "Practise technical questions, explain your answers and identify weak areas.",
        "prompts": [
            "Start an AI/ML Engineer mock interview, one question at a time.",
            "Ask me Python interview questions for a fresher role.",
            "Test my Machine Learning fundamentals.",
        ],
    },
    "resume": {
        "title": "Resume & Projects",
        "subtitle": "Build a stronger portfolio and present your projects professionally.",
        "prompts": [
            "Suggest practical AI/ML projects for my resume.",
            "How should I explain a RAG project in an interview?",
            "Help me improve my project descriptions for recruiters.",
        ],
    },
    "jobs": {
        "title": "Find Jobs",
        "subtitle": "Learn where to search, how to apply and how to verify opportunities.",
        "prompts": [
            "Where can freshers find AI/ML Engineer jobs in India?",
            "List the best job portals and company career pages for AI/ML roles.",
            "How do I apply for internships and entry-level AI jobs?",
        ],
    },
}


FINANCIAL_SAMPLE_QUESTIONS = [
    "What was TCS consolidated revenue in FY 2023-24?",
    "What was TCS revenue in FY 2024-25?",
    "What was TCS revenue in FY 2025-26?",
    "What was TCS operating margin in FY 2023-24?",
    "What was TCS net income in FY 2024-25?",
    "What were TCS total assets in FY 2025-26?",
    "What was Infosys revenue in FY 2025-26?",
    "What was Infosys operating margin in FY 2025-26?",
    "What was Infosys net profit in FY 2025-26?",
    "What were Infosys total assets in FY 2025-26?",
    "What was Wipro revenue in FY 2025-26?",
    "What was Wipro operating margin in FY 2025-26?",
    "What was Wipro net income in FY 2025-26?",
    "What were Wipro total assets in FY 2025-26?",
    "What was HCLTech revenue in FY 2025-26?",
    "What was HCLTech operating margin in FY 2025-26?",
    "What was HCLTech net income in FY 2025-26?",
    "What were HCLTech total assets in FY 2025-26?",
    "What was LTIMindtree revenue in FY 2025-26?",
    "What was LTIMindtree operating margin in FY 2025-26?",
    "What was Tech Mahindra revenue in FY 2025-26?",
    "What was Tech Mahindra operating margin in FY 2025-26?",
    "What was Tech Mahindra net income in FY 2025-26?",
    "What were Tech Mahindra total assets in FY 2025-26?",
    "Compare TCS and Infosys revenue using their annual reports.",
    "Compare the operating margins of the covered companies.",
    "Which company reported the highest revenue among the indexed companies?",
    "Find the consolidated revenue figures reported by TCS.",
    "What financial risks does TCS discuss in its annual report?",
    "Summarize the financial performance discussed in the TCS annual report.",
]


# ---------------------------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --background: #f5f7fb;
        --surface: #ffffff;
        --text: #172033;
        --muted: #5d687d;
        --border: #e2e7f0;
        --accent: #4058d6;
        --accent-light: #eef1ff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text) !important;
    }

    .stApp {
        background: var(--background);
        color: var(--text) !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: #263149;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted) !important;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #172033 !important;
        margin-bottom: 3px;
    }

    .brand span {
        color: var(--accent) !important;
    }

    .eyebrow {
        color: var(--accent) !important;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(30px, 4vw, 44px);
        line-height: 1.18;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #172033 !important;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #5d687d !important;
        font-size: 16px;
        line-height: 1.7;
        max-width: 720px;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #172033 !important;
    }

    .muted {
        color: #5d687d !important;
        font-size: 14px;
        line-height: 1.65;
    }

    .workspace-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 23px;
        min-height: 190px;
        box-shadow: 0 5px 22px rgba(26, 39, 75, 0.035);
    }

    .workspace-icon {
        font-size: 25px;
        margin-bottom: 12px;
    }

    .workspace-title {
        font-family: 'Manrope', sans-serif;
        font-size: 21px;
        font-weight: 800;
        color: #172033 !important;
        margin-bottom: 8px;
    }

    .workspace-description {
        color: #5d687d !important;
        font-size: 14px;
        line-height: 1.7;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 15px;
        min-height: 105px;
    }

    .section-card strong {
        color: #172033 !important;
        font-size: 14px;
    }

    .section-card p {
        color: #5d687d !important;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 6px;
    }

    .info-panel {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 16px 18px;
        color: #45516a !important;
        font-size: 13px;
        line-height: 1.7;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #e9f7ef;
        color: #17643a !important;
        font-size: 12px;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 40px;
        font-weight: 600;
        border: 1px solid #dfe4ee;
        color: #25314a !important;
        background: #ffffff;
    }

    .stButton > button[kind="primary"] {
        background: #4058d6 !important;
        border-color: #4058d6 !important;
        color: #ffffff !important;
    }

    .stButton > button:hover {
        border-color: #4058d6;
        color: #4058d6 !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover * {
        background: #4058d6 !important;
        border-color: #4058d6 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e2e7f0 !important;
        border-radius: 14px !important;
        padding: 16px !important;
        color: #172033 !important;
    }

    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th {
        color: #172033 !important;
    }

    [data-testid="stChatMessage"] a {
        color: #304ecb !important;
    }

    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #cfd6e4 !important;
        border-radius: 13px !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        caret-color: #172033 !important;
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #778197 !important;
        -webkit-text-fill-color: #778197 !important;
        opacity: 1 !important;
    }

    [data-testid="stTextInput"] input {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        background: #ffffff !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 14px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

def initialize_state():
    defaults = {
        "page": "home",
        "career_section": "overview",
        "general_messages": [],
        "general_history": [],
        "chat_messages": [],
        "chat_history": [],
        "pending_chat_prompt": None,
        "financial_messages": [],
        "pending_general_prompt": None,
        "pending_financial_prompt": None,
        "api_url": DEFAULT_API_URL,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ---------------------------------------------------------------------
# COMMON FUNCTIONS
# ---------------------------------------------------------------------

def check_api(base_url):
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/health",
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def go_to(page):
    st.session_state.page = page
    st.rerun()


def set_career_section(section):
    st.session_state.career_section = section
    st.session_state.pending_general_prompt = None
    st.rerun()


def submit_financial_sample(question):
    st.session_state.pending_financial_prompt = question


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand">Fin<span>Sight</span></div>',
            unsafe_allow_html=True,
        )

        st.caption("AI-powered intelligence workspace")
        st.divider()

        st.markdown("**WORKSPACES**")

        if st.button("Home", use_container_width=True):
            go_to("home")

        if st.button(
            "Career Guide",
            type="primary" if st.session_state.page == "general" else "secondary",
            use_container_width=True,
        ):
            go_to("general")

        if st.button(
            "General Chat",
            type="primary" if st.session_state.page == "general_chat" else "secondary",
            use_container_width=True,
        ):
            go_to("general_chat")

        if st.session_state.page == "general":
            st.caption("CAREER GUIDE")

            for key, section in CAREER_SECTIONS.items():
                if st.button(
                    section["title"],
                    key=f"career_nav_{key}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.career_section == key
                        else "secondary"
                    ),
                ):
                    set_career_section(key)

        if st.button(
            "Financial Research",
            type="primary" if st.session_state.page == "financial" else "secondary",
            use_container_width=True,
        ):
            go_to("financial")

        st.divider()
        st.markdown("**CONNECTION**")

        api_url = st.text_input(
            "Backend API URL",
            value=st.session_state.api_url,
            key="api_url_input",
        )

        st.session_state.api_url = api_url.rstrip("/")
        online = check_api(st.session_state.api_url)

        if online:
            st.markdown(
                '<span class="status-pill">API Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.error("API unavailable. Start the FastAPI backend.")

        st.divider()
        st.caption("FinSight | Career and Financial Intelligence")

    return online


# ---------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------

def render_home(api_online):
    st.markdown(
        '<div class="eyebrow">YOUR INTELLIGENCE WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">Build your future.<br>Make informed decisions.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Chat freely, build your career, or investigate company performance "
        "using evidence from financial reports."
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    general_col, career_col, financial_col = st.columns(3, gap="large")

    with general_col:
        st.markdown(
            """
            <div class="workspace-card">
                <div class="workspace-icon">&#128172;</div>
                <div class="workspace-title">General Chat</div>
                <div class="workspace-description">
                    Ask questions, explore ideas, understand concepts, and get
                    help with everyday tasks through a conversational AI assistant.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Open General Chat",
            key="open_general_chat",
            type="primary",
            use_container_width=True,
        ):
            go_to("general_chat")

    with career_col:
        st.markdown(
            """
            <div class="workspace-card">
                <div class="workspace-icon">&#127919;</div>
                <div class="workspace-title">Career Guide</div>
                <div class="workspace-description">
                    Explore career paths, learn skills in the right order,
                    create study plans, practise interviews, improve your resume,
                    and prepare for job applications.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Explore Career Guide",
            key="open_career",
            use_container_width=True,
        ):
            go_to("general")

    with financial_col:
        st.markdown(
            """
            <div class="workspace-card">
                <div class="workspace-icon">&#128202;</div>
                <div class="workspace-title">Financial Research</div>
                <div class="workspace-description">
                    Research company annual reports, retrieve supporting evidence,
                    explore financial information, and review source-aware answers
                    with document citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Open Financial Research",
            key="open_financial",
            use_container_width=True,
        ):
            go_to("financial")

    st.write("")

    st.markdown(
        '<div class="section-title">Explore Career Guide</div>',
        unsafe_allow_html=True,
    )

    st.caption("Choose an area to get started.")

    columns = st.columns(3, gap="medium")
    section_items = list(CAREER_SECTIONS.items())[1:]

    for index, (key, section) in enumerate(section_items):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="section-card">
                    <strong>{section["title"]}</strong>
                    <p>{section["subtitle"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"Open {section['title']}",
                key=f"home_section_{key}",
                use_container_width=True,
            ):
                st.session_state.career_section = key
                go_to("general")


# ---------------------------------------------------------------------
# CAREER GUIDE HELPERS
# ---------------------------------------------------------------------

def split_followups(answer):
    patterns = [
        r"(?im)^#{1,4}\s\*(?:suggested\s+)?follow[- ]?up questions\s\*:?\s*$",
        r"(?im)^(?:suggested\s+)?follow[- ]?up questions\s\*:?\s*$",
        r"(?im)^#{1,4}\s\*what would you like to explore next\?\s*$",
    ]

    match = None

    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            break

    if not match:
        return answer.strip(), []

    main_answer = answer[:match.start()].strip()
    followup_text = answer[match.end():].strip()

    questions = []

    for line in followup_text.splitlines():
        cleaned = re.sub(
            r"^\s*(?:[-*•]|\d+[.)])\s*",
            "",
            line,
        ).strip()

        cleaned = cleaned.strip('`"\\ ')

        if cleaned and cleaned.endswith("?"):
            questions.append(cleaned)

    return main_answer, questions[:4]


def render_message(message, index):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("followups"):
            st.markdown("**Continue exploring**")
            columns = st.columns(2)

            for question_index, question in enumerate(message["followups"]):
                with columns[question_index % 2]:
                    if st.button(
                        question,
                        key=f"followup_{index}_{question_index}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_general_prompt = question
                        st.rerun()


# ---------------------------------------------------------------------
# CAREER GUIDE
# ---------------------------------------------------------------------

def render_career_guide(api_online):
    section = CAREER_SECTIONS[st.session_state.career_section]

    st.markdown(
        '<div class="eyebrow">CAREER PLANNING AND JOB READINESS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-title">{section["title"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="muted">{section["subtitle"]}</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="info-panel">
            <strong>How Career Guide helps</strong><br>
            Get practical explanations, ordered learning steps, study plans,
            recommended resources, project guidance and interview preparation.
            Current job openings and application links require live verification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_online:
        st.warning(
            "The backend is not connected. Start the FastAPI server to use Career Guide."
        )
        return

    if not st.session_state.general_messages:
        st.markdown("#### Start with a question")
        st.caption("Choose a suggestion or type your own career question below.")

        prompts = section["prompts"]

        for index in range(0, len(prompts), 2):
            columns = st.columns(2)

            for offset, column in enumerate(columns):
                prompt_index = index + offset

                if prompt_index < len(prompts):
                    with column:
                        if st.button(
                            prompts[prompt_index],
                            key=f"career_prompt_{st.session_state.career_section}_{prompt_index}",
                            use_container_width=True,
                        ):
                            st.session_state.pending_general_prompt = prompts[prompt_index]
                            st.rerun()

    for index, message in enumerate(st.session_state.general_messages):
        render_message(message, index)

    pending = st.session_state.pending_general_prompt

    prompt = st.chat_input(
        "Ask Career Guide about your career, skills or job preparation..."
    )

    if pending:
        prompt = pending
        st.session_state.pending_general_prompt = None

    if prompt:
        st.session_state.general_messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Preparing your career guidance..."):
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/chat",
                        json={
                            "message": prompt,
                            "history": st.session_state.general_history,
                        },
                        timeout=360,
                    )

                    if response.status_code == 200:
                        data = response.json()

                        raw_answer = data.get(
                            "response",
                            "I couldn't generate a response.",
                        )

                        answer, followups = split_followups(raw_answer)

                        st.session_state.general_history = data.get(
                            "conversation_history",
                            [],
                        )

                    else:
                        try:
                            detail = response.json().get(
                                "detail",
                                response.text,
                            )
                        except ValueError:
                            detail = response.text

                        answer = (
                            f"**Request failed ({response.status_code}).** {detail}"
                        )
                        followups = []

                except requests.Timeout:
                    answer = "The request timed out. Please try again."
                    followups = []

                except requests.RequestException as exc:
                    answer = f"Could not connect to the backend: {exc}"
                    followups = []

            st.text(answer)

            if followups:
                st.markdown("**Continue exploring**")
                columns = st.columns(2)

                for question_index, question in enumerate(followups):
                    with columns[question_index % 2]:
                        if st.button(
                            question,
                            key=f"new_followup_{len(st.session_state.general_messages)}_{question_index}",
                            use_container_width=True,
                        ):
                            st.session_state.pending_general_prompt = question

        st.session_state.general_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "followups": followups,
            }
        )

        st.rerun()

    if st.session_state.general_messages:
        if st.button("Start a new conversation", key="clear_general"):
            st.session_state.general_messages = []
            st.session_state.general_history = []
            st.session_state.pending_general_prompt = None
            st.rerun()


# ---------------------------------------------------------------------
# GENERAL CHAT
# ---------------------------------------------------------------------

def render_general_chat(api_online):
    st.markdown(
        '<div class="eyebrow">YOUR GENERAL-PURPOSE AI ASSISTANT</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">General Chat</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">Ask questions, explore ideas, learn concepts, '
        'and work through everyday problems in a separate conversation.</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    if not api_online:
        st.warning(
            "The backend is not connected. Start the FastAPI server to use General Chat."
        )
        return

    if not st.session_state.chat_messages:
        st.markdown("#### Start a conversation")

        starter_prompts = [
            "Explain a difficult concept in simple terms.",
            "Help me plan and break down a project.",
            "Give me ideas for learning and problem-solving.",
        ]

        prompt_columns = st.columns(3)

        for index, starter in enumerate(starter_prompts):
            with prompt_columns[index]:
                if st.button(
                    starter,
                    key=f"general_chat_starter_{index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_chat_prompt = starter
                    st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask FinSight anything...")

    pending = st.session_state.pending_chat_prompt

    if pending:
        prompt = pending
        st.session_state.pending_chat_prompt = None

    if prompt:
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/chat",
                        json={
                            "message": prompt,
                            "history": st.session_state.chat_history,
                        },
                        timeout=360,
                    )

                    if response.status_code == 200:
                        data = response.json()

                        answer = data.get(
                            "response",
                            "I couldn't generate a response.",
                        )

                        st.session_state.chat_history = data.get(
                            "conversation_history",
                            [],
                        )

                    else:
                        try:
                            detail = response.json().get(
                                "detail",
                                response.text,
                            )
                        except ValueError:
                            detail = response.text

                        answer = (
                            f"**Request failed ({response.status_code}).** {detail}"
                        )

                except requests.Timeout:
                    answer = "The request timed out. Please try again."

                except requests.RequestException as exc:
                    answer = f"Could not connect to the backend: {exc}"

            st.markdown(answer)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

        st.rerun()

    if st.session_state.chat_messages:
        if st.button("Start a new General Chat", key="clear_general_chat"):
            st.session_state.chat_messages = []
            st.session_state.chat_history = []
            st.session_state.pending_chat_prompt = None
            st.rerun()


# ---------------------------------------------------------------------
# FINANCIAL RESEARCH CITATIONS
# ---------------------------------------------------------------------

def render_citations(citations):
    if not citations:
        return

    st.markdown("#### Supporting evidence")

    for index, citation in enumerate(citations, start=1):
        document = citation.get("document_name", "Unknown document")
        page = citation.get("page_number", "N/A")
        chunk_id = citation.get("chunk_id", "")
        evidence = citation.get("evidence_text", "")

        with st.expander(f"Source {index} | {document} | Page {page}"):
            if evidence:
                st.write(evidence)

            if chunk_id:
                st.caption(f"Chunk ID: {chunk_id}")


# ---------------------------------------------------------------------
# FINANCIAL RESEARCH SAMPLE QUESTIONS
# ---------------------------------------------------------------------

@st.fragment(run_every="60s")
def render_financial_sample_questions():
    st.markdown("#### Try these questions")

    st.caption(
        "Choose any question to research it instantly. "
        "Suggestions automatically change every 60 seconds."
    )

    # Three rotating sets of ten questions.
    current_set = int(time.time() // 60) % 3
    start = current_set * 10
    questions = FINANCIAL_SAMPLE_QUESTIONS[start:start + 10]

    for index in range(0, len(questions), 2):
        columns = st.columns(2)

        for offset, column in enumerate(columns):
            question_index = index + offset

            if question_index < len(questions):
                question = questions[question_index]

                with column:
                    if st.button(
                        question,
                        key=f"financial_sample_{current_set}_{question_index}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_financial_prompt = question
                        st.rerun(scope="app")


# ---------------------------------------------------------------------
# FINANCIAL RESEARCH
# ---------------------------------------------------------------------

def render_financial_research(api_online):
    st.markdown(
        '<div class="eyebrow">EVIDENCE-GROUNDED COMPANY ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Financial Research</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">Explore company financials, compare performance, '
        'and uncover key business insights using data from annual reports. '
        'Ask FinSight questions about revenue, profitability, margins, '
        'business segments, employee strength, and more.</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="info-panel">
            <strong>Research workspace</strong><br>
            Answers use the financial research pipeline. Review the cited
            document evidence when it is provided. If evidence is insufficient,
            the system should say so rather than inventing an answer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if not api_online:
        st.warning(
            "The backend is not connected. Start the FastAPI server to use Financial Research."
        )
        return

    # Display rotating questions before the chat.
    render_financial_sample_questions()

    st.divider()

    for index, message in enumerate(st.session_state.financial_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("citations"):
                render_citations(message["citations"])

    question = st.chat_input(
        "Ask a question about a company's financial reports..."
    )

    # A clicked sample question takes priority over typed input.
    pending = st.session_state.pending_financial_prompt

    if pending:
        question = pending
        st.session_state.pending_financial_prompt = None

    if question:
        st.session_state.financial_messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching financial evidence and preparing an answer..."
            ):
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/ask",
                        json={"question": question},
                        timeout=360,
                    )

                    if response.status_code == 200:
                        data = response.json()

                        answer = data.get(
                            "answer",
                            "No answer was returned.",
                        )

                        citations = data.get("citations", [])
                        abstained = data.get("abstained", False)

                        if abstained:
                            st.info(answer)
                        else:
                            st.markdown(answer)

                        render_citations(citations)

                    else:
                        citations = []

                        try:
                            detail = response.json().get(
                                "detail",
                                response.text,
                            )
                        except ValueError:
                            detail = response.text

                        answer = (
                            f"**Request failed ({response.status_code}).** {detail}"
                        )

                        st.error(answer)

                except requests.Timeout:
                    citations = []
                    answer = "The request timed out. Please try again."
                    st.error(answer)

                except requests.RequestException as exc:
                    citations = []
                    answer = f"Could not connect to the backend: {exc}"
                    st.error(answer)

        st.session_state.financial_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
            }
        )

        st.rerun()

    if st.session_state.financial_messages:
        if st.button("Clear research conversation", key="clear_financial"):
            st.session_state.financial_messages = []
            st.session_state.pending_financial_prompt = None
            st.rerun()


# ---------------------------------------------------------------------
# APP ROUTING
# ---------------------------------------------------------------------

api_online = render_sidebar()

if st.session_state.page == "general_chat":
    render_general_chat(api_online)

elif st.session_state.page == "general":
    render_career_guide(api_online)

elif st.session_state.page == "financial":
    render_financial_research(api_online)

else:
    render_home(api_online)
