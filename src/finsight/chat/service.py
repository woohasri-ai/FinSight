
from pydantic import BaseModel, Field, field_validator

from finsight.llm.client import LLMClient

GENERAL_CHAT_SYSTEM_PROMPT = """
You are FinSight Career Guide, a helpful assistant for students, freshers,
and early-career professionals.

Help with career decisions, technical learning, interview preparation,
projects, resumes, job searches, and general questions.

Answer the latest question directly. Preserve conversation context and
the user's target role. Explain concepts simply and give practical advice.
Prioritize essential skills before advanced topics.

Never invent courses, URLs, vacancies, hiring status, salaries, deadlines,
experience, or achievements. Distinguish general advice from verified
current information.

For project recommendations, explain the problem, features, technology
stack, implementation stages, and resume value.

For mock interviews, ask one question at a time.

For substantial career-related answers, finish with:

### Suggested follow-up questions

Give three or four specific questions on separate lines.
Do not add follow-ups to simple conversational replies.

You are FinSight Career Guide, separate from Financial Research.
Financial Research has its own evidence-grounded retrieval workflow.
""".strip()


RESPONSE_GUIDANCE = {
    "roadmap": """
Create a practical learning roadmap for the user's requested target role.
Use conversation context to preserve their target role and current skills.
Prioritize beginner-friendly, essential skills and realistic exercises.
Include weekly milestones, practical projects, and measurable outcomes.
Never invent learning resources or URLs.
""",
    "interview": """
RESPONSE FORMAT: INTERVIEW PREPARATION

1. Target role and interview stages.
2. Important subjects in priority order.
3. Technical concepts and coding practice.
4. Relevant project-based questions.
5. Example questions and explanations when useful.
6. A realistic practice schedule.
7. Readiness checklist.
8. Suggested follow-up questions.

For a mock interview, ask one question at a time.
""",
    "project": """
RESPONSE FORMAT: PROJECT RECOMMENDATIONS

1. Project name and problem statement.
2. Why it is useful and what skills it demonstrates.
3. Technology stack and reasons.
4. Main features.
5. Step-by-step implementation milestones.
6. Testing and expected final outcome.
7. GitHub and resume presentation.
8. Suggested follow-up questions.

Match project difficulty to the user's current skills.
""",
    "resume": """
RESPONSE FORMAT: RESUME REVIEW

1. Overall assessment.
2. Strengths.
3. Important weaknesses.
4. Section-by-section feedback.
5. Concrete rewritten examples.
6. Missing information or improvements.
7. Suggested follow-up questions.

Never invent personal achievements or experience.
If no resume is provided, ask the user to share it.
""",
    "job_preparation": """
RESPONSE FORMAT: JOB PREPARATION

1. Target role and expected eligibility.
2. Important skills commonly required.
3. Skill gaps to address.
4. Recruitment stages and preparation strategy.
5. Weekly preparation plan.
6. Practice tasks and readiness checklist.
7. Suggested follow-up questions.

Do not claim company-specific requirements without evidence.
""",
    "job_search": """
RESPONSE FORMAT: JOB SEARCH GUIDANCE

1. Relevant job titles and search keywords.
2. Suitable job portals.
3. Official company career pages where confidently known.
4. Fresher and internship application strategy.
5. Resume and portfolio preparation.
6. Application tracking and follow-up strategy.
7. Suggested follow-up questions.

Keep the answer concise and practical. Prioritize the most useful
job portals and search keywords rather than lengthy descriptions.

Never invent current vacancies or application links.
Clearly distinguish general advice from verified live information.
""",
}


AI_ML_ROADMAP = [
    (
        "Python data structures and functions review",
        "Solve five small problems using lists, dictionaries, and functions.",
    ),
    (
        "NumPy arrays, shapes, indexing, and data types",
        "Create arrays and practise indexing, slicing, and reshaping.",
    ),
    (
        "NumPy arithmetic, aggregation, and broadcasting",
        "Calculate averages, minimums, maximums, and totals from arrays.",
    ),
    (
        "Pandas Series and DataFrames",
        "Load a CSV file and inspect its rows, columns, and data types.",
    ),
    (
        "Pandas filtering, sorting, and selecting columns",
        "Answer five questions by filtering and sorting a dataset.",
    ),
    (
        "Missing values, duplicates, and data type cleaning",
        "Clean a small dataset and explain each change.",
    ),
    (
        "Descriptive statistics and basic data visualization",
        "Calculate summary statistics and create three useful charts.",
    ),
    (
        "Exploratory data analysis and understanding distributions",
        "Explore a dataset and write five observations.",
    ),
    (
        "Feature columns, target columns, and data leakage",
        "Choose features and a target for a prediction problem.",
    ),
    (
        "Categorical encoding and numerical scaling",
        "Encode categorical columns and scale numerical features.",
    ),
    (
        "Training data, test data, and train-test splitting",
        "Split a dataset and explain why test data must remain unseen.",
    ),
    (
        "Linear regression and its basic assumptions",
        "Train a linear regression model on a simple numerical dataset.",
    ),
    (
        "Regression metrics: MAE, MSE, RMSE, and R-squared",
        "Calculate and compare regression evaluation metrics.",
    ),
    (
        "Review regression and practise SQL data analysis",
        "Solve five SQL questions involving filtering, grouping, and joins.",
    ),
    (
        "Classification problems and logistic regression",
        "Train a logistic regression classifier on a labelled dataset.",
    ),
    (
        "Decision trees and how predictions are made",
        "Train a decision tree and inspect its predictions.",
    ),
    (
        "Classification probabilities and decision thresholds",
        "Compare predictions using two classification thresholds.",
    ),
    (
        "Confusion matrix and classification metrics",
        "Calculate accuracy, precision, recall, and F1-score.",
    ),
    (
        "Compare classification models fairly",
        "Compare logistic regression and a decision tree on the same split.",
    ),
    (
        "Overfitting, underfitting, and generalization",
        "Compare training and test performance to identify overfitting.",
    ),
    (
        "Basic model improvement and validation",
        "Change one model parameter and compare the results.",
    ),
    (
        "Start the customer churn prediction project",
        "Download a suitable public telecom churn dataset and inspect it.",
    ),
    (
        "Clean and explore the project dataset",
        "Handle missing values, duplicates, data types, and distributions.",
    ),
    (
        "Prepare project features and target",
        "Encode categorical columns and create a train-test split.",
    ),
    (
        "Build a project baseline classification model",
        "Train a logistic regression model and record its metrics.",
    ),
    (
        "Train and compare a decision tree",
        "Compare both models using precision, recall, and F1-score.",
    ),
    (
        "Interpret errors and identify model limitations",
        "Inspect incorrect predictions and explain possible causes.",
    ),
    (
        "Practise Python and SQL interview questions",
        "Solve three Python problems and five SQL problems.",
    ),
    (
        "Prepare project documentation and interview explanation",
        "Write the problem, approach, metrics, results, and limitations.",
    ),
    (
        "Final project review and GitHub presentation",
        "Test the project, update its README, and explain it aloud.",
    ),
]


def build_ai_ml_roadmap() -> str:
    """Build a complete, deterministic 30-day beginner AI/ML roadmap."""

    sections = [
        "# 30-Day Learning Roadmap for a Fresher AI/ML Engineer",
        "",
        "This plan assumes basic Python and SQL knowledge. If a topic is "
        "already familiar, use its exercise for additional practice.",
        "",
    ]

    week_ranges = [
        ("Week 1: Data Foundations", 0, 7),
        ("Week 2: Core Machine Learning", 7, 14),
        ("Week 3: Model Building and Evaluation", 14, 21),
        ("Week 4: Project and Interview Preparation", 21, 30),
    ]

    for week_name, start, end in week_ranges:
        sections.extend(
            [
                f"### {week_name} (Days {start + 1}-{end})",
                "",
                "| Day | Learning task | Practical exercise |",
                "|---:|---|---|",
            ]
        )

        for index in range(start, end):
            task, exercise = AI_ML_ROADMAP[index]
            sections.append(f"| {index + 1} | {task} | {exercise} |")

        sections.append("")

    sections.extend(
        [
            "### Project: Customer Churn Prediction",
            "",
            "**Problem:** Predict whether a telecom customer is likely "
            "to leave the service using a public customer churn dataset.",
            "",
            "**Tools:** Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and SQL.",
            "",
            "- **Week 1:** Inspect, clean, and understand the dataset.",
            "- **Week 2:** Prepare features and build a baseline model.",
            "- **Week 3:** Evaluate models, compare metrics, and inspect errors.",
            "- **Week 4:** Test the final model, document limitations, "
            "and publish the project on GitHub.",
            "",
            "### Weekly Outcomes",
            "",
            "- **Week 1:** A cleaned dataset and basic exploratory analysis.",
            "- **Week 2:** A prepared dataset and working regression practice.",
            "- **Week 3:** Two classification models with evaluated results.",
            "- **Week 4:** A documented, tested beginner ML project.",
            "",
            "### Reliable Learning Resources",
            "",
            "1. [NumPy documentation](https://numpy.org/doc/stable/) "
            "- arrays and numerical operations.",
            "2. [Pandas documentation](https://pandas.pydata.org/docs/) "
            "- data manipulation and cleaning.",
            "3. [Scikit-learn documentation](https://scikit-learn.org/stable/) "
            "- preprocessing, models, and evaluation.",
            "4. [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/) "
            "- data visualization.",
            "5. [W3Schools SQL tutorial](https://www.w3schools.com/sql/) "
            "- SQL revision and practice.",
            "",
            "### Essential Now vs Later",
            "",
            "**Essential now:** Python, NumPy, Pandas, SQL, basic statistics, "
            "data cleaning, visualization, preprocessing, core ML models, "
            "and model evaluation.",
            "",
            "**Learn later:** Deep learning, neural networks, TensorFlow, "
            "PyTorch, reinforcement learning, and advanced ML techniques.",
            "",
            "### Suggested Follow-Up Questions",
            "",
            "- Can you turn this roadmap into a daily study timetable?",
            "- Which free datasets should I use for the project?",
            "- Can you test my understanding of Week 1 topics?",
            "- How should I explain this project in an interview?",
        ]
    )

    return "\n".join(sections)


def detect_request_type(request: "ChatRequest") -> str | None:
    """Classify the latest user message, not older unrelated requests."""

    message = request.message.lower()

    categories = {
        "resume": (
            "resume",
            "cv review",
            "review my cv",
            "improve my cv",
        ),
        "interview": (
            "interview",
            "mock interview",
            "technical round",
            "hr round",
        ),
        "project": (
            "project idea",
            "project recommendation",
            "suggest projects",
            "build a project",
            "project roadmap",
        ),
        "job_search": (
            "find jobs",
            "find a job",
            "where can freshers find",
            "where can i find",
            "where can we find",
            "job opportunities",
            "job openings",
            "job vacancies",
            "companies hiring",
            "where to apply",
            "job portals",
            "internship openings",
            "current vacancies",
            "jobs in india",
            "ai/ml engineer jobs",
            "aiml engineer jobs",
            "machine learning engineer jobs",
            "data scientist jobs",
            "fresher jobs",
            "internship jobs",
        ),
        "job_preparation": (
            "job preparation",
            "prepare for a job",
            "placement preparation",
            "campus placements",
            "prepare for placements",
            "campus recruitment",
            "placement drive",
            "aptitude preparation",
            "recruitment process",
        ),
        "roadmap": (
            "roadmap",
            "learning plan",
            "study plan",
            "30-day plan",
            "30 day plan",
            "60-day plan",
            "60 day plan",
            "learning schedule",
        ),
    }

    for category, keywords in categories.items():
        if any(keyword in message for keyword in keywords):
            return category

    return None


def _compact_history(
    history: list["ChatMessage"],
    *,
    max_messages: int = 6,
    max_content_chars: int = 900,
) -> list[dict[str, str]]:
    """Keep recent context while limiting oversized previous responses."""

    compacted = []

    for item in history[-max_messages:]:
        content = item.content.strip()

        if len(content) > max_content_chars:
            content = (
                content[:max_content_chars]
                + "\n[Earlier response shortened to preserve conversation context.]"
            )

        compacted.append(
            {
                "role": item.role,
                "content": content,
            }
        )

    return compacted


class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in {"user", "assistant"}:
            raise ValueError("Message role must be user or assistant.")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValueError("Message content cannot be blank.")
        return content


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Message cannot be blank.")
        return message


class ChatResponse(BaseModel):
    response: str
    conversation_history: list[ChatMessage]


class GeneralChatService:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request_type = detect_request_type(request)

        # Use the latest request to select a response format. Previous
        # messages are consulted only to recover the user's target role.
        recent_context = _compact_history(request.history)
        context_text = " ".join(
            item["content"].lower() for item in recent_context
        )

        is_ai_ml_roadmap = request_type == "roadmap" and any(
            term in f"{context_text} {request.message.lower()}"
            for term in (
                "ai/ml",
                "ai ml",
                "machine learning engineer",
                "aiml engineer",
            )
        )

        if is_ai_ml_roadmap:
            response = build_ai_ml_roadmap()
        else:
            messages = [
                *recent_context,
                {"role": "user", "content": request.message},
            ]

            conversation_context = "\n\n".join(
                f"{item['role'].capitalize()}: {item['content']}"
                for item in messages
            )

            response_guidance = RESPONSE_GUIDANCE.get(
                request_type,
                "Answer the latest question naturally and directly. "
                "Support general questions as well as career questions. "
                "Do not force a structured template.",
            )

            prompt = (
                "Continue the conversation as FinSight Career Guide.\n"
                "Answer the latest user message directly.\n"
                "Do not repeat previous answers unless the user asks you to.\n"
                "Use previous messages only when they help answer the latest question.\n"
                "Preserve the user's goals and target role when relevant.\n"
                "Follow the response guidance below. Be clear and complete.\n\n"
                f"{response_guidance}\n\n"
                "CONVERSATION:\n"
                f"{conversation_context}\n\n"
                "YOUR RESPONSE:"
            )

            # Shorter limits reduce generation time. Longer limits remain
            # available for requests that genuinely need detailed guidance.
            token_limits = {
                "job_search": 500,
                "resume": 700,
                "interview": 750,
                "project": 750,
                "job_preparation": 650,
                "roadmap": 850,
            }

            num_predict = token_limits.get(request_type, 500)

            response = await self.llm.generate(
                prompt,
                system=GENERAL_CHAT_SYSTEM_PROMPT,
                num_predict=num_predict,
            )

        response = response.strip()

        if not response:
            raise ValueError("The language model returned an empty response.")

        updated_history = [
            *request.history,
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=response),
        ][-20:]

        return ChatResponse(
            response=response,
            conversation_history=updated_history,
        )
