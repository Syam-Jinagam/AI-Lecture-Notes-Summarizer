import os
import json
import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Lecture Notes Summarizer",
    page_icon="📚",
    layout="wide"
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# --------------------------------------------------
# 2. PAGE DESIGN
# --------------------------------------------------

st.title("📚 AI Lecture Notes Summarizer")

st.markdown(
    """
    Turn lengthy lecture PDFs into useful study material.

    - 📘 **AI Summary:** Understand the main concepts.
    - 🔑 **Key Points:** Extract important definitions and ideas.
    - 🗂️ **Flashcards:** Practise questions and answers.
    """
)

st.divider()

# --------------------------------------------------
# 3. API CLIENT
# --------------------------------------------------

@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


# --------------------------------------------------
# 4. PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)

    if reader.is_encrypted:
        raise ValueError(
            "This PDF is password-protected. "
            "Please upload an accessible PDF."
        )

    page_texts = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts.append(
                    f"Page {page_number}:\n{page_text}"
                )
        except Exception:
            continue

    extracted_text = "\n\n".join(page_texts).strip()

    return extracted_text, len(reader.pages)


# --------------------------------------------------
# 5. CLEAN AND VALIDATE AI JSON
# --------------------------------------------------

def parse_json_response(response_text):
    if not response_text:
        raise ValueError(
            "The AI returned an empty response. "
            "Check your API access and try again."
        )

    # Remove accidental Markdown code fences
    cleaned = response_text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting a JSON object from the response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            raise ValueError(
                "The AI response was not valid JSON. "
                "Please generate the material again."
            )

        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError(
            "The AI response has an unexpected format."
        )

    return result


def normalize_result(result):
    """Normalize AI fields into predictable Python lists."""

    summary = result.get("summary", "")

    if isinstance(summary, (dict, list)):
        summary = json.dumps(
            summary, ensure_ascii=False, indent=2
        )

    if not isinstance(summary, str):
        summary = str(summary) if summary else ""

    key_points = result.get("key_points", [])

    if isinstance(key_points, str):
        key_points = [
            line.strip().lstrip("•-*0123456789.) ")
            for line in key_points.splitlines()
            if line.strip()
        ]

    if not isinstance(key_points, list):
        key_points = []

    normalized_points = []

    for point in key_points:
        if isinstance(point, dict):
            point = (
                point.get("point")
                or point.get("text")
                or point.get("concept")
                or ""
            )

        if isinstance(point, str) and point.strip():
            normalized_points.append(point.strip())

    flashcards = result.get("flashcards", [])

    if isinstance(flashcards, dict):
        flashcards = flashcards.get("flashcards", [])

    if isinstance(flashcards, str):
        try:
            flashcards = json.loads(flashcards)
        except json.JSONDecodeError:
            flashcards = []

    if not isinstance(flashcards, list):
        flashcards = []

    normalized_cards = []

    for card in flashcards:
        if not isinstance(card, dict):
            continue

        question = (
            card.get("question")
            or card.get("front")
            or card.get("q")
            or ""
        )

        answer = (
            card.get("answer")
            or card.get("back")
            or card.get("a")
            or ""
        )

        if isinstance(question, str) and isinstance(answer, str):
            if question.strip() and answer.strip():
                normalized_cards.append({
                    "question": question.strip(),
                    "answer": answer.strip()
                })

    return {
        "summary": summary.strip(),
        "key_points": normalized_points,
        "flashcards": normalized_cards
    }


# --------------------------------------------------
# 6. AI GENERATION
# --------------------------------------------------

def generate_study_material(
    text,
    flashcard_count,
    client
):
    # Keep the first version manageable.
    # Long documents should later be processed in chunks.
    max_characters = 40000
    source_text = text[:max_characters]

    prompt = f"""
You are an expert educational tutor and lecture-notes assistant.

Carefully analyse the lecture text below.

Return ONLY one valid JSON object with these exact keys:

{{
  "summary": "A detailed and understandable summary",
  "key_points": [
    "First important concept with explanation",
    "Second important concept with explanation"
  ],
  "flashcards": [
    {{
      "question": "A clear question about the lecture",
      "answer": "A correct and useful answer"
    }}
  ]
}}

STRICT REQUIREMENTS:

SUMMARY:
- Explain the main concepts in simple English.
- Include important definitions, principles, processes,
  and examples found in the lecture.
- Do not write only a few generic sentences.

KEY POINTS:
- Generate 8 to 12 meaningful key points.
- Explain each point in at least one useful sentence.
- Cover different topics from the provided lecture.
- Avoid repeating the same concept.

FLASHCARDS:
- Generate exactly {flashcard_count} flashcards if the source
  contains enough distinct information.
- Each card must have a non-empty question and answer.
- Test definitions, concepts, differences, applications,
  and examples where available.
- Do not repeat questions.
- Do not make up facts.

OUTPUT RULES:
- All three fields must contain actual content.
- key_points must be a JSON array of strings.
- flashcards must be a JSON array of objects.
- Do not return Markdown code fences.
- Do not include explanations outside the JSON.
- Use only information supported by the lecture.
- If a topic is not covered, do not invent information.

LECTURE TEXT:
{source_text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    result = normalize_result(
        parse_json_response(response.text)
    )

    # If the model omitted sections, make one focused retry.
    missing = []

    if not result["summary"]:
        missing.append("summary")

    if not result["key_points"]:
        missing.append("key_points")

    if not result["flashcards"]:
        missing.append("flashcards")

    if missing:
        retry_prompt = f"""
Create ONLY the missing fields listed here:
{", ".join(missing)}

Return a valid JSON object containing those fields.

Requirements:
- summary: a detailed, clear explanation of the lecture.
- key_points: an array of 8 to 12 explanatory strings.
- flashcards: an array of {flashcard_count} objects with
  non-empty "question" and "answer" fields.
- Use only the lecture text. Do not invent facts.
- Do not return empty fields or Markdown code fences.

LECTURE TEXT:
{source_text}
"""

        retry_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=retry_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )

        retry_result = normalize_result(
            parse_json_response(retry_response.text)
        )

        for field in missing:
            if retry_result.get(field):
                result[field] = retry_result[field]

    return result, len(text), len(source_text)


# --------------------------------------------------
# 7. DOWNLOAD HELPER
# --------------------------------------------------

def create_flashcard_text(cards):
    return "\n\n".join(
        f"Question: {card['question']}\n"
        f"Answer: {card['answer']}"
        for card in cards
    )


# --------------------------------------------------
# 8. API KEY VALIDATION
# --------------------------------------------------

if not API_KEY:
    st.error(
        "GEMINI_API_KEY was not found. "
        "Add it to your .env file and restart Streamlit."
    )

    st.code(
        "GEMINI_API_KEY=your_actual_api_key",
        language="text"
    )

    st.stop()

try:
    client = get_client(API_KEY)
except Exception as error:
    st.error(f"Could not initialize the AI client: {error}")
    st.stop()


# --------------------------------------------------
# 9. PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your lecture notes (PDF)",
    type=["pdf"],
    help="Start with a small PDF containing selectable text."
)

col1, col2 = st.columns(2)

with col1:
    flashcard_count = st.selectbox(
        "Number of flashcards",
        [5, 10, 15, 20],
        index=1
    )

with col2:
    st.info(
        "The current version extracts selectable PDF text. "
        "Scanned PDFs require OCR."
    )


# --------------------------------------------------
# 10. PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(f"Selected PDF: {uploaded_file.name}")

    if st.button(
        "✨ Generate Study Material",
        type="primary",
        use_container_width=True
    ):

        # Clear previous results before processing a new file.
        st.session_state.pop("study_result", None)
        st.session_state.pop("show_answer", None)
        st.session_state.pop("card_index", None)

        try:
            with st.spinner("📖 Extracting PDF text..."):
                text, page_count = extract_text(uploaded_file)

            if not text:
                st.error(
                    "No selectable text was found in this PDF. "
                    "It may contain scanned images. OCR is required "
                    "to read the text in those pages."
                )
                st.stop()

            if len(text.strip()) < 100:
                st.warning(
                    "Very little text was extracted. "
                    "Try another PDF or check whether it is scanned."
                )
                st.stop()

            st.write(f"**Pages detected:** {page_count}")

            if len(text) > 40000:
                st.warning(
                    "This PDF contains more text than the current "
                    "version sends to the AI. Only the first 40,000 "
                    "characters are processed. Chunking is needed "
                    "to cover the complete document."
                )

            with st.spinner(
                "🤖 AI is generating your summary, key points, "
                "and flashcards. Please wait..."
            ):
                result, total_chars, processed_chars = (
                    generate_study_material(
                        text,
                        flashcard_count,
                        client
                    )
                )

            st.session_state["study_result"] = result
            st.session_state["source_name"] = uploaded_file.name
            st.session_state["source_pages"] = page_count
            st.session_state["processed_chars"] = processed_chars

            st.success("Study material generation finished!")

        except Exception as error:
            st.error(
                "Unable to generate study material. "
                "Check your internet connection, API key, model "
                "availability, API quota, and PDF. "
                f"Technical details: {error}"
            )


# --------------------------------------------------
# 11. DISPLAY GENERATED RESULTS
# --------------------------------------------------

if "study_result" in st.session_state:

    result = st.session_state["study_result"]

    summary = result.get("summary", "")
    key_points = result.get("key_points", [])
    flashcards = result.get("flashcards", [])

    st.divider()
    st.header("🎓 Your Generated Study Material")

    st.caption(
        f"Source: {st.session_state.get('source_name', 'PDF')} "
        f"| Pages: {st.session_state.get('source_pages', 'N/A')}"
    )

    summary_tab, points_tab, cards_tab = st.tabs(
        [
            "📘 Summary",
            "🔑 Key Points",
            "🗂️ Flashcards"
        ]
    )

    # ----------------------------------------------
    # SUMMARY TAB
    # ----------------------------------------------

    with summary_tab:

        st.subheader("Lecture Summary")

        if summary:
            st.markdown(summary)

            st.download_button(
                label="⬇️ Download Summary",
                data=summary,
                file_name="lecture_summary.txt",
                mime="text/plain",
                key="download_summary"
            )
        else:
            st.error(
                "The AI did not return a summary. "
                "Try generating the material again."
            )

    # ----------------------------------------------
    # KEY POINTS TAB
    # ----------------------------------------------

    with points_tab:

        st.subheader("Important Concepts")

        if key_points:

            st.success(
                f"{len(key_points)} key points generated"
            )

            for index, point in enumerate(
                key_points, start=1
            ):
                st.markdown(f"**{index}. {point}**")

                if index < len(key_points):
                    st.divider()

            points_text = "\n\n".join(
                f"{index}. {point}"
                for index, point in enumerate(
                    key_points, start=1
                )
            )

            st.download_button(
                label="⬇️ Download Key Points",
                data=points_text,
                file_name="lecture_key_points.txt",
                mime="text/plain",
                key="download_key_points"
            )

        else:
            st.error(
                "No key points were returned by the AI. "
                "Check Debug Information below."
            )

    # ----------------------------------------------
    # FLASHCARDS TAB
    # ----------------------------------------------

    with cards_tab:

        st.subheader("Test Your Knowledge")

        if flashcards:

            total_cards = len(flashcards)

            if "card_index" not in st.session_state:
                st.session_state["card_index"] = 0

            if "show_answer" not in st.session_state:
                st.session_state["show_answer"] = False

            # Ensure the selected index is valid
            index = (
                st.session_state["card_index"] % total_cards
            )

            card = flashcards[index]

            st.progress(
                (index + 1) / total_cards,
                text=f"Flashcard {index + 1} of {total_cards}"
            )

            st.markdown(
                """
                <div style="
                    padding: 24px;
                    border: 1px solid #808080;
                    border-radius: 14px;
                    text-align: center;
                ">
                    <p style="font-size: 14px;">
                        QUESTION
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f"### {card['question']}")

            if st.button(
                "👀 Show Answer",
                use_container_width=True,
                key=f"show_answer_{index}"
            ):
                st.session_state["show_answer"] = True

            if st.session_state["show_answer"]:
                st.success("**Answer**")
                st.write(card["answer"])

            previous_col, next_col = st.columns(2)

            with previous_col:
                if st.button(
                    "⬅️ Previous",
                    use_container_width=True,
                    key="previous_card"
                ):
                    st.session_state["card_index"] = (
                        index - 1
                    ) % total_cards

                    st.session_state["show_answer"] = False
                    st.rerun()

            with next_col:
                if st.button(
                    "Next ➡️",
                    use_container_width=True,
                    key="next_card"
                ):
                    st.session_state["card_index"] = (
                        index + 1
                    ) % total_cards

                    st.session_state["show_answer"] = False
                    st.rerun()

            st.divider()

            with st.expander("View all flashcards"):
                for number, item in enumerate(
                    flashcards, start=1
                ):
                    st.markdown(
                        f"**{number}. {item['question']}**"
                    )
                    st.write(f"Answer: {item['answer']}")
                    st.divider()

            st.download_button(
                label="⬇️ Download All Flashcards",
                data=create_flashcard_text(flashcards),
                file_name="lecture_flashcards.txt",
                mime="text/plain",
                key="download_flashcards"
            )

        else:
            st.error(
                "No valid flashcards were generated. "
                "Try again or inspect the AI response below."
            )

    # ----------------------------------------------
    # DEBUG INFORMATION
    # ----------------------------------------------

    with st.expander("🔧 Debug Information"):

        st.write("**Returned fields:**")
        st.write(list(result.keys()))

        st.write("**Summary available:**", bool(summary))

        st.write("**Key points count:**", len(key_points))

        st.write("**Flashcards count:**", len(flashcards))

        st.write(
            "**Characters processed:**",
            st.session_state.get("processed_chars", "N/A")
        )

        st.write("**Raw AI result:**")
        st.json(result)
        st.balloons()

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Lecture Notes Summarizer | "
    "Built with Python, Streamlit, PyPDF2, and Gemini AI"
)

