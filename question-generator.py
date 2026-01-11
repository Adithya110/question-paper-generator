import re
import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import os

# ------------------- Page Setup -------------------
st.set_page_config(page_title="AI Exam Generator", page_icon="📝", layout="wide")
st.title("📝 AI Exam Paper Generator")
st.write("Upload syllabus → Analyze patterns → Generate exam paper")

hf_token = os.getenv("HF_TOKEN") 

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    # hf_token = st.text_input("Hugging Face Token", type="password")
    
    # Stable model for free tier academic reasoning
    model_id = "meta-llama/Llama-3.1-8B-Instruct" 
    
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    total_marks = st.selectbox("Exam Type", ["50 Marks", "75 Marks", "100 Marks"])

    if st.button("🔌 Test API Connection"):
        if not hf_token:
            st.error("Please enter a token first!")
        else:
            try:
                # 'provider="auto"' routes to any available free server
                client = InferenceClient(api_key=hf_token, provider="auto")
                client.chat_completion(
                    model=model_id,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                st.success(f"✅ Connection successful!")
            except Exception as e:
                st.error(f"❌ Connection Failed: {str(e)}")

# ------------------- Utilities -------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

def clean_text(text: str) -> str:
    text = re.sub(r"Page\s+\d+", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_topics(text: str, k: int = 12):
    """Filters reference/bibliography noise before embedding."""
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 30 and not l.strip().isdigit()]
    
    filtered_lines = []
    for line in lines:
        if any(word in line.lower() for word in ["text books", "references", "reference books", "bibliography"]):
            break
        filtered_lines.append(line)
    
    if not filtered_lines: return []
    
    embeddings = np.array(embed_model.encode(filtered_lines)).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    _, indices = index.search(centroid, k)
    return [filtered_lines[i] for i in indices[0]]

def exam_structure(marks: str) -> str:
    structures = {
        "50 Marks": "Section A: 10 MCQs (1M each), Section B: 4 Short Qs (5M each), Section C: 2 Long Qs (10M each)",
        "75 Marks": "Section A: 15 MCQs (1M each), Section B: 5 Short Qs (5M each), Section C: 2 Long Qs (15M each + 1 for 5M)",
        "100 Marks": "Section A: 20 MCQs (1M each), Section B: 6 Short Qs (5M each), Section C: 3 Long Qs (15M each + 1 for 5M)"
    }
    return structures.get(marks, "")

# ------------------- PDF Upload -------------------
uploaded = st.file_uploader("📄 Upload Syllabus (PDF)", type="pdf")
text_data = ""

if uploaded:
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    for page in doc:
        text_data += clean_text(page.get_text())
    st.success("✅ Syllabus Loaded")

# ------------------- Generate Exam -------------------
if st.button("📝 Generate Question Paper"):
    if not hf_token or not text_data:
        st.error("Missing Token or PDF data")
    else:
        topics_list = extract_topics(text_data)
        topics_summary = "\n".join([f"- {t}" for t in topics_list])

        system_instruction = (
            "You are a strict University Professor. Create a formal exam paper. "
            "Ignore all author names and book titles. Focus strictly on scientific concepts."
        )
        
        user_prompt = f"""
        Create a {total_marks} exam paper based on these topics:
        {topics_summary}
        
        STRUCTURE: {exam_structure(total_marks)}
        DIFFICULTY: {difficulty}
        
        STRICT RULES:
        1. NO diagrams, tables, or charts.
        2. NO author names or textbook citations.
        3. Clear headings for SECTION A, B, and C.
        """

        client = InferenceClient(api_key=hf_token, provider="auto")
        
        with st.spinner("Generating Paper..."):
            try:
                response = ""
                paper_placeholder = st.empty()
                
                # Streaming with stream=True
                stream = client.chat_completion(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1800,
                    temperature=0.2, 
                    stream=True,
                )
                
                for msg in stream:
                    # SAFETY CHECK: Ensure choices exists before indexing
                    if hasattr(msg, 'choices') and msg.choices:
                        token = msg.choices[0].delta.content
                        if token:
                            response += token
                            paper_placeholder.write(response)
                
                st.session_state["paper"] = response
                st.success("✅ Exam Paper Ready!")
                st.download_button("⬇️ Download Paper", response, "exam_paper.txt")
                
            except (IndexError, StopIteration):
                # Safely handle the end of stream or empty chunks
                pass
            except Exception as e:
                st.error(f"Generation Error: {e}")