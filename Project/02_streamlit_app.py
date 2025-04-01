import json
import streamlit as st
from operator import rshift
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = (
            text.replace("```", "")
                .replace("json", "")
                .replace(", ]", "]")
                .replace(", }", "}")
        )
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(page_title="QuizGPT", page_icon="❓")
st.title("QuizGPT")

# Sidebar config
with st.sidebar:
    st.markdown("### 🔑 OpenAI API Key")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    st.markdown("### GitHub")
    st.markdown("[View this project on GitHub](https://github.com/kyuwoncho/GPT)")

    source_choice = st.selectbox("Choose your source:", ("File", "Wikipedia Article"))
    difficulty = st.selectbox("Select difficulty:", ("Easy", "Hard"))

    docs = None
    if source_choice == "File":
        file = st.file_uploader("Upload a file (.pdf, .txt, .docx)", type=["pdf", "txt", "docx"])
        if file:
            file_content = file.read()
            file_path = f"./.cache/quiz_files/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file_content)
            splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600, chunk_overlap=100)
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load_and_split(text_splitter=splitter)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            docs = retriever.get_relevant_documents(topic)

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

questions_prompt = ChatPromptTemplate.from_messages([
    ("system",
     f"""
     You are a helpful assistant that is role playing as a teacher.
     Based ONLY on the following context, create 10 questions to test the user's knowledge.
     Questions should have 4 options. One correct (marked with (o)) and three incorrect.
     Make the questions {'difficult' if difficulty == 'Hard' else 'easy'}.
     
     Context: {{context}}
     """)
])

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a powerful formatting algorithm.
     Format the following questions into valid JSON format.
     Answers marked with (o) are correct.

     Questions: {context}
     """)
])

formatting_chain = formatting_prompt | llm

if docs:
    st.session_state.response = None
    if st.button("Generate Quiz") or "response" not in st.session_state:
        with st.spinner("Generating quiz..."):
            chain = {"context": questions_chain} | formatting_chain | output_parser
            st.session_state.response = chain.invoke(docs)

if "response" in st.session_state and st.session_state.response:
    response = st.session_state.response
    score = 0
    total = len(response["questions"])

    with st.form("quiz_form"):
        user_answers = []
        for idx, question in enumerate(response["questions"]):
            st.write(f"**{idx+1}. {question['question']}**")
            options = [a["answer"] for a in question["answers"]]
            selected = st.radio("", options, key=f"q{idx}")
            user_answers.append((selected, question))

        submitted = st.form_submit_button("Submit Quiz")

        if submitted:
            for selected, question in user_answers:
                for ans in question["answers"]:
                    if ans["answer"] == selected and ans["correct"]:
                        score += 1
            st.write(f"### You scored {score} out of {total}.")

            if score == total:
                st.balloons()
                st.success("🎉 Perfect score! Great job!🎉 ")
            else:
                if st.button("Try Again"):
                    st.session_state.response = None
