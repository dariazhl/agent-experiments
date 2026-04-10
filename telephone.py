from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")

st.set_page_config(
    page_title="The LLM Telephone Game",
    page_icon="",
    layout="centered"
)

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.4,
        openai_api_key=api_key
    )

llm = get_llm()

STEPS = [
    ("Customer support agent", "You handle complaints. You care about tone and urgency. You ignore financial details."),
    ("Billing system", "You process financial transactions. You care about amounts and account status. You ignore emotional context."),
    ("Fraud detection", "You assess risk. You care about unusual patterns and account flags. You ignore refund amounts."),
    ("Refund processor", "You execute refunds. You care about order IDs and amounts. You ignore why the customer is upset."),
    ("Finance approval", "You approve transactions. You care about policy compliance and amounts. You ignore customer history."),
    ("CRM updater", "You update records. You care about status changes and account notes. You ignore transaction details."),
    ("Email dispatcher", "You send communications. You care about what to say to the customer. You ignore all internal processing details."),
]


def pass_through_agent(
    message: str,
    agent_name: str,
    agent_focus: str
) -> str:
    response = llm.invoke([
        SystemMessage(content=f"""
        You are the {agent_name} in an automated pipeline.
        
        Your role: {agent_focus}
        
        Rules you must follow strictly:
        1. Read the incoming message
        2. Extract ONLY what is relevant to your specific role
        3. Drop everything outside your domain — do not preserve it
        4. Write your output in maximum 15 words
        5. Write only what YOU will do next, from your role's perspective
        6. Do not repeat the full message — compress it through your lens only
        
        Your output becomes the next agent's only input.
        They will not see the original message.
        """),
        HumanMessage(content=f"Incoming message: {message}")
    ])
    return response.content.strip()


st.title("The Telephone Game")
st.caption(
    "A message passes through 7 agents. "
    "Each one filters it through its own role and drops everything else. "
    "Watch what arrives at the end."
)

st.divider()

original = st.text_input(
    "Original instruction:",
    value=(
        "Make it right for the customer — they've been waiting "
        "three weeks, they're furious, and they were overcharged "
        "150 DKK on a premium account."
    )
)

if st.button(
    "Pass through the pipeline",
    type="primary",
    use_container_width=True
):
    st.divider()
    st.markdown("### What each agent heard")
    st.markdown("")

    # Original
    st.markdown("**Human**")
    st.success(f'"{original}"')
    st.markdown(
        f"<div style='font-size:12px;color:gray;'>"
        f"{len(original.split())} words</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    current_message = original
    all_outputs = [original]

    for i, (step, focus) in enumerate(STEPS):
        with st.spinner(f"{step} processing..."):
            output = pass_through_agent(current_message, step, focus)

        word_count = len(output.split())
        original_count = len(original.split())
        pct_retained = round(word_count / original_count * 100)

        # Colour degrades as information is lost
        if pct_retained > 60:
            render = st.info
        elif pct_retained > 35:
            render = st.warning
        else:
            render = st.error

        st.markdown(f"**{i + 1}. {step}**")
        st.caption(f"*Role: {focus}*")
        render(f'"{output}"')
        st.markdown(
            f"<div style='font-size:12px;color:gray;'>"
            f"{word_count} words — {pct_retained}% of original retained</div>",
            unsafe_allow_html=True
        )
        st.markdown("")

        current_message = output
        all_outputs.append(output)

    st.divider()
    st.markdown("### Start vs finish")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**What the human said:**")
        st.success(f'"{original}"')
        st.caption(f"{len(original.split())} words")

    with col2:
        st.markdown("**What the last agent acted on:**")
        st.error(f'"{current_message}"')
        st.caption(f"{len(current_message.split())} words")

    final_retention = round(
        len(current_message.split()) / len(original.split()) * 100
    )

    st.divider()

    st.error(f"""
    **{100 - final_retention}% of the original context was lost.**

    """)
