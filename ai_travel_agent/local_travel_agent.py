from textwrap import dedent
import streamlit as st
import re
from icalendar import Calendar, Event
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def generate_ics_content(plan_text:str, start_date: datetime = None) -> bytes:
    """
        Generate an ICS calendar file from a travel itinerary text.

        Args:
            plan_text: The travel itinerary text
            start_date: Optional start date for the itinerary (defaults to today)

        Returns:
            bytes: The ICS file content as bytes
        """
    cal = Calendar()
    cal.add('prodid','-//AI Travel Planner//github.com//' )
    cal.add('version', '2.0')

    if start_date is None:
        start_date = datetime.today()

    # Split the plan into days
    day_pattern = re.compile(r'Day (\d+)[:\s]+(.*?)(?=Day \d+|$)', re.DOTALL)
    days = day_pattern.findall(plan_text)

    if not days: # If no day pattern found, create a single all-day event with the entire content
        event = Event()
        event.add('summary', "Travel Itinerary")
        event.add('description', plan_text)
        event.add('dtstart', start_date.date())
        event.add('dtend', start_date.date())
        event.add("dtstamp", datetime.now())
        cal.add_component(event)  
    else:
        # Process each day
        for day_num, day_content in days:
            day_num = int(day_num)
            current_date = start_date + timedelta(days=day_num - 1)
            
            # Create a single event for the entire day
            event = Event()
            event.add('summary', f"Day {day_num} Itinerary")
            event.add('description', day_content.strip())
            
            # Make it an all-day event
            event.add('dtstart', current_date.date())
            event.add('dtend', current_date.date())
            event.add("dtstamp", datetime.now())
            cal.add_component(event)

    return cal.to_ical()


# Set up the Streamlit app
st.title("AI Book Planner (Hugging Face Local)")
st.caption("Plan your next reading journey with AI Book Planner by researching books and creating a personalized reading plan using a local Hugging Face model")

# Initialize session state to store the generated itinerary
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None

# HF model selection and lazy initialization
default_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = st.text_input("Hugging Face model id", value=default_model_id)

@st.cache_resource(show_spinner=False)
def load_generator(selected_model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(selected_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        selected_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto"
    )
    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return generator

generator = load_generator(model_id)

# Input fields for the user's destination and the number of days they want to travel for
destination = st.text_input("What theme/genre or author are you planning to read?")
num_days = st.number_input("How many days is your reading plan?", min_value=1, max_value=30, value=7)
st.caption("Note: First-time model load downloads weights and may take a few minutes.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Reading Plan"):
        with st.spinner("Creating your personalized reading plan..."):
            system_prompt = "You are an expert reading planner."
            user_prompt = f"""
            Theme/Genre/Author: {destination}
            Duration: {num_days} days

            Generate a detailed reading plan including:
            - Book recommendations (title, author)
            - Suggested order with rationale
            - Daily reading goals or chapters
            - Brief synopsis per book
            - Any prerequisites or complementary resources
            Format the plan by Day 1, Day 2, ... for calendar export.
            """

            # Use chat template if available (for chat-tuned models)
            if hasattr(generator.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.strip()},
                ]
                formatted = generator.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = f"{system_prompt}\n\n{user_prompt.strip()}"

            gen_kwargs = {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "return_full_text": False,
            }
            outputs = generator(formatted, **gen_kwargs)
            text = outputs[0]["generated_text"] if isinstance(outputs, list) else str(outputs)
            st.session_state.itinerary = text
            st.write(text)

# Only show download button if there's an itinerary
with col2:
    if st.session_state.itinerary:
        try:
            # Generate the ICS file
            ics_content = generate_ics_content(str(st.session_state.itinerary))

            # Provide the file for download
            st.download_button(
                label="Download Reading Plan as Calendar (.ics)",
                data=ics_content,
                file_name="reading_plan.ics",
                mime="text/calendar"
            )
        except Exception as e:
            st.error(f"Failed to create calendar file: {e}")
            with st.expander("Show generated plan text"):
                st.write(st.session_state.itinerary)