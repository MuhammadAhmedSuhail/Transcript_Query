import streamlit as st
import os
import google.generativeai as genai

global transcript 

transcript = """
Doctor (D): Good morning, how are you feeling today?

Patient (P): Good morning, Doctor. I've been feeling very anxious and stressed
lately.

D: I'm sorry to hear that. Can you describe your symptoms in more detail?

P: I've been having trouble sleeping, my heart races for no reason, and I often feel
like I'm on edge. I also feel exhausted all the time.

D: It sounds like you might be experiencing symptoms of Generalized Anxiety
Disorder (GAD). Have you experienced these symptoms before?

P: Yes, I've had anxiety for a few years, but it's gotten worse recently.

D: I understand. Based on your symptoms and history, I'm diagnosing you with
Generalized Anxiety Disorder. We'll need to address this with a combination of
medication, therapy, and lifestyle changes. Does that sound okay to you?

P: Yes, I just want to feel better.

D: For medication, I'm going to prescribe you an SSRI (Selective Serotonin
Reuptake Inhibitor) called Sertraline. This should help manage your anxiety
symptoms. It's important to take it as prescribed and be patient, as it may take a
few weeks to see the full effects.

P: Okay, I can do that.

D: In addition to the medication, I'd like you to try some cognitive-behavioral
therapy (CBT). This type of therapy can help you identify and change negative
thought patterns and behaviors. I'll refer you to a therapist who specializes in
CBT.

P: That sounds helpful. I've heard of CBT before.

D: Great. Now, let's talk about some exercises and lifestyle changes. Regular
physical exercise can be very beneficial for reducing anxiety. Aim for at least 30
minutes of moderate exercise, like walking or yoga, most days of the week.

P: I can try to incorporate that into my routine.

D: Good. Also, practicing mindfulness or meditation daily can help reduce stress.
There are many apps and online resources that can guide you through these
practices.

P: I've never tried meditation, but I'm willing to give it a go.

D: Excellent. Finally, let's discuss some precautions. Avoid caffeine and alcohol as
they can worsen anxiety symptoms. Make sure to get enough sleep, and try to
maintain a regular sleep schedule.

P: I do drink a lot of coffee. I'll try to cut back.

D: It's all about making small, sustainable changes. We will monitor your progress
closely and adjust the treatment plan as needed. Do you have any questions or
concerns?

P: Not at the moment. Thank you, Doctor.

D: You're welcome. Remember, you're not alone in this, and we're here to support
you. I'll see you in two weeks for a follow-up.

P: Thank you, Doctor. I appreciate it.
"""

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 0.5,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ],
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction=f'''
    Here is a transcription between a patient and a doctor.
    You will be asked a query regarding this transcription if the query is not related to this transcript then reply with 
    "This query is out of scope. Make sure your question is relevant to the transcription."
    
    Transcription = {transcript}

  ''')

chat_session = model.start_chat()


def load_transcript(file):
    if file is not None:
        return file.read().decode('utf-8')
    else:
        st.info("Using sample transcript from provided PDF.")
        return transcript

def main():
    st.title("Transcript Query System")

    # File upload
    uploaded_file = st.file_uploader("Upload a transcript (TXT file)", type="txt")

    # Load transcript
    loaded_transcript = load_transcript(uploaded_file)

    if st.button("Load Transcript"):
        st.text_area("Transcript", value=loaded_transcript, height=300)

        transcript = loaded_transcript
        
    # Query input
    query = st.text_input("Ask a question about the transcript")

    if st.button("Query"):
        # if query:
        response = chat_session.send_message(query)

        st.write(response.text)
        
if __name__ == "__main__":
    main()
