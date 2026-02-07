
Project Overview: The AI English Teacher is a full-stack local AI application that transforms a Large Language Model (like Gemma:2b) into a patient and encouraging language instructor. It focuses on real-time feedback and pedagogical engagement.

Key Pedagogical Features:

Error Correction: Automatically identifies and gently corrects grammar, spelling, and punctuation mistakes with detailed explanations.

Fluency Suggestions: Offers alternative phrasing to help students sound more natural and fluent.

Vocabulary Building: Explains complex words and provides contextual examples for better retention.

Interactive Dialogue: Proactively asks questions to keep the learner engaged and practicing.

Technical Implementation:

System Prompt Engineering: Features a highly tuned system persona that dictates the AI's behavior as a friendly and supportive teacher.

Streaming Architecture: Implements a streaming response system using requests and json to provide immediate feedback as the AI "thinks".

Robust Backend: Includes a retry mechanism (MAX_RETRIES), comprehensive logging, and asynchronous task handling with asyncio.

User Interface: Built with Gradio, featuring a Dracula-themed dashboard, temperature/token sliders for model control, and a "Stop" button for real-time generation control.

Technical Stack:

Frontend: Gradio.

Local AI Engine: Ollama API.

Language: Python.
