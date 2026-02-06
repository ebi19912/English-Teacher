import gradio as gr
import requests
import json
import logging
from typing import Generator, Tuple, List, Any
from time import sleep

# --- Basic Configuration ---
logging.basicConfig(
    filename='english_teacher_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Ollama Settings ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "gemma:2b"  # Or any other model you prefer
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT = 45
MAX_RETRIES = 3

# --- System Prompt for the English Teacher Persona ---
# This prompt defines the AI's role and behavior.
SYSTEM_PROMPT = (
    "You are a friendly, patient, and encouraging English language teacher. Your goal is to help me practice and improve my English skills. "
    "When I write something, please do the following:\n"
    "1.  **Correct my mistakes:** Gently correct any grammar, spelling, or punctuation errors. Explain *why* it was a mistake if the reason isn't obvious.\n"
    "2.  **Suggest improvements:** Offer alternative phrasing to make my sentences sound more natural and fluent.\n"
    "3.  **Engage in conversation:** Ask me questions to keep the conversation going and encourage me to practice more.\n"
    "4.  **Explain vocabulary:** If I use a word incorrectly or if there's a better word, explain the meaning and provide an example.\n"
    "5.  **Maintain a positive tone:** Always be supportive and encouraging. Start your responses with a friendly greeting."
)

# Global flag to control the generation process
should_stop = False

def chat_with_ollama(message: str, history: List[Tuple[str, str]], model: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
    """
    Main function to handle the chat generation with the English teacher prompt.
    Checks the 'should_stop' flag during response generation.
    """
    global should_stop
    should_stop = False  # Reset the flag at the beginning of a new request

    try:
        # Build the full conversation history for context
        conversation = "\n".join([f"User: {text}" if speaker == "user" else f"Assistant: {text}" for speaker, text in history])
        full_prompt = f"{SYSTEM_PROMPT}\n\n{conversation}\nUser: {message}\nAssistant:"

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        logging.info(f"Sending request to model {model} with temperature {temperature} and max_tokens {max_tokens}")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    OLLAMA_ENDPOINT,
                    json=payload,
                    stream=True,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                complete_response = ""
                for line in response.iter_lines():
                    if should_stop:
                        yield "⏹️ Generation stopped by user."
                        should_stop = False
                        return

                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if "response" in data:
                                complete_response += data["response"]
                                yield complete_response
                        except json.JSONDecodeError:
                            logging.warning("JSON decoding failed for a line from the response stream.")
                            continue
                logging.info("Response successfully received.")
                return

            except requests.exceptions.RequestException as e:
                error_text = f"⛔ Connection Error: {str(e)}"
                logging.error(error_text)
                if attempt < MAX_RETRIES - 1:
                    yield f"{error_text}\nRetrying..."
                    sleep(2)
                else:
                    yield "❌ All retries failed. Please check your connection to Ollama and try again later."
                    return
    except Exception as e:
        sys_error = f"❌ An unexpected system error occurred: {str(e)}"
        logging.exception("An exception occurred in chat_with_ollama")
        yield sys_error

def stop_generation():
    """
    Sets the global flag to stop the current streaming operation.
    """
    global should_stop
    should_stop = True
    logging.info("Stop button pressed. Halting generation.")

def user_interaction(user_message, history, model, temperature, max_tokens):
    """
    Receives user message, adds it to history, and starts the response stream.
    Manages the visibility of Send/Stop buttons.
    """
    history = history or []
    history.append(("user", user_message))

    # Switch to Stop button and yield history to display user message
    yield gr.update(visible=False), gr.update(visible=True), history, ""

    # Stream the response from the assistant
    assistant_response = ""
    for response_part in chat_with_ollama(user_message, history, model, temperature, max_tokens):
        assistant_response = response_part
        # Update the chatbot interface with the streaming response
        history[-1] = ("assistant", assistant_response)
        yield gr.update(visible=False), gr.update(visible=True), history, ""

    # Switch back to Send button after generation is complete
    yield gr.update(visible=True), gr.update(visible=False), history, ""


def build_interface():
    """
    Creates the Gradio UI with a new theme, layout, and components.
    """
    with gr.Blocks(theme="gradio/dracula_soft", css="#chat-box { min-height: 500px; }") as demo:
        gr.Markdown("# 🇬🇧 English Language Teacher AI")
        gr.Markdown("Practice your English with a friendly AI assistant. Type a message to start the conversation.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    elem_id="chat-box",
                    bubble_full_width=False
                )
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        container=True,
                        scale=8
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)
                    stop_button = gr.Button("Stop", variant="stop", visible=False, scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                with gr.Accordion("Model & Parameters", open=False):
                    model_name_input = gr.Textbox(label="Ollama Model Name", value=DEFAULT_MODEL_NAME)
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.0, maximum=1.0, value=DEFAULT_TEMPERATURE, step=0.05,
                        info="Lower values are more predictable, higher are more creative."
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max New Tokens",
                        minimum=128, maximum=4096, value=DEFAULT_MAX_TOKENS, step=64,
                        info="Maximum number of tokens to generate."
                    )
        
        # State object to store the conversation history
        chat_history = gr.State([])

        # Event Listeners
        submit_action = text_input.submit(
            fn=user_interaction,
            inputs=[text_input, chat_history, model_name_input, temperature_slider, max_tokens_slider],
            outputs=[send_button, stop_button, chatbot, text_input],
        )

        send_button.click(
            fn=user_interaction,
            inputs=[text_input, chat_history, model_name_input, temperature_slider, max_tokens_slider],
            outputs=[send_button, stop_button, chatbot, text_input],
        )

        stop_button.click(
            fn=stop_generation,
            inputs=None,
            outputs=None,
            cancels=[submit_action] # This will cancel the running event
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            None,
            [send_button, stop_button],
            queue=False
        )

    return demo

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_port=7860, show_error=True)