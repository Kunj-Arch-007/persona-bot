# 🤖 Personal Details Collection Bot

A conversational **Streamlit web app** powered by **LangGraph** and **Ollama LLM**, designed to collect and summarize a user's personal details in an interactive, chat-like flow.

---

## 📌 Features

* Interactive bot that asks 7 structured questions:

  1. Full Name
  2. Age
  3. Gender
  4. Email Address
  5. Mobile Number
  6. Country
  7. Profession / Occupation
* Maintains conversation history
* Validates responses and retries on errors
* Generates a summary at the end
* Built with **Streamlit** for easy deployment and clean UI
* Uses **LangGraph** to manage conversational state
* Powered by **Ollama LLM** (`llama3.1:8b-instruct-q4_K_M`)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Kunj-Arch-007/persona-bot.git
cd persona-bot
```

### 2. Create & activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run web.py
```

By default, Streamlit runs on [http://localhost:8501](http://localhost:8501).

---

## 📂 Project Structure

```
.
├── web.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```

---

## ⚙️ Configuration

* **Model**: The app uses the `OllamaLLM` wrapper with `llama3.1:8b-instruct-q4_K_M`.
* Ensure you have **Ollama** installed and the model pulled locally.

👉 [Get Ollama here](https://ollama.ai)

---

## 🛠️ Tech Stack

* **[Streamlit](https://streamlit.io/)** – UI framework
* **[LangGraph](https://github.com/langchain-ai/langgraph)** – State management
* **[Ollama](https://ollama.ai/)** – Local LLM engine

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Streamlit](https://streamlit.io/)
* [Ollama](https://ollama.ai/)
