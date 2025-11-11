# 🧠 NLP to NoSQL (MongoDB) Converter
🚀 Overview
This project converts natural language queries (English sentences) into MongoDB NoSQL queries using FastAPI.
It acts as an intelligent interface between users and a NoSQL database — allowing users to query the database without writing MongoDB syntax.
Example:

Input: “Show me all users who joined after 2021”
Output: { "query": { "join_date": { "$gt": "2021-01-01" } } }


🏗️ Project Structure
📂 NOSQL
│
├── app_server.py        # FastAPI server handling API requests

├── main.py              # Entry point to start the server

├── No_Sql.py            # NLP processor that generates MongoDB queries

├── nosql.ipynb          # Notebook for testing query generation

├── requirements.txt     # Python dependencies

├── req.txt              # Alternative dependency file (optional)

└── venv/                # Virtual environment (ignored in version control)


🧩 Key Features


🧠 NLP-powered query translation – Converts English queries into MongoDB syntax.


⚡ FastAPI backend – High-performance REST API for NLP processing.


🕒 Async query handling – Supports async execution and timeouts for large datasets.


🧾 Logging system – Tracks API requests, responses, and processing time.


🧰 Configurable timeouts – Adjustable limits for query and DB operation duration.



🧠 Core Components
1. app_server.py


Implements FastAPI routes.


Handles request parsing and error handling.


Imports NLP functions from No_Sql.py.


Includes timeout handling and logging configuration.


2. No_Sql.py


Contains the NLP processor that parses user input.


Generates MongoDB-compatible queries.


Includes helper function generate_mongo_query_with_timeout() to handle execution safely.


3. main.py


Entry script to start the FastAPI app.


Typically runs the server using:
uvicorn app_server:app --reload

⚙️ Setup and Installation

1. Open a terminal in VS Code
(Menu → Terminal → New Terminal)

2. Create a virtual environment

python -m venv venv

3.Activate the environment

On Windows:

venv\Scripts\activate


On Mac/Linux:

source venv/bin/activate


4. Install dependencies


pip install -r requirements.txt

▶️ Run the App
uvicorn app_server:app --reload


