# PipPiko
PipPiko is a local AI shopping chatbot built with flask. It asks you whether you want to buy or browse, then walks through your budget, brand preference, and intended use one question at a time before suggesting products. Uses a lightweight local language model, to have a natural, one-question-at-a-time conversation with users.

Pip first tries to understand what the user is actually looking for by gathering key shopping requirements such as product category, budget, brand preference, and intended use.

Once enough information has been collected, Pip provides product suggestions based on the extracted requirements.

📁 Project Structure
pip_shopping_assistant/
│
├── app.py
│   └── Main Flask application
│
├── requirements.txt
│   └── Python dependencies
│
└── templates/
    └── index.html
        └── Frontend chat interface

⚙️ Requirements
Before installing Pip, make sure you have:

Python 3
pip
An internet connection for the initial model download
Sufficient RAM/storage to run the local model

🚀 Installation
1. Clone the Repository
Clone the repository using Git:
git clone <your-repository-url>

Then navigate into the project:
cd pip_shopping_assistant

Alternatively, download the repository as a ZIP file and extract it.

2.Install Dependencies
Install all required Python packages:

pip install -r requirements.txt

The requirements file installs the necessary dependencies, including:
flask
transformers
accelerate
torch

3. Running the Application
Start PipPiko using:

python app.py

When launched, Pip will:

Start loading the Qwen model in a background thread.
Start the Flask web server.
Run the server on port 5000.
Automatically open the application in your browser.

The application will be available at:

http://localhost:5000

If the browser does not open automatically, manually navigate to the address above.


