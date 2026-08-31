# PipPiko
PipPiko is a local AI shopping chatbot built with flask. It asks you whether you want to buy or browse, then walks through your budget, brand preference, and intended use one question at a time before suggesting products. Uses a lightweight local language model, to have a natural, one-question-at-a-time conversation with users.

Pip first tries to understand what the user is actually looking for by gathering key shopping requirements such as product category, budget, brand preference, and intended use.

Once enough information has been collected, Pip provides product suggestions based on the extracted requirements.

🎯 Project Goal
The goal of PipPiko is to explore how conversational AI can improve the online shopping experience even better!

Traditional shopping search often requires users to know exactly what they want before searching. But what if AI helps to improve search by understanding whether a user is buying or browsing?

PipPiko takes a different approach:

User's Intent
     ↓
Conversation
     ↓
Requirements
     ↓
Product Discovery
     ↓
Recommendations

By asking targeted questions, PipPiko aims to make product discovery more natural, accessible and efficient.

⚙️ Requirements to Install
Before installing Pip, make sure you have:

Python 3
pip
An internet connection for the initial model download
Sufficient RAM/storage to run the local model

🚀 Installation
1. Clone the Repository

Clone the repository using Git:

git clone <your-repository-url>

Navigate into the project directory:

cd pip_shopping_assistant

Alternatively, you can download the repository as a ZIP file and extract it.

2. Install Dependencies

Install all required Python packages using:

pip install -r requirements.txt

The requirements.txt file installs the necessary dependencies, including:

flask
transformers
accelerate
torch


▶️ Running the Application

Start PipPiko using:

python app.py

When launched, PipPiko will:

Load the local Qwen language model in a background thread.
Start the Flask web server.
Run the application on port 5000.
Automatically open the application in your browser.

The application will be available at:

http://localhost:5000

If the browser does not open automatically, manually navigate to the address above.

🛒 Using PipPiko!!
Once the model has loaded, enter what you are looking for into the chat interface.

For example:
I need a laptop for university.

Pip will then ask follow-up questions to determine your requirements.

Typical requirements include:

    Product Category
    What are you looking for?
    
        Example: A laptop
        
    What's your budget?
    
        Example: Around $1,500
        
    Brand Preference
    
    Do you have a preferred brand?
    
        Example: ASUS
        
    Intended Use
    
    What will you mainly use it for?
    
        Example: Gaming and university work

After enough information has been collected, Pip returns:

SEARCH_READY

The interface then displays product suggestions from the mock catalogue.



Current Limitations
1. Product recommendations currently come from a mock catalogue.
2. Product prices and availability are not live.
3. There is no connection to real-time e-commerce platforms.

Future Improvements
1. Real-time product prices
2. Persistent conversation history
3. Personalized user profiles

