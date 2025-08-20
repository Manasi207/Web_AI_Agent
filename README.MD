# Install in Project Folder (💯 Clean & Professional)

🔁 Step-by-step:
Go to your project folder:

cd path/to/your/project

Create a virtual environment:

python -m venv venv

Activate the venv:

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Playwright in that environment:

pip install playwright
playwright install

✅ Now playwright is only tied to this specific project.

🎁 Bonus: Save Dependencies
To make your project portable and patent-ready, save all packages:

pip freeze > requirements.txt

Later, any dev can run:

pip install -r requirements.txt
