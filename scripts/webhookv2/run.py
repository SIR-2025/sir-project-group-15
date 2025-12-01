from flask import Flask

app = Flask(__name__)

import routes # Ensure routes are imported so that they are registered
        
if __name__ == "__main__":
    app.run(port=8080)