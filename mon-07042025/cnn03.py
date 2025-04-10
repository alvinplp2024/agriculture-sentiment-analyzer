from flask import Flask, request

import joblib

app = Flask(__name__)

# Load sentiment model and vectorizer
sentiment_model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route("/", methods=["GET"])
def index():
    return '''
    <html>
      <head>
        <title>Sentiment Analyzer - Agriculture</title>
        <style>
          body {
            font-family: 'Segoe UI', sans-serif;
            background-image: url('https://images.unsplash.com/photo-1587049352859-01e6ddcd460a?auto=format&fit=crop&w=1400&q=80');
            background-size: cover;
            background-position: center;
            margin: 0;
            padding: 0;
            color: #fff;
          }
          .container {
            background-color: rgba(0, 0, 0, 0.6);
            margin: 100px auto;
            padding: 40px;
            border-radius: 15px;
            max-width: 600px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
          }
          textarea {
            width: 90%;
            height: 100px;
            margin: 15px 0;
            padding: 10px;
            font-size: 16px;
            border-radius: 10px;
            border: none;
          }
          input[type="submit"] {
            padding: 10px 20px;
            font-size: 18px;
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
          }
          h1 {
            margin-bottom: 10px;
            font-size: 32px;
          }
          img {
            max-width: 100%;
            border-radius: 10px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>🌾 ALVIN - DATA SCIENCE & AI</h1>
          <h1>🌾 Agriculture Sentiment Analyzer</h1>
          <p>Analyze farmers' sentiments about crops and planting</p>
          <img src="https://images.unsplash.com/photo-1611689941343-e2fd2f5ebf7b?auto=format&fit=crop&w=800&q=80" alt="🌾">
          <form action="/predict_text" method="post">
            <textarea name="user_text" placeholder="Type your message here..."></textarea><br>
            <input type="submit" value="Analyze Sentiment">
          </form>
        </div>
      </body>
    </html>
    '''

@app.route("/predict_text", methods=["POST"])
def predict_text():
    try:
        text = request.form["user_text"]
        if not text.strip():
            return "Please enter some text.", 400

        X = vectorizer.transform([text])
        prediction = sentiment_model.predict(X)[0]
        result = f"The sentiment of the text is: <strong>{prediction.upper()}</strong>"

        return f'''
        <html>
          <head>
            <title>Sentiment Result</title>
            <style>
              body {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #eef2e6;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
              }}
              .result-container {{
                background-color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
              }}
              h2 {{
                color: #2e7d32;
              }}
              a {{
                margin-top: 20px;
                display: inline-block;
                text-decoration: none;
                color: #4caf50;
                font-weight: bold;
              }}
            </style>
          </head>
          <body>
            <div class="result-container">
              <h2>📊 Sentiment Analysis Result</h2>
              <p>{result}</p>
              <a href="/">⬅️ Analyze Another Text</a>
            </div>
          </body>
        </html>
        '''
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
