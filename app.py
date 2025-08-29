from flask import Flask, request, jsonify, render_template_string
import requests

app = Flask(__name__)

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyDka3K_z-bY58Bh_IX9wYWnd1jSZYGobR4"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def generate_with_gemini(prompt):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        try:
            # Gemini 2.0 Flash response parsing may differ, adjust if needed
            return response.json()
        except Exception as e:
            return f"Error parsing Gemini response: {e}"
    else:
        return f"Gemini API Error: {response.text}"

@app.route('/generate_data', methods=['POST'])
def generate_data():
    data = request.json
    num_transactions = data.get("num_transactions", 10)
    num_products = data.get("num_products", 5)
    context = data.get("context", "grocery store")

    prompt = f"""
    Generate {num_transactions} unique transactions for frequent pattern mining.
    Each transaction should include 1-{num_products} products.
    Context: {context}.
    Output in strict JSON format like:
    [
      {{"transaction_id": 1, "items": ["apple", "milk", "bread"]}},
      {{"transaction_id": 2, "items": ["rice", "oil"]}}
    ]
    Ensure each run produces unique sets.
    """

    result = generate_with_gemini(prompt)

    try:
        # Try to convert response to JSON
        transactions = eval(result) if isinstance(result, str) else result
    except:
        return jsonify({"error": "Invalid JSON from Gemini", "raw": result}), 500

    return jsonify(transactions)


# Minimal web interface
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        num_transactions = request.form.get('num_transactions', 10)
        num_products = request.form.get('num_products', 5)
        context = request.form.get('context', 'grocery store')

        prompt = f"""
        Generate {num_transactions} unique transactions for frequent pattern mining.
        Each transaction should include 1-{num_products} products.
        Context: {context}.
        Output in strict JSON format like:
        [
          {{"transaction_id": 1, "items": ["apple", "milk", "bread"]}},
          {{"transaction_id": 2, "items": ["rice", "oil"]}}
        ]
        Ensure each run produces unique sets.
        """

        gemini_result = generate_with_gemini(prompt)
        import json
        try:
            # If Gemini response is a dict with 'candidates', extract the generated text
            if isinstance(gemini_result, dict) and 'candidates' in gemini_result:
                content = gemini_result['candidates'][0]['content']['parts'][0]['text']
                # Remove markdown code block if present
                if content.strip().startswith('```'):
                    content = content.strip()
                    # Remove the first line (```json or ```)
                    lines = content.splitlines()
                    if lines[0].startswith('```'):
                        lines = lines[1:]
                    # Remove the last line if it is ```
                    if lines and lines[-1].startswith('```'):
                        lines = lines[:-1]
                    content = '\n'.join(lines)
                transactions = json.loads(content)
            elif isinstance(gemini_result, str):
                transactions = json.loads(gemini_result)
            else:
                transactions = gemini_result
            result = transactions
        except Exception as e:
            error = f"Invalid JSON from Gemini: {gemini_result}"

    import json
    # Filter result to only include transaction_id and items
    filtered_result = None
    if result:
        try:
            # If result is a dict with 'candidates', extract the actual transactions list
            if isinstance(result, dict) and 'candidates' in result:
                # Try to extract the transactions from Gemini response
                content = result['candidates'][0]['content']['parts'][0]['text']
                transactions = eval(content)
            else:
                transactions = result
            filtered_result = [
                {"transaction_id": t["transaction_id"], "items": t["items"]}
                for t in transactions if "transaction_id" in t and "items" in t
            ]
        except Exception as e:
            filtered_result = None

    html = '''
    <html>
    <head><title>Transaction Generator</title></head>
    <body>
        <h2>Generate Transactions</h2>
        <form method="post">
            Number of Transactions: <input type="number" name="num_transactions" value="10"><br>
            Number of Products: <input type="number" name="num_products" value="5"><br>
            Context: <input type="text" name="context" value="grocery store"><br>
            <input type="submit" value="Generate">
        </form>
        {% if filtered_result %}
            <h3>Result:</h3>
            <pre id="json-output">{{ filtered_result | tojson(indent=2) }}</pre>
            <button onclick="downloadJSON()">Download JSON</button>
            <script>
            function downloadJSON() {
                var data = document.getElementById('json-output').textContent;
                var blob = new Blob([data], {type: 'application/json'});
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = 'output.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
            </script>
        {% endif %}
        {% if error %}
            <h3 style="color:red;">Error:</h3>
            <pre>{{ error }}</pre>
        {% endif %}
    </body>
    </html>
    '''
    return render_template_string(html, filtered_result=filtered_result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
