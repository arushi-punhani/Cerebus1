from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
import input_guardrail
from hashlib import sha256

load_dotenv()   
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

app = Flask(__name__)

@app.route("/")
def home():
    return "Cerebus is RUNNING"

def _make_user_id():
    addr = request.remote_addr or "unknown"
    # if you want a little more entropy you can include other headers
    return sha256(addr.encode()).hexdigest()

chat_histories = {}

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("text", "")

        user_id = _make_user_id()

        if not prompt:
            return jsonify({"error": "No text provided"}), 400
        
        # conversation_history = data.get("conversation_history", None)
        history = chat_histories.setdefault(user_id, [])
        history.append({"role": "user", "content": prompt})
        print(history)

        classification = input_guardrail.classify_prompt(prompt, history)

        is_blocked = (
            classification.get("skeleton_key") == "yes"
            or classification.get("deceptive_delight") == "yes"
            or classification.get("many_shot") == "yes"
            or classification.get("crescendo") == "yes"
            or classification.get("safety") == "unsafe"
        )

    
        if is_blocked:
            history.append({"role": "assistant", "content": "Prompt has been blocked for the following reasons: "+';'.join(classification['reasons'])})
            return jsonify({
                "error": "Prompt blocked by input guardrail",
                "classification": classification
            }), 403
        
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)

        
        history.append({"role": "assistant", "content": response.text})

        return jsonify({
            "response": response.text,
            "guardrail": classification
        })

    except Exception as e:
        print(" Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)