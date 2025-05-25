import os
import io
import base64
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai

# Load biến môi trường
load_dotenv()

# Cấu hình OpenAI client với OpenRouter
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Cấu hình Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Kiểm tra câu hỏi Toán học
def is_math_question(question):
    keywords = ["giải", "phương trình", "tính", "đạo hàm", "tích phân", "biểu đồ", "hình học", "vẽ"]
    return any(kw in question.lower() for kw in keywords)

# Tạo ảnh kết quả toán học
def generate_math_image(text):
    try:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, text, fontsize=14, ha='center', va='center', wrap=True)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print("❌ Lỗi tạo ảnh:", e)
        return None

# Gửi tới OpenAI GPT-4o qua OpenRouter
def ask_openai(question, image):
    try:
        messages = [{"role": "user", "content": question}]
        if image:
            image = Image.open(image).convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024
        )
        answer = response.choices[0].message.content
        if is_math_question(question):
            return {"image": generate_math_image(answer), "ai": "openai"}
        return {"text": answer, "ai": "openai"}
    except Exception as e:
        print("❌ OpenAI error:", e)
        return {"error": str(e), "ai": "openai"}

# Gửi tới Gemini 1.5 Flash
def ask_gemini(question, image):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        if image:
            image = Image.open(image).convert("RGB")
            response = model.generate_content([question, image])
        else:
            response = model.generate_content(question)

        if hasattr(response, 'text'):
            answer = response.text
        elif hasattr(response, 'parts') and response.parts:
            answer = response.parts[0].text if response.parts[0].text else str(response.parts)
        else:
            raise ValueError("Gemini API không trả về nội dung hợp lệ.")

        if is_math_question(question):
            return {"image": generate_math_image(answer), "ai": "gemini"}
        return {"text": answer, "ai": "gemini"}
    except Exception as e:
        print("❌ Gemini error:", e)
        return {"error": str(e), "ai": "gemini"}

# Giao diện
@app.route('/')
def home():
    return render_template('index.html')

# API chính
@app.route('/ask-image', methods=['POST'])
def ask_with_image():
    question = request.form.get('question')
    image = request.files.get('image')
    ai_model = request.form.get('ai', 'openai')

    if not question:
        return jsonify({"error": "Câu hỏi không được để trống.", "ai": ai_model}), 400

    if ai_model == "gemini":
        result = ask_gemini(question, image)
    else:
        result = ask_openai(question, image)

    if "image" in result:
        return send_file(result["image"], mimetype='image/png')
    elif "text" in result:
        return jsonify({"answer": result["text"], "ai": result["ai"]})
    else:
        return jsonify({"error": result.get("error", "Lỗi không xác định."), "ai": result["ai"]}), 500

if __name__ == '__main__':
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    print("🔧 OPENAI_API_KEY:", (openai_key[:10] + "...") if openai_key else "Chưa đặt")
    print("🔧 GEMINI_API_KEY:", (gemini_key[:10] + "...") if gemini_key else "Chưa đặt")
    app.run(host='0.0.0.0', port=5000, debug=True)
