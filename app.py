import os
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import io
import matplotlib.pyplot as plt

# Load biến môi trường
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Cấu hình Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
CORS(app)

# 🔍 Hàm kiểm tra xem câu hỏi có phải Toán học không
def is_math_question(question):
    keywords = ["giải", "phương trình", "tính", "đạo hàm", "tích phân", "biểu đồ", "hình học", "vẽ"]
    return any(kw in question.lower() for kw in keywords)

# 🖼️ Tạo ảnh từ lời giải văn bản
def generate_math_image(solution_text):
    try:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, solution_text, fontsize=14, ha='center', va='center', wrap=True)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf
    except Exception as e:
        print("❌ Lỗi khi tạo ảnh Toán:", e)
        return None

# 📤 Gửi ảnh + câu hỏi tới Gemini và xử lý kết quả
def ask_gemini_with_image(question, image):
    try:
        img = Image.open(image)
        print("📤 Gửi hình ảnh + câu hỏi tới Gemini:", question)

        response = model.generate_content(
            [question, img],
            stream=False,
        )

        answer = response.text
        print("📥 Phản hồi từ Gemini:", answer)

        if is_math_question(question):
            img_buf = generate_math_image(answer)
            return {"image": img_buf}

        return {"text": answer}

    except Exception as e:
        print("❌ Lỗi xử lý ảnh:", e)
        return {"error": str(e)}

# Trang chính
@app.route('/')
def home():
    return render_template('index.html')

# API: Gửi ảnh + câu hỏi
@app.route('/ask-image', methods=['POST'])
def ask_with_image():
    try:
        question = request.form.get('question')
        image = request.files.get('image')

        if not question or not image:
            return jsonify({"error": "Cần cung cấp cả câu hỏi và hình ảnh."}), 400

        result = ask_gemini_with_image(question, image)

        if "image" in result:
            return send_file(result["image"], mimetype='image/png')

        elif "text" in result:
            return jsonify({"answer": result["text"]})

        else:
            return jsonify({"error": result.get("error", "Lỗi không xác định.")}), 500

    except Exception as e:
        print("❌ Lỗi tổng quát:", e)
        return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

if __name__ == '__main__':
    print("🔧 GEMINI_API_KEY:", GEMINI_API_KEY[:8] + "***")
    app.run(host='0.0.0.0', port=5000, debug=True)
