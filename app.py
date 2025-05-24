import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import io

# Load biến môi trường
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Cấu hình Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
CORS(app)

# 👉 Hàm gửi ảnh và câu hỏi tới Gemini
def ask_gemini_with_image(question, image):
    try:
        # Mở hình ảnh từ file-like object
        img = Image.open(image)

        print("📤 Gửi hình ảnh + câu hỏi tới VNES:", question)
        response = model.generate_content(
            [question, img],  # Danh sách nội dung gồm text và hình ảnh
            stream=False,
        )
        print("📥 Phản hồi từ VNES:", response.text)
        return response.text
    except Exception as e:
        print("❌ Lỗi khi xử lý ảnh:", e)
        return f"Lỗi khi xử lý ảnh: {str(e)}"

# Trang chính
@app.route('/')
def home():
    return render_template('index.html')

# API upload hình ảnh và câu hỏi
@app.route('/ask-image', methods=['POST'])
def ask_with_image():
    try:
        question = request.form.get('question')
        image = request.files.get('image')

        if not question or not image:
            return jsonify({"error": "Both question and image are required"}), 400

        answer = ask_gemini_with_image(question, image)
        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ Lỗi:", e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    print("🔧 GEMINI_API_KEY:", GEMINI_API_KEY[:8] + "***")
    app.run(host='0.0.0.0', port=5000, debug=True)
