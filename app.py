import os
import io
import base64
import time
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai

# Load biến môi trường
load_dotenv()

# Cấu hình VNES AI API (sử dụng Gemini backend)
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY không được thiết lập. Vui lòng thêm key trong Render Environment Variables.")
genai.configure(api_key=gemini_key)
print("🔧 GEMINI_API_KEY loaded:", gemini_key[:10] + "..." if gemini_key else "Chưa đặt")

# Cấu hình DeepSeek API
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_key:
    raise ValueError("DEEPSEEK_API_KEY không được thiết lập. Vui lòng thêm key trong Render Environment Variables.")
deepseek_client = OpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com"
)
print("🔧 DEEPSEEK_API_KEY loaded:", deepseek_key[:10] + "..." if deepseek_key else "Chưa đặt")

# Cấu hình OpenRouter API
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY không được thiết lập trong Render Environment Variables. Vui lòng thêm key từ https://openrouter.ai/")
print("🔑 OpenRouter API Key loaded:", openrouter_api_key[:10] + "...")
openrouter_client = OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

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

# Gửi tới VNES AI (sử dụng Gemini backend)
def ask_vnes_ai(question, image, retries=2, delay=60):
    for attempt in range(retries + 1):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            print(f"🔍 Sending request to VNES AI (Gemini), attempt {attempt + 1}/{retries + 1}...")
            if image:
                image = Image.open(image).convert("RGB")
                response = model.generate_content([question, image])
            else:
                response = model.generate_content(question)

            if hasattr(response, 'text'):
                answer = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    answer = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else str(candidate.content.parts[0])
                else:
                    raise ValueError("VNES AI không có nội dung hợp lệ trong candidates. Phản hồi: " + str(response))
            elif hasattr(response, 'parts') and response.parts:
                answer = response.parts[0].text if hasattr(response.parts[0], 'text') else str(response.parts[0])
            else:
                raise ValueError("VNES AI không trả về nội dung hợp lệ. Phản hồi: " + str(response))

            print(f"✅ VNES AI responded: {answer[:50]}...")
            if is_math_question(question):
                return {"image": generate_math_image(answer), "ai": "vnes_ai"}
            return {"text": answer, "ai": "vnes_ai"}
        except Exception as e:
            error_message = str(e)
            if "rate_limit_error" in error_message or "400" in error_message:
                if attempt < retries:
                    print(f"⚠️ VNES AI rate limit error. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                error_message = "Vượt quá giới hạn tốc độ VNES AI. Vui lòng thử lại sau vài phút."
            print("❌ VNES AI error:", error_message)
            return {"error": error_message, "ai": "vnes_ai"}

# Gửi tới DeepSeek API
def ask_deepseek(question, image, retries=2, delay=60):
    for attempt in range(retries + 1):
        try:
            if image:
                return {"error": "DeepSeek API hiện không hỗ trợ hình ảnh.", "ai": "deepseek"}

            print(f"🔍 Sending request to DeepSeek, attempt {attempt + 1}/{retries + 1}...")
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": question}],
                max_tokens=1024
            )
            answer = response.choices[0].message.content
            print(f"✅ DeepSeek responded: {answer[:50]}...")

            if is_math_question(question):
                return {"image": generate_math_image(answer), "ai": "deepseek"}
            return {"text": answer, "ai": "deepseek"}
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                if attempt < retries:
                    print(f"⚠️ DeepSeek rate limit error. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                error_message = "Vượt quá giới hạn tốc độ DeepSeek. Vui lòng thử lại sau vài phút."
            print("❌ DeepSeek error:", error_message)
            return {"error": error_message, "ai": "deepseek"}

# Gửi tới OpenRouter API
def ask_openrouter(question, image, retries=2, delay=60):
    for attempt in range(retries + 1):
        try:
            model = "openai/gpt-4o"
            messages = [{"role": "user", "content": question}]
            print(f"🔍 Sending request to OpenRouter with model: {model}, attempt {attempt + 1}/{retries + 1}...")

            if image:
                image = Image.open(image).convert("RGB")
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                })
                print("📷 Image included in request.")

            response = openrouter_client.chat.completions.create(
                model=model,
                messages=messages if not image else messages[1:],
                max_tokens=1024
            )
            answer = response.choices[0].message.content
            print(f"✅ OpenRouter responded: {answer[:50]}...")

            if is_math_question(question):
                return {"image": generate_math_image(answer), "ai": "openrouter"}
            return {"text": answer, "ai": "openrouter"}
        except Exception as e:
            error_message = str(e)
            print(f"❌ OpenRouter error on attempt {attempt + 1}: {error_message}")
            if "401" in error_message:
                error_message = "Lỗi xác thực OpenRouter: Key không hợp lệ. Vui lòng kiểm tra lại OPENROUTER_API_KEY trong Render Environment."
            elif "402" in error_message:
                error_message = "Tài khoản OpenRouter không đủ số dư. Vui lòng nạp thêm tiền tại https://openrouter.ai/credits."
            elif "429" in error_message:
                if attempt < retries:
                    print(f"⚠️ OpenRouter rate limit error. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                error_message = "Vượt quá giới hạn tốc độ OpenRouter. Vui lòng thử lại sau vài phút."
            return {"error": error_message, "ai": "openrouter"}

# Giao diện
@app.route('/')
def home():
    return render_template('index.html')

# API chính
@app.route('/ask-image', methods=['POST'])
def ask_with_image():
    question = request.form.get('question')
    image = request.files.get('image')
    ai_model = request.form.get('ai', 'vnes_ai')

    if not question:
        return jsonify({"error": "Câu hỏi không được để trống.", "ai": ai_model}), 400

    if ai_model == "deepseek":
        result = ask_deepseek(question, image)
    elif ai_model == "openrouter":
        result = ask_openrouter(question, image)
    else:
        result = ask_vnes_ai(question, image)

    if "image" in result:
        return send_file(result["image"], mimetype='image/png')
    elif "text" in result:
        return jsonify({"answer": result["text"], "ai": result["ai"]})
    else:
        return jsonify({"error": result.get("error", "Lỗi không xác định."), "ai": result["ai"]}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
