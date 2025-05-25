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

# Cấu hình VNES AI API (sử dụng Gemini backend)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Cấu hình DeepSeek API
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Cấu hình OpenRouter API
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY không được thiết lập trong .env. Vui lòng thêm key từ https://openrouter.ai/")
print("🔑 OpenRouter API Key detected and loaded successfully.")
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
def ask_vnes_ai(question, image):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        if image:
            image = Image.open(image).convert("RGB")
            response = model.generate_content([question, image])
        else:
            response = model.generate_content(question)

        # Xử lý phản hồi từ VNES AI
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

        if is_math_question(question):
            return {"image": generate_math_image(answer), "ai": "vnes_ai"}
        return {"text": answer, "ai": "vnes_ai"}
    except Exception as e:
        print("❌ VNES AI error:", e)
        return {"error": str(e), "ai": "vnes_ai"}

# Gửi tới DeepSeek API
def ask_deepseek(question, image):
    try:
        if image:
            return {"error": "DeepSeek API hiện không hỗ trợ hình ảnh.", "ai": "deepseek"}

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": question}],
            max_tokens=1024
        )
        answer = response.choices[0].message.content

        if is_math_question(question):
            return {"image": generate_math_image(answer), "ai": "deepseek"}
        return {"text": answer, "ai": "deepseek"}
    except Exception as e:
        print("❌ DeepSeek error:", e)
        return {"error": str(e), "ai": "deepseek"}

# Gửi tới OpenRouter API (với mô hình trả phí ChatGPT, hỗ trợ hình ảnh)
def ask_openrouter(question, image):
    try:
        # Sử dụng mô hình trả phí (ChatGPT, hỗ trợ hình ảnh)
        model = "openai/gpt-4o"  # Mô hình trả phí, hỗ trợ hình ảnh, yêu cầu tài khoản OpenRouter có số dư
        # Nếu không muốn hỗ trợ hình ảnh, có thể dùng "openai/gpt-3.5-turbo"
        # Nếu muốn thử mô hình miễn phí, thay bằng: "meta-llama/llama-3.1-8b-instruct:free"

        messages = [{"role": "user", "content": question}]

        if image:
            # Mã hóa hình ảnh thành base64
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

        response = openrouter_client.chat.completions.create(
            model=model,
            messages=messages if not image else messages[1:],  # Nếu có hình ảnh, chỉ gửi tin nhắn có hình
            max_tokens=1024
        )
        answer = response.choices[0].message.content

        if is_math_question(question):
            return {"image": generate_math_image(answer), "ai": "openrouter"}
        return {"text": answer, "ai": "openrouter"}
    except Exception as e:
        error_message = str(e)
        if "402" in error_message:
            error_message = "Tài khoản OpenRouter không đủ số dư để sử dụng mô hình trả phí. Vui lòng nạp thêm tiền tại https://openrouter.ai/credits."
        elif "401" in error_message:
            error_message = "Lỗi xác thực OpenRouter: Key không hợp lệ hoặc không được thiết lập. Vui lòng kiểm tra OPENROUTER_API_KEY trong .env."
        print("❌ OpenRouter error:", error_message)
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
    gemini_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    print("🔧 GEMINI_API_KEY (dùng cho VNES AI):", (gemini_key[:10] + "...") if gemini_key else "Chưa đặt")
    print("🔧 DEEPSEEK_API_KEY:", (deepseek_key[:10] + "...") if deepseek_key else "Chưa đặt")
    print("🔧 OPENROUTER_API_KEY:", (openrouter_key[:10] + "...") if openrouter_key else "Chưa đặt")
    app.run(host='0.0.0.0', port=5000, debug=True)
