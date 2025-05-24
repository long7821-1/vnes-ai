import google.generativeai as genai

# Cấu hình API Key
genai.configure(api_key="AIzaSyB1VvayOvIgqWUCAwgwXLJy7Wx34vJileU")

# Sử dụng model có thật
model = genai.GenerativeModel("gemini-1.5-flash")

# Gửi câu hỏi
response = model.generate_content("Thủ đô Việt Nam là gì?")

# In kết quả
print(response.text)
