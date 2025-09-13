import os
import requests

# Lấy token và chat_id từ Render (sau này sẽ config trong Environment Variables)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    requests.post(url, data=data)

if __name__ == "__main__":
    send_message("✅ Bot đã chạy thành công trên Render!")
