import os, time, requests

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

def send_message(text: str):
    if not TOKEN or not CHAT_ID:
        print("[ERR] Missing TELEGRAM_TOKEN or CHAT_ID")
        return
    try:
        r = requests.post(API_URL, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        print("Status:", r.status_code, r.text[:200])
    except Exception as e:
        print("[ERR] send_message:", e)

if __name__ == "__main__":
    send_message("🚀 Bot đã chạy thành công trên Render!")
    while True:
        time.sleep(30)   # Giữ bot sống
