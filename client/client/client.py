# client.py
import requests
import datetime
import sys

# ❗️ สำคัญ: แก้ไข IP Address นี้ให้ตรงกับ IP ของเครื่องครู (Server)
# (ถ้าทดสอบเครื่องเดียว ใช้ "http://127.0.0.1:5000")
# (ถ้าใช้ในห้องคอม ใช้ "http://192.168.1.XX:5000" ตาม IP เครื่องครู)
SERVER_URL = "http://10.3.10.32:5000" 

def login():
    """รับข้อมูลนักเรียน"""
    print("=== ระบบสอบวิชา Natural Language Processing ===")
    print("กรุณากรอกข้อมูลเพื่อเข้าสู่ระบบ (ทำได้ครั้งเดียว)")
    
    first_name = input("ชื่อจริง: ")
    last_name = input("นามสกุล: ")
    student_id = input("รหัสนักศึกษา: ")
    access_code = input("รหัสเข้าสอบ (Exam Code): ")
    
    if not all([first_name, last_name, student_id, access_code]):
        print("!!! กรุณากรอกข้อมูลให้ครบ")
        sys.exit(1)
        
    return {
        "first_name": first_name,
        "last_name": last_name,
        "student_id": student_id,
        "access_code": access_code
    }

def start_exam_session(user_info):
    """ติดต่อเซิร์ฟเวอร์เพื่อเริ่มสอบ"""
    try:
        response = requests.post(f"{SERVER_URL}/start_exam", json=user_info)
        response.raise_for_status() # เช็คว่ามี error ไหม (เช่น 4xx, 5xx)
        
        data = response.json()
        print(f"\nServer: {data.get('message')}")
        
        duration = data.get('duration_minutes')
        deadline_str = data.get('deadline')
        
        if duration and deadline_str:
            deadline_time = datetime.datetime.fromisoformat(deadline_str)
            print(f"--- คุณมีเวลา {duration} นาที ---")
            print(f"--- ต้องส่งคำตอบก่อนเวลา: {deadline_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            
        return True
        
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            try:
                error_msg = e.response.json().get('error', 'ข้อผิดพลาดไม่ทราบสาเหตุ')
                print(f"!!! [Error {e.response.status_code}] {error_msg}")
            except requests.exceptions.JSONDecodeError:
                print(f"เกิดข้อผิดพลาดร้ายแรงจากเซิร์ฟเวอร์: {e.response.text}")
        else:
            print(f"!!! [Connection Error] ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้: {e}")
            print(f"    กรุณาตรวจสอบว่า: ")
            print(f"    1. Server เปิดอยู่หรือไม่")
            print(f"    2. ป้อน SERVER_URL ถูกต้องหรือไม่ (ปัจจุบันคือ {SERVER_URL})")
        return False

def get_exam_questions():
    """ดึงข้อสอบจากเซิร์ฟเวอร์"""
    try:
        print("\nกำลังดึงข้อสอบ...")
        response = requests.get(f"{SERVER_URL}/get_questions")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"!!! [Error] ไม่สามารถดึงข้อสอบได้: {e}")
        return None

def run_exam_loop(questions):
    """วงลูปการทำข้อสอบ"""
    answers = {} # ที่เก็บคำตอบ { "q_id": "answer" }
    
    print("\n--- เริ่มทำข้อสอบ ---")
    print("พิมพ์ A, B, C, D เพื่อตอบ หรือ 'skip' เพื่อข้าม")
    print(f"มีทั้งหมด {len(questions)} ข้อ\n")

    for i, q in enumerate(questions):
        print(f"ข้อที่ {i + 1} / {len(questions)}")
        print(f"(หัวข้อ: {q['topic']})")
        print(f"Q: {q['question']}")
        
        for key, value in q['options'].items():
            print(f"  {key}: {value}")
            
        while True:
            answer = input("คำตอบของคุณ: ").strip().upper()
            
            if answer in q['options'].keys():
                answers[q['id']] = answer
                break
            elif answer == 'SKIP':
                answers[q['id']] = "skipped"
                print("--- ข้ามข้อนี้ ---")
                break
            else:
                print("กรุณาเลือก A, B, C, D หรือ 'skip'")
        print("-" * 20)
        
    return answers

def submit_results(user_info, answers):
    """ส่งคำตอบไปยังเซิร์ฟเวอร์"""
    payload = {
        "student_id": user_info['student_id'],
        "answers": answers
    }
    
    print("\nกำลังส่งคำตอบ...")
    try:
        response = requests.post(f"{SERVER_URL}/submit_exam", json=payload)
        response.raise_for_status() 
        
        result = response.json()
        print(f"✅ ส่งข้อสอบสำเร็จ! {result.get('message')}")
        print(f"คุณได้คะแนน: {result.get('score')} / {result.get('total')}")
        
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            try:
                error_msg = e.response.json().get('error', 'ข้อผิดพลาดไม่ทราบสาเหตุ')
                print(f"!!! [Error {e.response.status_code}] {error_msg}")
            except requests.exceptions.JSONDecodeError:
                 print(f"เกิดข้อผิดพลาดร้ายแรงจากเซิร์ฟเวอร์: {e.response.text}")
        else:
            print(f"!!! [Connection Error] เกิดข้อผิดพลาดในการส่งคำตอบ: {e}")

def main():
    # 1. Login
    user_info = login()
    
    # 2. แจ้งเซิร์ฟเวอร์ว่าเริ่มสอบ
    if not start_exam_session(user_info):
        print("\nไม่สามารถเริ่มการสอบได้ กรุณาติดต่อผู้คุมสอบ")
        sys.exit(1) 

    # 3. ดึงข้อสอบ
    questions = get_exam_questions()
    if not questions or len(questions) == 0:
        print("!!! ไม่ได้รับข้อสอบจากเซิร์ฟเวอร์")
        sys.exit(1)

    # 4. เริ่มทำข้อสอบ
    start_time = datetime.datetime.now()
    print(f"เวลาเริ่มทำ (บนเครื่องของคุณ): {start_time.strftime('%H:%M:%S')}")
    
    user_answers = run_exam_loop(questions)
    
    end_time = datetime.datetime.now()
    print(f"ทำข้อสอบเสร็จสิ้น เวลา: {end_time.strftime('%H:%M:%S')}")
    print(f"ใช้เวลาไป: {end_time - start_time}")

    # 5. ยืนยันการส่ง
    while True:
        confirm = input("\nคุณต้องการส่งคำตอบหรือไม่? (y/n): ").strip().lower()
        if confirm == 'y':
            submit_results(user_info, user_answers)
            break
        elif confirm == 'n':
            print("ยกเลิกการส่งข้อสอบ (โปรดทราบว่าเวลายังคงเดินอยู่ และหากหมดเวลาจะไม่สามารถส่งได้)")
            # วนลูปกลับไปถามใหม่
        else:
            print("กรุณาพิมพ์ 'y' (ใช่) หรือ 'n' (ไม่ใช่)")

if __name__ == "__main__":
    main()