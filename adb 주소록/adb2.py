import subprocess

def get_contacts():
    # ADB 명령어를 통해 주소록 가져오기
    command = "adb shell content query --uri content://contacts/phones"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # 결과 파싱
    contacts = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            # 각 줄에서 이름과 전화번호 추출
            contact_info = {}
            for item in line.split(","):
                key_value = item.split("=")
                if len(key_value) == 2:
                    contact_info[key_value[0].strip()] = key_value[1].strip()
            contacts.append(contact_info)
    else:
        print("Error retrieving contacts:", result.stderr)

    return contacts

def save_contacts_to_file(contacts, filename='contacts.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for contact in contacts:
            name = contact.get('display_name', '이름 없음')
            phone = contact.get('number', '번호 없음')
            f.write(f"{name}: {phone}\n")

if __name__ == "__main__":
    contacts = get_contacts()
    save_contacts_to_file(contacts)
    print("연락처가 contacts.txt 파일에 저장되었습니다.")
