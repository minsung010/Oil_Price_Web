import requests
import xml.etree.ElementTree as ET
import cx_Oracle

# ------------------------------
# Oracle DB 접속 정보
# ------------------------------
username = "oil"
password = "oil"
host = "localhost"  # 실제 Oracle 호스트
port = 1521  # Oracle 포트
service_name = "xe"  # 서비스 이름 (XE, ORCL 등)

dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
conn = cx_Oracle.connect(username, password, dsn)
cur = conn.cursor()

# ------------------------------
# 오피넷 API 정보
# ------------------------------
api_key = "F251024945"  # 인증키
sido_list = [f"{i:02}" for i in range(1, 20)]  # 01 ~ 19 지역 코드

# ------------------------------
# 테이블 생성 (없으면)
# ------------------------------
cur.execute("""
CREATE TABLE OIL_PRICE (
    SIGUNCD VARCHAR2(10),
    SIGUNNM VARCHAR2(50),
    PRODCD VARCHAR2(10),
    PRICE NUMBER(10,2),
    DIFF NUMBER(10,2),
    UPDATE_DATE DATE DEFAULT SYSDATE
)
""")
conn.commit()

# ------------------------------
# 데이터 가져오기 & DB 저장
# ------------------------------
for sido in sido_list:
    url = f"https://www.opinet.co.kr/api/avgSigunPrice.do?out=xml&sido={sido}&code={api_key}"
    resp = requests.get(url)

    if resp.status_code != 200:
        print(f"{sido} 지역 데이터 요청 실패!")
        continue

    root = ET.fromstring(resp.text)

    for oil in root.findall('OIL'):
        siguncd = oil.find('SIGUNCD').text
        sigunnm = oil.find('SIGUNNM').text
        prodcd = oil.find('PRODCD').text
        price = float(oil.find('PRICE').text)
        diff = float(oil.find('DIFF').text)

        # MERGE로 삽입/업데이트
        cur.execute("""
            MERGE INTO OIL_PRICE t
            USING DUAL
            ON (t.SIGUNCD = :siguncd AND t.PRODCD = :prodcd)
            WHEN MATCHED THEN
                UPDATE SET PRICE = :price, DIFF = :diff, UPDATE_DATE = SYSDATE
            WHEN NOT MATCHED THEN
                INSERT (SIGUNCD, SIGUNNM, PRODCD, PRICE, DIFF)
                VALUES (:siguncd, :sigunnm, :prodcd, :price, :diff)
        """, siguncd=siguncd, sigunnm=sigunnm, prodcd=prodcd, price=price, diff=diff)

    conn.commit()
    print(f"{sido} 지역 데이터 저장 완료!")

# ------------------------------
# 종료
# ------------------------------
cur.close()
conn.close()
print("모든 데이터 저장 완료!")
