from flask import Flask, render_template, request, jsonify
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import oracledb as cx_Oracle
import os

app = Flask(__name__)

API_KEY = "F251024945"
DB_USER = "oil"
DB_PASSWORD = "oil"
DB_DSN = "localhost:1521/XE"

DATA_PATH = "merged_fuel_data.csv"
PRED_CSV = "prediction_3days_fixed.csv"

# ---------------- 데이터 로드 ----------------
if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
    data['DATE'] = pd.to_datetime(data['DATE'])
    available_ids = data['UNI_ID'].unique().tolist()
else:
    data = None
    available_ids = []

if os.path.exists(PRED_CSV):
    pred_df = pd.read_csv(PRED_CSV)
else:
    pred_df = pd.DataFrame(columns=["UNI_ID", "pred_day1", "pred_day2", "pred_day3", "avg_predicted_price_3days"])

# ---------------- DB 연결 ----------------
def get_connection():
    return cx_Oracle.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)

def get_sido_list():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT SIDOCD, SIDONM FROM OPINET_AREA ORDER BY SIDONM")
    sido_list = [{"code": r[0], "name": r[1]} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return sido_list

# ---------------- 시군구 조회 ----------------
@app.route("/get_sigungu")
def get_sigungu():
    sido_code = request.args.get("sido")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT SIGUNCD, SIGUNNM
        FROM OIL_PRICE
        WHERE SUBSTR(SIGUNCD,1,2) = :sido
        ORDER BY SIGUNNM
    """, [sido_code])
    sigungu_list = [{"code": r[0], "name": r[1]} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify(sigungu_list)

# ---------------- 주유소 검색 ----------------
@app.route("/search", methods=["POST"])
def search():
    sigungu = request.form.get("sigungu")
    brands = request.form.getlist("brand")
    results = []

    url = "http://www.opinet.co.kr/api/lowTop10.do"
    prodcd_list = ["B027", "D047"]

    for pc in prodcd_list:
        params = {"code": API_KEY, "out": "xml", "prodcd": pc, "area": sigungu, "cnt": 50}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            for oil in root.findall(".//OIL"):
                poll_div = oil.findtext("POLL_DIV_CO")
                if brands and poll_div not in brands:
                    continue
                uni_id = oil.findtext("UNI_ID")
                os_nm = oil.findtext("OS_NM")
                addr = oil.findtext("NEW_ADR") or oil.findtext("VAN_ADR")
                price = oil.findtext("PRICE")
                existing = next((r for r in results if r["UNI_ID"] == uni_id), None)
                if existing:
                    existing[pc] = price
                else:
                    results.append({
                        "UNI_ID": uni_id,
                        "OS_NM": os_nm,
                        "ADDR": addr,
                        "POLL_DIV_CD": poll_div,
                        "B027": price if pc == "B027" else None,
                        "D047": price if pc == "D047" else None
                    })
    return jsonify({"results": results})

@app.route("/get_price_and_pred")
def get_price_and_pred():
    uni_id = request.args.get("uni_id")

    # ---------------- 과거 데이터 ----------------
    actual_dates = []
    actual_prices = []
    monthly_avg = []

    if data is not None and uni_id in available_ids:
        df_uni = data[data['UNI_ID'] == uni_id].sort_values('DATE')
        this_year = pd.Timestamp.now().year
        df_year = df_uni[df_uni['DATE'].dt.year == this_year]
        df_1day = df_year[df_year['DATE'].dt.day == 1]

        if not df_1day.empty:
            # 과거 월 1일 가격
            df_avg = df_1day.groupby('DATE')['B027'].mean().reset_index()
            actual_dates = df_avg['DATE'].dt.strftime("%Y %m월").tolist()
            actual_prices = df_avg['B027'].astype(float).round(0).tolist()

            # 월별 평균
            monthly_avg = df_1day.groupby(df_1day['DATE'].dt.month)['B027'] \
                                 .mean() \
                                 .reindex(range(1, 13), fill_value=0) \
                                 .round(2) \
                                 .tolist()

    # ---------------- 오늘 가격 ----------------
    today_price = None
    url = "https://www.opinet.co.kr/api/detailById.do"
    params = {"code": API_KEY, "id": uni_id, "out": "xml"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        oil_elem = root.find(".//OIL")
        if oil_elem is not None:
            price_b027 = None
            for p in oil_elem.findall("OIL_PRICE"):
                if p.findtext("PRODCD") == "B027":
                    price_b027 = p.findtext("PRICE")
                    break
            if price_b027:
                today_price = float(price_b027)

    if today_price is not None:
        actual_dates.append("오늘")
        actual_prices.append(today_price)

    # ---------------- 3일 예측 ----------------
    predicted_prices = []
    pred_dates = []
    row = pred_df[pred_df["UNI_ID"] == uni_id]
    if not row.empty:
        today = pd.Timestamp.now().normalize()
        pred_dates = pd.date_range(today + pd.Timedelta(days=1), periods=3).strftime("%Y-%m-%d").tolist()
        for col in ["pred_day1", "pred_day2", "pred_day3"]:
            try:
                val = float(row.iloc[0][col])
                predicted_prices.append(val)
            except:
                predicted_prices.append(None)  # 값 없으면 None

    # ---------------- 전체 날짜 ----------------
    all_dates = actual_dates + pred_dates

    return jsonify({
        "dates": all_dates,
        "actual_prices": actual_prices,
        "predicted_prices": predicted_prices,
        "monthly_avg": monthly_avg
    })




# ---------------- 주유소 상세 ----------------
@app.route("/get_detail")
def get_detail():
    uni_id = request.args.get("uni_id")
    url = "https://www.opinet.co.kr/api/detailById.do"
    params = {"code": API_KEY, "id": uni_id, "out": "xml"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return jsonify({"error": "API 호출 실패"}), 500
    root = ET.fromstring(response.text)
    oil_elem = root.find(".//OIL")
    if oil_elem is None:
        return jsonify({"error": "데이터 없음"}), 404

    data_dict = {
        "UNI_ID": oil_elem.findtext("UNI_ID"),
        "OS_NM": oil_elem.findtext("OS_NM"),
        "VAN_ADR": oil_elem.findtext("VAN_ADR"),
        "NEW_ADR": oil_elem.findtext("NEW_ADR"),
        "TEL": oil_elem.findtext("TEL"),
        "POLL_DIV_CD": oil_elem.findtext("POLL_DIV_CO"),
        "LPG_YN": oil_elem.findtext("LPG_YN"),
        "MAINT_YN": oil_elem.findtext("MAINT_YN"),
        "CAR_WASH_YN": oil_elem.findtext("CAR_WASH_YN"),
        "CVS_YN": oil_elem.findtext("CVS_YN"),
        "KPETRO_YN": oil_elem.findtext("KPETRO_YN"),
        "OIL_PRICE": []
    }
    for price_elem in oil_elem.findall("OIL_PRICE"):
        data_dict["OIL_PRICE"].append({
            "PRODCD": price_elem.findtext("PRODCD"),
            "PRICE": price_elem.findtext("PRICE"),
            "TRADE_DT": price_elem.findtext("TRADE_DT"),
            "TRADE_TM": price_elem.findtext("TRADE_TM")
        })

    return jsonify(data_dict)

# ---------------- 메인 페이지 ----------------
@app.route("/", methods=["GET"])
def index():
    sido_list = get_sido_list()
    return render_template("index.html",
                           sido_list=sido_list,
                           center_lat=37.5665,
                           center_lon=126.9780)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
