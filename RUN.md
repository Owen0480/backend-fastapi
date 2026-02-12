# 실행 방법 (변경사항 확인용)

## 1. 의존성 설치

프로젝트 루트에서:

```bash
pip install -r requirements.txt
```

(Gemini LLM 사용 시 추가)

```bash
pip install langchain-google-genai
```

---

## 2. 서버 실행

```bash
uvicorn main:app --reload
```

- 기본 주소: **http://127.0.0.1:8000**
- `--reload`: 코드 수정 시 자동 재시작

---

## 3. 변경사항 확인 경로

### (1) 데모 페이지 (브라우저)

다음 주소 **둘 다** 데모 페이지로 연결됩니다.

| 주소 | 설명 |
|------|------|
| **http://127.0.0.1:8000/demo** | 짧은 주소, 자동 리다이렉트 |
| **http://127.0.0.1:8000/api/v1/demo/travel-demo** | 전체 경로 |

첫 접속 시 **http://127.0.0.1:8000/** 에서 「여행 데모 페이지 열기」 링크로 이동해도 됩니다.

- 프리셋 버튼으로 예시 문장 선택 후 **「그래프 실행」** 클릭
- 터미널에 `[graph] 노드 실행: ...`, `[Travel Graph] 실행 시작/완료`, `LLM 사용 여부` 등 로그 출력

### (2) API로 직접 호출

**POST** `http://127.0.0.1:8000/api/v1/travel/travel`

Body (JSON):

```json
{
  "message": "부산 바다 보이는 감성 숙소 추천해줘",
  "user_id": 123
}
```

- PowerShell 예시:
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/travel/travel" -Method POST -ContentType "application/json" -Body '{"message":"부산 바다 보이는 감성 숙소 추천해줘","user_id":123}'
  ```
- 또는 curl:
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/v1/travel/travel" -H "Content-Type: application/json" -d "{\"message\":\"부산 바다 보이는 감성 숙소 추천해줘\",\"user_id\":123}"
  ```

### (3) API 문서에서 테스트

**http://127.0.0.1:8000/docs**

- `POST /api/v1/travel/travel` 선택 → "Try it out" → Body 입력 후 Execute

---

## 4. 터미널에서 확인되는 로그

실행 시 터미널에는 예를 들어 다음과 같이 출력됩니다.

```
==================================================
[Travel Graph] 실행 시작
[Travel Graph] 입력 메시지: 부산 바다 보이는 감성 숙소 추천해줘
[Travel Graph] LLM(Gemini) 사용 가능 여부: False
==================================================
[graph] 노드 실행: user_input_node
[graph] 노드 실행: intent_classifier_node
[graph] 노드 실행: travel_classifier_llm_node
[graph] LLM 사용 여부 (분류): False
[graph] fallback 분류 결과: who=unknown, why=unknown, stage=refinement
...
[graph] 노드 실행: clarifying_question_node
[graph] LLM 사용 여부 (추가질문): False
[graph] 추가질문 fallback 사용 (question_map)
==================================================
[Travel Graph] 실행 완료
[Travel Graph] intent: recommend_accommodation, who: unknown, why: unknown
[Travel Graph] needs_clarification: True, recommendations: 0건
==================================================
```

- **LLM 사용 가능 여부**: `.env`에 `GOOGLE_API_KEY`(또는 `GEMINI_API_KEY`)가 있으면 `True`
- **LLM 사용 여부 (분류/추가질문)**: 해당 단계에서 실제로 Gemini를 호출했는지

---

## 5. 타임아웃·에러 표출

- **LLM 호출**: `LLM_TIMEOUT_SEC`(기본 25초) 초과 시 fallback 적용, 터미널에 `[graph] LLM 분류 타임아웃 (...) → fallback 적용` 로그.
- **그래프 전체**: `GRAPH_TIMEOUT_SEC`(기본 60초) 초과 시 요청 중단, 응답에 `error: "그래프 실행 타임아웃 (60초 초과)"` 포함.
- **데모 페이지**: fetch 타임아웃 70초. 타임아웃·서버 에러 시 결과 영역을 붉은 테두리(`pre.error`)로 표시하고 메시지 출력.
- **환경 변수**: `.env` 에 `LLM_TIMEOUT_SEC=25`, `GRAPH_TIMEOUT_SEC=60` 으로 변경 가능.

---

## 6. LLM 사용해서 테스트하려면

프로젝트 루트에 `.env` 파일을 만들고:

```env
GOOGLE_API_KEY=여기에_Gemini_API_키_입력
```

저장 후 서버를 다시 띄우면, 분류·추가질문 모두 LLM이 사용됩니다.

---

## 7. reload 시 "Exception ignored in: BaseEventLoop.__del__" / CancelledError

`uvicorn main:app --reload` 로 실행 중 **파일을 저장하면** WatchFiles가 변경을 감지하고 이전 프로세스를 종료합니다. 이때 **이전 프로세스**의 asyncio 이벤트 루프를 정리하는 과정에서 다음 로그가 나올 수 있습니다.

- `Exception ignored in: <function BaseEventLoop.__del__ at ...>`
- `asyncio.exceptions.CancelledError`

**의미**: 코드가 멈춘 것이 아니라, **재시작(teardown)** 과정에서 나오는 로그입니다. 새 프로세스는 그대로 뜨고, 이후 요청은 정상 처리됩니다.

**대응**:
- 새로 고침 후 `INFO: Application startup complete.` 가 다시 찍히면 그대로 사용해도 됩니다.
- 이 로그가 거슬리면 **reload 없이** 실행해 보세요:  
  `uvicorn main:app --host 0.0.0.0 --port 8000`  
  코드 수정 후에는 서버를 수동으로 한 번씩 끄고 다시 실행하면 됩니다.
- 그래프는 **첫 요청 시에만** 생성되도록 바꿔 두어서, reload 시 teardown 시점에 연결된 객체가 줄어들었습니다.

---

## 8. Ctrl+C로 서버를 끌 때 "ERROR: Traceback ... KeyboardInterrupt"

서버를 **Ctrl+C**로 중지하면 uvicorn이 `KeyboardInterrupt` 를 받으면서 위와 비슷한 Traceback을 출력할 수 있습니다.

- **의미**: 서버를 사용자가 중지했다는 신호(KeyboardInterrupt)를 uvicorn이 받고, 그 과정에서 나오는 로그입니다.
- **대응**: 무시해도 됩니다. 서버가 종료된 것이 정상입니다. 다시 켜려면 `uvicorn main:app --reload` 를 다시 실행하면 됩니다.

---

## 9. 국내 여행로그 4개 폴더 → DB에 넣고 이미지 추천 쓰기

이미지 추천을 **backend-fastapi/images** 대신 **국내 여행로그 데이터(수도권/동부권/서부권/제주도 및 도서지역)** 4개 폴더 기준으로 쓰려면, 아래 순서대로 하면 됩니다.

### 폴더 구조 (travel_pj 기준)

```
D:\travel_pj\
  국내 여행로그 데이터(수도권)\Sample\01.원천데이터\photo\  ...  02.라벨링데이터\csv\*.csv
  국내 여행로그 데이터(동부권)\...
  국내 여행로그 데이터(서부권)\...
  국내 여행로그 데이터(제주도 및 도서지역)\...
  backend-fastapi\
```

### 1) MariaDB 테이블 생성

- Docker MariaDB 3306에서 `travel` DB 사용 중이어야 함.
- HeidiSQL / DBeaver / MySQL Workbench 등에서 **travel** DB 선택 후 아래 SQL 실행:

```sql
-- docs/sql/create-place-tables.sql 내용 실행
```

또는 터미널에서:

```powershell
cd D:\travel_pj\backend-fastapi
Get-Content docs\sql\create-place-tables.sql | mysql -h 127.0.0.1 -P 3306 -u root -p1104 travel
```

### 2) .env 설정 (DB 비밀번호)

`backend-fastapi\.env` 에 다음 추가 (Docker MariaDB 비밀번호가 1104인 경우):

```env
MARIADB_HOST=127.0.0.1
MARIADB_PASSWORD=1104
```

(선택) 4개 폴더가 있는 루트가 `D:\travel_pj` 가 아니면:

```env
TRAVEL_DATA_ROOT=D:\경로\to\국내여행로그_상위폴더
```

### 3) 국내 여행로그 → place / place_photo INSERT

```powershell
cd D:\travel_pj\backend-fastapi
.\venv\Scripts\Activate.ps1
python scripts/insert_place_data.py
```

- **처음 한 번만** DB를 비우고 넣으려면:  
  `$env:CLEAR_BEFORE_INSERT="1"; python scripts/insert_place_data.py`
- **실제 이미지 파일이 있는 행만** 넣으려면:  
  `$env:ONLY_EXISTING_IMAGES="1"; python scripts/insert_place_data.py`
- 데이터가 너무 크면 샘플만 (예: 10%):  
  `$env:INSERT_SAMPLE_RATIO="0.1"; python scripts/insert_place_data.py`

### 4) 서버 재시작

```powershell
uvicorn main:app --reload
```

- 터미널에 **"이미지 추천 DB(3306) 로드: place_photo N건"** 이 나오면 DB 사용 중.
- **"DB 이미지 분석 중... (3306 place_photo 기준, 이미지 루트: ...)"** 후 4개 폴더 이미지로 임베딩 계산 (첫 실행 시 몇 분 걸릴 수 있음).
- 이미지 URL은 `/images/파일명` 으로 서빙되며, `main.py` 의 `_resolve_image_path` 가 **backend-fastapi/images** 없으면 **국내 여행로그 4개 폴더 photo** 에서 찾습니다.

---

## 10. 데모 페이지가 안 열릴 때

1. **주소 확인**  
   - `http://127.0.0.1:8000/demo` 또는 `http://127.0.0.1:8000/api/v1/demo/travel-demo`  
   - `http://localhost:8000/demo` 도 동일하게 사용 가능  

2. **서버 실행 여부**  
   - 터미널에서 `uvicorn main:app --reload` 가 실행 중인지 확인  
   - 브라우저에서 `http://127.0.0.1:8000/` 이 열리면 서버는 정상  

3. **패키지 오류로 서버가 안 뜨는 경우**  
   - `langchain-google-genai` 는 **선택** 사항 (없어도 데모는 키워드 fallback으로 동작)  
   - `ModuleNotFoundError: langchain_google_genai` 가 나오면:  
     - `pip install langchain-google-genai` 로 설치하거나  
     - `nodes.py` 에서 해당 패키지는 `_get_travel_llm()` 안에서만 import 되므로, 위 수정이 반영된 상태면 서버는 떠야 함  

4. **페이지는 열리는데 "그래프 실행"만 계속 실패**  
   - 개발자 도구(F12) → Network 탭에서 `/api/v1/travel/travel` 요청의 상태 코드와 응답 내용 확인  
   - 500 이면 서버 터미널 로그에 에러 메시지가 찍힘  
   - CORS/네트워크 문제일 수 있으니, 반드시 **같은 호스트**(예: `http://127.0.0.1:8000/...`)에서 데모 페이지를 열었는지 확인  
   - `file://` 로 HTML만 연 경우 API 호출이 실패할 수 있으므로, 항상 `http://127.0.0.1:8000/demo` 로 접속
