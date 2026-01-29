"""
간단한 프론트 테스트 페이지 (Who/Why/Constraints/When mock_docs 테스트)
"""
import html as html_lib
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


# Who/Why/Constraints/When 조건별 테스트용 프리셋
DEMO_PRESETS = [
    {"label": "부산 바다·감성", "message": "부산 바다 보이는 감성 숙소 추천해줘"},
    {"label": "부산 가족 여행", "message": "부산에서 아이랑 가볼 만한 데 있어? 유모차 가능하고"},
    {"label": "제주 나홀 힐링", "message": "제주 나홀로 힐링 가려고요. 혼밥 가능한 바다쪽 카페 알려줘"},
    {"label": "강릉 커플", "message": "강릉 커플 여행으로 바다 뷰 카페랑 산책 코스 추천해"},
    {"label": "부산 효도 여행", "message": "부모님이랑 부산 가는데 걷기 편하고 한식 맛집이나 온천 추천해줘"},
    {"label": "제주 미식·문화", "message": "제주 동문시장 근처 로컬 맛집이나 문화 체험 있나?"},
]


@router.get("/travel-demo", response_class=HTMLResponse, tags=["demo"])
async def travel_demo_page():
    """
    여행 그래프 한 바퀴 테스트용 페이지. Who/Why/Constraints/When mock_docs 조건 테스트 가능.
    """
    def esc(s: str) -> str:
        return html_lib.escape(s, quote=True)

    presets_html = "".join(
        f'<button type="button" class="preset" data-msg="{esc(p["message"])}">{esc(p["label"])}</button>'
        for p in DEMO_PRESETS
    )
    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="UTF-8" />
      <title>Travel Graph Demo (Who/Why/Constraints/When)</title>
      <style>
        body {{ font-family: "Malgun Gothic", sans-serif; margin: 24px; max-width: 900px; }}
        textarea {{ width: 100%; height: 80px; box-sizing: border-box; }}
        .summary-box {{ background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 10px 14px; border-radius: 6px; font-size: 14px; margin: 12px 0; }}
        .thread {{ margin: 12px 0; }}
        .thread .bubble {{ max-width: 85%; padding: 10px 14px; border-radius: 12px; margin: 6px 0; white-space: pre-wrap; }}
        .thread .bubble.user {{ background: #e3f2fd; margin-left: 0; }}
        .thread .bubble.assistant {{ background: #f3e5f5; margin-left: auto; }}
        .answer-bubble {{ background: #e8f4fd; border-left: 4px solid #0d6efd; padding: 14px 18px; border-radius: 8px; font-size: 15px; line-height: 1.6; white-space: pre-wrap; }}
        .answer-bubble.error {{ background: #fff0f0; border-left-color: #c00; }}
        pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; white-space: pre-wrap; font-size: 12px; overflow-x: auto; }}
        pre.error, .detail-json {{ font-size: 12px; }}
        button {{ padding: 10px 16px; margin: 4px 4px 4px 0; cursor: pointer; }}
        .preset {{ background: #e8f4fd; border: 1px solid #0d6efd; color: #0d6efd; }}
        .preset:hover {{ background: #0d6efd; color: #fff; }}
        #run {{ background: #0d6efd; color: #fff; border: none; }}
        .meta {{ margin: 12px 0; padding: 10px; background: #f0f7ff; border-radius: 6px; font-size: 14px; }}
        .meta strong {{ margin-right: 6px; }}
      </style>
    </head>
    <body>
      <h2>Travel Graph Demo</h2>
      <p>Who/Why/Constraints/When 조건에 맞는 mock_docs로 추천이 필터링됩니다. 아래 프리셋이나 자유 입력으로 테스트하세요.</p>
      <div class="presets">
        {presets_html}
      </div>
      <p style="margin-top: 12px;"><strong>직접 입력:</strong></p>
      <textarea id="message" placeholder="예: 부산 바다 보이는 감성 숙소 추천해줘">부산 바다 보이는 감성 숙소 추천해줘</textarea>
      <br/>
      <button id="run" onclick="runDemo()">그래프 실행</button>
      <div id="summary" class="summary-box" style="display:none;"></div>
      <div id="thread" class="thread"></div>
      <div id="answer" class="answer-bubble">여행 전문가의 답변이 여기에 표시됩니다. (누구와 / 어디로 / 테마 등을 나눠 말해도 이어서 반영됩니다)</div>
      <details style="margin-top:12px;"><summary>세부 정보 (who, why, filters 등)</summary><pre id="result" class="detail-json"></pre></details>

      <script>
        var FETCH_TIMEOUT_MS = 70000;
        var transcript = [];
        var lastFilters = {{}};
        document.querySelectorAll('.preset').forEach(function(btn) {{
          btn.onclick = function() {{ document.getElementById('message').value = this.dataset.msg; }};
        }});
        function renderSummary(filters) {{
          if (!filters) return;
          var who = filters.who, region = filters.region, theme = filters.theme, why = filters.why;
          var arr = [];
          if (who) arr.push('누구와: ' + who);
          if (region) arr.push('어디로: ' + region);
          if (theme && (Array.isArray(theme) ? theme.length : theme)) arr.push('테마: ' + (Array.isArray(theme) ? theme.join(', ') : theme));
          if (why) arr.push('목적: ' + why);
          var el = document.getElementById('summary');
          if (arr.length) {{ el.style.display = 'block'; el.textContent = '지금까지 파악한 정보: ' + arr.join(' · '); }}
          else {{ el.style.display = 'none'; }}
        }}
        function appendToThread(role, content) {{
          var div = document.createElement('div');
          div.className = role === 'user' ? 'bubble user' : 'bubble assistant';
          div.textContent = content;
          document.getElementById('thread').appendChild(div);
        }}
        async function runDemo() {{
          const message = document.getElementById('message').value;
          const answerBox = document.getElementById('answer');
          const resultBox = document.getElementById('result');
          answerBox.textContent = '요청 중... (최대 ' + (FETCH_TIMEOUT_MS/1000) + '초)';
          answerBox.className = 'answer-bubble';
          resultBox.textContent = '';
          var controller = new AbortController();
          var to = setTimeout(function() {{ controller.abort(); }}, FETCH_TIMEOUT_MS);
          try {{
            const res = await fetch('/api/v1/travel/travel', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ message: message, user_id: 123, conversation_history: transcript, previous_filters: lastFilters }}),
              signal: controller.signal
            }});
            clearTimeout(to);
            var data = await res.json();
            var reply = (data.response || '').trim();
            transcript.push({{ role: 'user', content: message }}, {{ role: 'assistant', content: reply }});
            lastFilters = data.filters || {{}};
            renderSummary(lastFilters);
            appendToThread('user', message);
            appendToThread('assistant', reply || '(답변 없음)');
            answerBox.textContent = reply || '(답변 없음)';
            answerBox.className = 'answer-bubble' + (data.error || !res.ok ? ' error' : '');
            resultBox.textContent = JSON.stringify(data, null, 2);
          }} catch (err) {{
            clearTimeout(to);
            var msg = err.name === 'AbortError' ? '요청 타임아웃 (' + (FETCH_TIMEOUT_MS/1000) + '초 초과)' : ('에러: ' + err);
            answerBox.textContent = msg;
            answerBox.className = 'answer-bubble error';
            resultBox.textContent = (err.stack || '') + '\\n' + msg;
          }}
        }}
      </script>
    </body>
    </html>
    """
