
# 작성한 app.py가 문법적으로 유효한지(파이썬 파싱 가능) 확인하기 위한 단계
# 1) app.py 전체 코드를 문자열로 작성
# 2) compile()로 SyntaxError가 없는지 점검
# 3) 파일로 저장하여 첨부 가능 상태로 만든다

app_code = r'''# -*- coding: utf-8 -*-
"""
🎵 중학생을 위한 음악 감상 & 데이터 대시보드
- 단일 파일(app.py) Streamlit 앱
- 사용 라이브러리: streamlit, pandas, numpy (추가 설치 불필요)
- 실행:  streamlit run app.py
- 배포:  Streamlit Cloud (requirements.txt에 streamlit, pandas, numpy 명시)
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 0) 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="중학생 음악 대시보드",
    page_icon="🎵",
    layout="wide",
)

# =========================================================
# 1) 가상 데이터셋(Mock Data) 정의
#    - 다양한 장르의 6곡
#    - 밝기(Valence), 에너지(Energy), 템포(Tempo) 0~100 점수
#    - 미리듣기용 샘플 오디오 URL (SoundHelix 공개 샘플)
#    - 섹션별(도입부/후렴구/아웃트로) 에너지·템포 타임라인 데이터
# =========================================================
@st.cache_data
def load_mock_songs() -> pd.DataFrame:
    """가상 음악 데이터셋을 DataFrame으로 반환한다."""
    songs = [
        {
            "title": "봄의 왈츠",
            "artist": "가상 오케스트라",
            "genre": "클래식",
            "valence": 80,   # 밝기: 장조 느낌, 밝고 따뜻
            "energy": 45,    # 에너지: 중간 정도 셈여림
            "tempo": 60,     # 빠르기: Moderato 수준
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "timeline": {
                "sections": ["도입부", "전개부", "후렴구", "전환부", "아웃트로"],
                "energy":   [30, 55, 75, 60, 35],
                "tempo":    [55, 60, 65, 62, 55],
            },
        },
        {
            "title": "달빛 소나타 by Night",
            "artist": "Lunar Trio",
            "genre": "재즈",
            "valence": 35,
            "energy": 30,
            "tempo": 40,
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
            "timeline": {
                "sections": ["도입부", "전개부", "후렴구", "전환부", "아웃트로"],
                "energy":   [20, 30, 45, 35, 15],
                "tempo":    [38, 40, 45, 42, 38],
            },
        },
        {
            "title": "한여름 햇살",
            "artist": "Sunny Pop",
            "genre": "팝",
            "valence": 90,
            "energy": 85,
            "tempo": 80,
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
            "timeline": {
                "sections": ["도입부", "전개부", "후렴구", "전환부", "아웃트로"],
                "energy":   [50, 70, 95, 80, 60],
                "tempo":    [78, 80, 85, 82, 78],
            },
        },
        {
            "title": "비 오는 거리",
            "artist": "안개의 노래",
            "genre": "발라드",
            "valence": 20,
            "energy": 25,
            "tempo": 35,
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
            "timeline": {
                "sections": ["도입부", "전개부", "후렴구", "전환부", "아웃트로"],
                "energy":   [15, 25, 40, 30, 15],
                "tempo":    [32, 35, 38, 35, 32],
            },
        },
        {
            "title": "전자 우주 여행",
            "artist": "Neon Pulse",
            "genre": "일렉트로닉",
            "valence": 65,
            "energy": 95,
            "tempo": 95,
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
            "timeline": {
                "sections": ["도입부", "전개부", "후렴구", "전환부", "아웃트로"],
                "energy":   [60, 85, 98, 90, 70],
                "tempo":    [90, 95, 98, 95, 90],
            },
        },
        {
            "title": "산책길의 기타",
            "artist": "Acoustic Days",
            "genre": "어쿠스틱",
            "valence": 70,
            "energy": 40,
            "tempo": 55,
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
            "timeline": {
                "sections": ["도입부", "전개부", "후렴구", "전환부", "아웃트로"],
                "energy":   [25, 40, 55, 45, 30],
                "tempo":    [52, 55, 58, 55, 50],
            },
        },
    ]
    return pd.DataFrame(songs)


# 음악 교과서 용어 매핑 (수치 옆에 병기하기 위함)
TERM_MAP = {
    "valence": "Valence (장조/단조의 느낌, 밝기)",
    "energy":  "Energy (셈여림, 다이내믹)",
    "tempo":   "Tempo (빠르기말, BPM 수준)",
}

# 템포 수치를 음악 용어(빠르기말)로 환산하는 헬퍼
def tempo_to_term(tempo_score: float) -> str:
    """0~100 템포 점수를 한국어 빠르기말로 변환한다."""
    if tempo_score < 20:
        return "Largo (아주 느리게)"
    elif tempo_score < 40:
        return "Adagio (느리게)"
    elif tempo_score < 60:
        return "Andante (걷는 빠르기)"
    elif tempo_score < 75:
        return "Moderato (보통 빠르기)"
    elif tempo_score < 90:
        return "Allegro (빠르게)"
    else:
        return "Presto (매우 빠르게)"


# 밝기 수치를 장/단조 느낌으로 환산
def valence_to_term(v: float) -> str:
    if v < 30:
        return "단조적·어두움"
    elif v < 55:
        return "차분함"
    elif v < 75:
        return "밝은 편"
    else:
        return "장조적·매우 밝음"


# 에너지 수치를 셈여림으로 환산
def energy_to_dynamics(e: float) -> str:
    if e < 25:
        return "pp ~ p (여리게)"
    elif e < 50:
        return "mp ~ mf (보통)"
    elif e < 75:
        return "f (세게)"
    else:
        return "ff (매우 세게)"


# =========================================================
# 2) 세션 스테이트 초기화
#    - 감상평 텍스트, 퀴즈 선택 곡, 비평지 출력 여부 등을 저장
# =========================================================
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "show_critique" not in st.session_state:
    st.session_state.show_critique = False
if "quiz_song_idx" not in st.session_state:
    # 퀴즈 곡은 한 번 정해지면 새로고침 전에는 고정되도록 무작위 선택
    rng = np.random.default_rng(seed=None)
    st.session_state.quiz_song_idx = int(rng.integers(0, 6))


# 데이터 로드
df_songs = load_mock_songs()

# =========================================================
# 레이아웃 1: 앱 타이틀 & 설명
# =========================================================
st.title("🎵 중학생을 위한 음악 감상 & 데이터 대시보드")
st.markdown(
    """
    > 음악도 **데이터**로 표현할 수 있어요!  
    > 밝기·에너지·빠르기 같은 **오디오 특징(Feature)** 을 직접 조절해 보고,  
    > 내 감정에 맞는 음악을 추천받아 들으며 **감상 비평지** 까지 완성해 봅시다.
    """
)
st.markdown("---")

# =========================================================
# 사이드바: 레이아웃 3의 감정 슬라이더
# =========================================================
st.sidebar.header("🎚️ 내 감정 슬라이더")
st.sidebar.caption("지금 듣고 싶은 음악의 느낌을 조절해 보세요.")

user_valence = st.sidebar.slider(
    "1. 음악의 밝기  (어둡고 슬픔 ↔ 밝고 경쾌함)",
    min_value=0, max_value=100, value=60, step=1,
)
user_energy = st.sidebar.slider(
    "2. 음악의 에너지  (잔잔함 ↔ 역동적·강렬함)",
    min_value=0, max_value=100, value=50, step=1,
)
user_tempo = st.sidebar.slider(
    "3. 음악의 빠르기  (매우 느림 ↔ 매우 빠름)",
    min_value=0, max_value=100, value=55, step=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**현재 설정 요약**  \n"
    f"- 밝기: `{user_valence}` → {valence_to_term(user_valence)}  \n"
    f"- 에너지: `{user_energy}` → {energy_to_dynamics(user_energy)}  \n"
    f"- 빠르기: `{user_tempo}` → {tempo_to_term(user_tempo)}"
)

# =========================================================
# 레이아웃 2: 음악 퀴즈 (Guess the Feature!)
# =========================================================
with st.expander("🧩 오늘의 음악 퀴즈 힌트 (Guess the Feature!)", expanded=False):
    st.markdown(
        "아래 곡을 듣고, 이 곡의 **밝기(Valence)** 와 **에너지(Energy)** 점수가 "
        "몇 점일지 직접 맞춰보세요! (0~100점)"
    )

    quiz_song = df_songs.iloc[st.session_state.quiz_song_idx]
    st.markdown(f"**🎧 퀴즈 곡:** *{quiz_song['title']}* — {quiz_song['artist']} ({quiz_song['genre']})")
    st.audio(quiz_song["audio_url"])

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        guess_valence = st.slider("내가 생각하는 밝기 점수", 0, 100, 50, key="guess_valence")
    with col_q2:
        guess_energy = st.slider("내가 생각하는 에너지 점수", 0, 100, 50, key="guess_energy")

    if st.button("✅ 정답 확인", key="check_quiz"):
        diff_v = abs(guess_valence - quiz_song["valence"])
        diff_e = abs(guess_energy - quiz_song["energy"])
        avg_diff = (diff_v + diff_e) / 2

        st.write(
            f"- 실제 밝기: **{quiz_song['valence']}점** / 내 예측: **{guess_valence}점** "
            f"→ 차이 `{diff_v}`"
        )
        st.write(
            f"- 실제 에너지: **{quiz_song['energy']}점** / 내 예측: **{guess_energy}점** "
            f"→ 차이 `{diff_e}`"
        )

        if avg_diff <= 10:
            st.success("🎉 음악적 직관이 뛰어나요! 거의 정확하게 맞췄어요.")
        elif avg_diff <= 25:
            st.info("👍 꽤 비슷해요! 조금만 더 귀를 기울이면 완벽해질 거예요.")
        else:
            st.warning("🙂 다음엔 더 정확히 맞춰봐요. 셈여림과 분위기에 집중해 보세요.")

st.markdown("---")

# =========================================================
# 레이아웃 3: 감정 슬라이더 기반 음악 추천
#  - 사용자 입력값과 각 곡의 특징 벡터의 유클리드 거리를 계산
#  - 가장 가까운 2곡을 추천
# =========================================================
st.header("🎯 내 감정에 맞는 음악 추천")

user_vec = np.array([user_valence, user_energy, user_tempo], dtype=float)
song_vecs = df_songs[["valence", "energy", "tempo"]].to_numpy(dtype=float)
distances = np.linalg.norm(song_vecs - user_vec, axis=1)

df_rec = df_songs.copy()
df_rec["거리"] = distances
df_rec = df_rec.sort_values("거리").reset_index(drop=True)

top2 = df_rec.head(2)

st.markdown(
    f"슬라이더 값 **(밝기 {user_valence} / 에너지 {user_energy} / 빠르기 {user_tempo})** "
    f"에 가장 가까운 추천곡 **2곡** 입니다."
)

rec_cols = st.columns(2)
for i, (_, row) in enumerate(top2.iterrows()):
    with rec_cols[i]:
        st.subheader(f"#{i+1}  {row['title']}")
        st.caption(f"{row['artist']} · {row['genre']}")
        st.markdown(
            f"- 밝기: **{row['valence']}** ({valence_to_term(row['valence'])})  \n"
            f"- 에너지: **{row['energy']}** ({energy_to_dynamics(row['energy'])})  \n"
            f"- 빠르기: **{row['tempo']}** ({tempo_to_term(row['tempo'])})  \n"
            f"- 내 감정과의 거리: `{row['거리']:.1f}`"
        )
        st.audio(row["audio_url"])

st.markdown("---")

# =========================================================
# 레이아웃 4: 오디오 특징 시각화
#  - 사용자가 분석할 곡을 직접 선택 가능 (기본값: 추천 1순위)
#  - 막대그래프: 밝기/에너지/템포 비교
#  - 선그래프: 곡 진행에 따른 에너지·템포 타임라인
# =========================================================
st.header("📊 오디오 특징(Feature) 분석")

song_options = df_songs["title"].tolist()
default_idx = song_options.index(top2.iloc[0]["title"])
selected_title = st.selectbox(
    "분석할 곡을 선택하세요",
    options=song_options,
    index=default_idx,
)
selected_song = df_songs[df_songs["title"] == selected_title].iloc[0]

# 4-1) 음악적 특징 막대그래프
st.subheader("🎼 음악적 특징 점수 (0~100)")

feature_df = pd.DataFrame(
    {
        "점수": [
            selected_song["valence"],
            selected_song["energy"],
            selected_song["tempo"],
        ]
    },
    index=[
        TERM_MAP["valence"],
        TERM_MAP["energy"],
        TERM_MAP["tempo"],
    ],
)
st.bar_chart(feature_df, height=260)

# 수치 옆에 음악 용어 병기
c1, c2, c3 = st.columns(3)
c1.metric("Valence (밝기)", f"{selected_song['valence']}점", valence_to_term(selected_song["valence"]))
c2.metric("Energy (셈여림)", f"{selected_song['energy']}점", energy_to_dynamics(selected_song["energy"]))
c3.metric("Tempo (빠르기말)", f"{selected_song['tempo']}점", tempo_to_term(selected_song["tempo"]))

# 4-2) 곡 진행에 따른 타임라인 선그래프
st.subheader("⏱️ 곡 진행 타임라인 (도입부 → 후렴구 → 아웃트로)")

tl = selected_song["timeline"]
timeline_df = pd.DataFrame(
    {
        "에너지 (셈여림)": tl["energy"],
        "템포 (빠르기)":   tl["tempo"],
    },
    index=tl["sections"],
)
st.line_chart(timeline_df, height=280)
st.caption(
    "선그래프는 곡이 시작해서 끝날 때까지 **셈여림(Energy)** 과 **빠르기(Tempo)** 가 "
    "어떻게 변하는지를 보여줍니다. 후렴구에서 에너지가 가장 커지는 패턴을 관찰해 보세요."
)

st.markdown("---")

# =========================================================
# 레이아웃 5: 감상 비평지 작성 및 저장
# =========================================================
st.header("📝 감상 비평지 작성")

st.markdown("**감상평 도우미 단어**를 눌러 텍스트 상자에 단어를 추가할 수 있어요.")

# 도우미 단어 칩 (버튼)
helper_words = [
    "웅장한", "경쾌한", "우아한", "쓸쓸한", "몽환적인",
    "잔잔한", "강렬한", "따뜻한", "차가운", "신비로운",
    "희망찬", "긴장감 있는",
]

# 버튼은 한 줄에 6개씩 배치
def add_word(word: str):
    """버튼 클릭 시 감상평 텍스트에 단어를 덧붙인다."""
    current = st.session_state.review_text
    sep = " " if current and not current.endswith(" ") else ""
    st.session_state.review_text = f"{current}{sep}{word}"

chip_rows = [helper_words[i:i+6] for i in range(0, len(helper_words), 6)]
for row in chip_rows:
    cols = st.columns(len(row))
    for col, w in zip(cols, row):
        col.button(w, key=f"chip_{w}", on_click=add_word, args=(w,))

# 감상평 입력
st.text_area(
    "감상평을 자유롭게 작성해 보세요",
    key="review_text",
    height=140,
    placeholder="예) 이 곡은 후렴구에서 에너지가 크게 올라가서 마음이 두근거렸다...",
)

# 비평지 완성 버튼
if st.button("📄 감상 비평지 완성"):
    st.session_state.show_critique = True

# 비평지 출력
if st.session_state.show_critique:
    st.markdown("### 🧾 나의 감상 비평지")

    critique_md = f"""
**🎵 곡 정보**
- 제목: **{selected_song['title']}**
- 아티스트: {selected_song['artist']}
- 장르: {selected_song['genre']}

**📊 데이터 차트 요약**
- 밝기 (Valence): **{selected_song['valence']}점** — {valence_to_term(selected_song['valence'])}
- 에너지 (Energy): **{selected_song['energy']}점** — {energy_to_dynamics(selected_song['energy'])}
- 빠르기 (Tempo): **{selected_song['tempo']}점** — {tempo_to_term(selected_song['tempo'])}
- 곡 진행 패턴: {', '.join(f"{s}({e})" for s, e in zip(tl['sections'], tl['energy']))}

**✍️ 나의 감상평**
{st.session_state.review_text if st.session_state.review_text.strip() else "_(감상평이 비어 있습니다)_"}
"""
    st.markdown(critique_md)

    # 다운로드용 텍스트 파일 생성
    txt_bytes = critique_md.encode("utf-8")

    # 다운로드용 CSV (한 행 요약)
    csv_df = pd.DataFrame([{
        "title":   selected_song["title"],
        "artist":  selected_song["artist"],
        "genre":   selected_song["genre"],
        "valence": selected_song["valence"],
        "energy":  selected_song["energy"],
        "tempo":   selected_song["tempo"],
        "review":  st.session_state.review_text,
    }])
    csv_buf = io.StringIO()
    csv_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    csv_bytes = csv_buf.getvalue().encode("utf-8-sig")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ 감상 비평지 TXT 다운로드",
            data=txt_bytes,
            file_name=f"감상비평지_{selected_song['title']}.txt",
            mime="text/plain",
        )
    with d2:
        st.download_button(
            label="⬇️ 감상 비평지 CSV 다운로드",
            data=csv_bytes,
            file_name=f"감상비평지_{selected_song['title']}.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("ⓒ 음악과 데이터 융합 수업용 데모 · Streamlit으로 만든 교실용 대시보드")
'''

# 문법 검증
try:
    compile(app_code, "app.py", "exec")
    print("OK: app.py compiles (no SyntaxError).")
except SyntaxError as e:
    print("SyntaxError:", e)

# 파일로 저장
with open("/tmp/app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

import os
print("size:", os.path.getsize("/tmp/app.py"), "bytes")
print("lines:", app_code.count("\n"))
