# -*- coding: utf-8 -*-
"""
🎵 중학생을 위한 음악 감상 & 데이터 대시보드
- 감정 기반 음악 추천 + 오디오 특징(Feature) 분석 + 감상 비평지 작성
- 사용 라이브러리: streamlit, pandas, numpy 만 사용 (외부 설치 최소화)
- 실행 방법: streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 0) 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="중학생 음악 감상 & 데이터 대시보드",
    page_icon="🎵",
    layout="wide",
)

# =========================================================
# 1) 가상 데이터셋(Mock Data) 정의
#    - 6곡, 다양한 장르
#    - Valence(밝기), Energy(에너지), Tempo(빠르기) : 0~100
#    - preview_url : st.audio에서 재생 가능한 mp3 링크 (SoundHelix 공개 데모)
#    - timeline    : 도입부 → 전개부 → 후렴구 → 브릿지 → 아웃트로 (에너지/템포 변화)
# =========================================================
@st.cache_data
def load_mock_data():
    """6곡의 가상 음악 데이터를 반환하는 함수"""
    songs = [
        {
            "title": "봄의 왈츠",
            "artist": "AI 클래식 챔버",
            "genre": "클래식",
            "valence": 78,   # 밝기 (장조 느낌)
            "energy": 45,    # 에너지 (셈여림)
            "tempo": 60,     # 빠르기 (Allegretto 정도)
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "timeline": {
                "section": ["도입부", "전개부", "후렴구", "브릿지", "아웃트로"],
                "energy":  [30, 50, 70, 55, 25],
                "tempo":   [55, 60, 65, 60, 50],
            },
        },
        {
            "title": "달빛 카페",
            "artist": "Midnight Jazz Trio",
            "genre": "재즈",
            "valence": 45,
            "energy": 35,
            "tempo": 40,
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
            "timeline": {
                "section": ["도입부", "전개부", "후렴구", "브릿지", "아웃트로"],
                "energy":  [25, 35, 50, 45, 20],
                "tempo":   [38, 40, 42, 40, 38],
            },
        },
        {
            "title": "여름밤의 댄스",
            "artist": "K-Pop Stars",
            "genre": "팝",
            "valence": 88,
            "energy": 92,
            "tempo": 85,
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
            "timeline": {
                "section": ["도입부", "전개부", "후렴구", "브릿지", "아웃트로"],
                "energy":  [60, 80, 95, 85, 55],
                "tempo":   [80, 85, 88, 85, 80],
            },
        },
        {
            "title": "비 오는 창가",
            "artist": "이서연",
            "genre": "발라드",
            "valence": 25,
            "energy": 30,
            "tempo": 35,
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
            "timeline": {
                "section": ["도입부", "전개부", "후렴구", "브릿지", "아웃트로"],
                "energy":  [15, 30, 55, 40, 15],
                "tempo":   [32, 35, 38, 35, 30],
            },
        },
        {
            "title": "Neon Pulse",
            "artist": "DJ Aurora",
            "genre": "일렉트로닉",
            "valence": 65,
            "energy": 95,
            "tempo": 95,
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
            "timeline": {
                "section": ["도입부", "전개부", "후렴구", "브릿지", "아웃트로"],
                "energy":  [50, 85, 98, 90, 60],
                "tempo":   [90, 95, 98, 95, 88],
            },
        },
        {
            "title": "숲속의 산책",
            "artist": "Acoustic Friends",
            "genre": "어쿠스틱",
            "valence": 70,
            "energy": 40,
            "tempo": 50,
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
            "timeline": {
                "section": ["도입부", "전개부", "후렴구", "브릿지", "아웃트로"],
                "energy":  [25, 40, 55, 45, 30],
                "tempo":   [45, 50, 52, 50, 45],
            },
        },
    ]
    return songs


SONGS = load_mock_data()
DF = pd.DataFrame(SONGS)


# =========================================================
# 2) 음악 교과서 용어 매핑
# =========================================================
def tempo_term(tempo_score: int) -> str:
    """0~100 점수를 음악 교과서의 '빠르기말'로 변환"""
    if tempo_score < 25:
        return "Largo (아주 느리게)"
    elif tempo_score < 45:
        return "Andante (느리게/걷는 빠르기)"
    elif tempo_score < 65:
        return "Moderato (보통 빠르기)"
    elif tempo_score < 85:
        return "Allegro (빠르게)"
    else:
        return "Presto (매우 빠르게)"


def valence_term(v: int) -> str:
    """밝기 점수를 장조/단조 느낌으로 변환"""
    if v >= 70:
        return "장조의 밝고 경쾌한 느낌"
    elif v >= 45:
        return "장조와 단조가 섞인 차분한 느낌"
    else:
        return "단조의 어둡고 쓸쓸한 느낌"


def energy_term(e: int) -> str:
    """에너지 점수를 셈여림으로 변환"""
    if e < 25:
        return "pp ~ p (아주 여리게)"
    elif e < 50:
        return "mp ~ mf (조금 여리게/조금 세게)"
    elif e < 75:
        return "f (세게)"
    else:
        return "ff (아주 세게)"


# =========================================================
# 3) 추천 알고리즘: 유클리드 거리
# =========================================================
def recommend_songs(user_v: int, user_e: int, user_t: int, top_k: int = 2):
    """학생 슬라이더 값과 가장 가까운 곡 top_k 추천"""
    user_vec = np.array([user_v, user_e, user_t])
    distances = []
    for i, song in enumerate(SONGS):
        song_vec = np.array([song["valence"], song["energy"], song["tempo"]])
        dist = float(np.linalg.norm(user_vec - song_vec))
        distances.append((i, dist))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]


# =========================================================
# 4) 세션 상태 초기화
# =========================================================
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "final_report" not in st.session_state:
    st.session_state.final_report = ""


def append_chip(word: str):
    """도우미 단어 칩을 누르면 감상평 텍스트에 단어 자동 추가"""
    current = st.session_state.review_text.rstrip()
    if current and not current.endswith((",", ".", " ")):
        current += ", "
    elif current:
        current += " "
    st.session_state.review_text = current + word


# =========================================================
# [레이아웃 1] 앱 타이틀 & 설명
# =========================================================
st.title("🎵 중학생을 위한 음악 감상 & 데이터 대시보드")
st.markdown(
    """
    > 음악을 **귀로만** 듣지 않고, **데이터로도** 만나보는 시간!  
    > 슬라이더로 내 기분을 표현하면 어울리는 곡을 추천받고,  
    > 곡의 **밝기 · 에너지 · 빠르기** 를 그래프로 살펴본 뒤  
    > 나만의 **감상 비평지**까지 완성해 봅시다.
    """
)
st.divider()


# =========================================================
# [레이아웃 2] 음악 퀴즈
# =========================================================
with st.expander("🧩 오늘의 음악 퀴즈 힌트 — Guess the Feature!", expanded=False):
    quiz_song = SONGS[0]  # 첫 번째 곡(봄의 왈츠)으로 퀴즈
    st.markdown(
        f"**문제 곡:** {quiz_song['title']} — *{quiz_song['artist']}* ({quiz_song['genre']})"
    )
    st.audio(quiz_song["preview_url"])
    st.caption("🎧 위 곡을 들어보고, 이 곡의 '밝기'와 '에너지' 점수가 얼마일지 예상해 보세요!")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        guess_v = st.slider("내가 예상한 밝기 (Valence)", 0, 100, 50, key="quiz_v")
    with col_q2:
        guess_e = st.slider("내가 예상한 에너지 (Energy)", 0, 100, 50, key="quiz_e")

    if st.button("✅ 정답 확인", key="quiz_check"):
        real_v = quiz_song["valence"]
        real_e = quiz_song["energy"]
        diff_v = abs(real_v - guess_v)
        diff_e = abs(real_e - guess_e)
        avg_diff = (diff_v + diff_e) / 2

        st.markdown(
            f"- 실제 밝기: **{real_v}점** / 내 예상: {guess_v}점 (차이 {diff_v})\n"
            f"- 실제 에너지: **{real_e}점** / 내 예상: {guess_e}점 (차이 {diff_e})"
        )
        if avg_diff <= 10:
            st.success("🏆 대단해요! 평균 차이 10점 이내 — 음악 귀가 정말 좋네요!")
        elif avg_diff <= 25:
            st.info("👍 좋아요! 곡의 분위기를 잘 잡아냈어요.")
        else:
            st.warning("🤔 조금 더 집중해서 들어볼까요? 다른 곡으로도 연습해 보세요!")

st.divider()


# =========================================================
# [레이아웃 3] 사이드바 — 감정 슬라이더 & 음악 추천
# =========================================================
st.sidebar.header("🎚️ 내 기분 표현하기")
st.sidebar.caption("슬라이더를 움직여 지금 듣고 싶은 음악의 느낌을 표현해 보세요.")

user_valence = st.sidebar.slider(
    "1. 음악의 밝기 (어둡고 슬픔 0 ↔ 100 밝고 경쾌함)", 0, 100, 60,
)
user_energy = st.sidebar.slider(
    "2. 음악의 에너지 (잔잔함 0 ↔ 100 역동적/강렬함)", 0, 100, 50,
)
user_tempo = st.sidebar.slider(
    "3. 음악의 빠르기 (매우 느림 0 ↔ 100 매우 빠름)", 0, 100, 50,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**현재 내 기분 벡터**  \n"
    f"- 밝기: `{user_valence}` ({valence_term(user_valence)})  \n"
    f"- 에너지: `{user_energy}` ({energy_term(user_energy)})  \n"
    f"- 빠르기: `{user_tempo}` ({tempo_term(user_tempo)})"
)

st.subheader("🎯 내 기분에 어울리는 추천 곡")
recs = recommend_songs(user_valence, user_energy, user_tempo, top_k=2)

rec_cols = st.columns(2)
for col, (idx, dist) in zip(rec_cols, recs):
    song = SONGS[idx]
    with col:
        st.markdown(f"### 🎵 {song['title']}")
        st.markdown(
            f"- 아티스트: **{song['artist']}**\n"
            f"- 장르: `{song['genre']}`\n"
            f"- 기분과의 거리: `{dist:.1f}` (작을수록 비슷함)"
        )
        st.audio(song["preview_url"])

st.divider()


# =========================================================
# [레이아웃 4] 오디오 특징 시각화
# =========================================================
st.subheader("📊 오디오 특징 분석 — 데이터로 음악 들여다보기")

song_titles = [s["title"] for s in SONGS]
default_idx = recs[0][0]
selected_title = st.selectbox(
    "분석하고 싶은 곡을 골라보세요", song_titles, index=default_idx,
)
selected_song = next(s for s in SONGS if s["title"] == selected_title)

# (1) 수치 카드 + 음악 교과서 용어
m1, m2, m3 = st.columns(3)
m1.metric(
    label="🌞 밝기 (Valence) — 장조/단조의 느낌",
    value=f"{selected_song['valence']} / 100",
    delta=valence_term(selected_song["valence"]),
    delta_color="off",
)
m2.metric(
    label="💪 에너지 (Energy) — 셈여림",
    value=f"{selected_song['energy']} / 100",
    delta=energy_term(selected_song["energy"]),
    delta_color="off",
)
m3.metric(
    label="⏱️ 템포 (Tempo) — 빠르기말",
    value=f"{selected_song['tempo']} / 100",
    delta=tempo_term(selected_song["tempo"]),
    delta_color="off",
)

# (2) 막대그래프
st.markdown("#### 🎼 곡 전체 특징 (Bar Chart)")
feature_df = pd.DataFrame(
    {"점수": [selected_song["valence"], selected_song["energy"], selected_song["tempo"]]},
    index=["밝기(Valence)", "에너지(Energy)", "템포(Tempo)"],
)
st.bar_chart(feature_df, height=260)

# (3) 섹션별 타임라인
st.markdown("#### 📈 곡의 진행에 따른 에너지·템포 변화 (Timeline)")
timeline_df = pd.DataFrame(
    {
        "에너지(Energy)": selected_song["timeline"]["energy"],
        "템포(Tempo)":   selected_song["timeline"]["tempo"],
    },
    index=selected_song["timeline"]["section"],
)
st.line_chart(timeline_df, height=280)
st.caption(
    "💡 도입부 → 전개부 → 후렴구 → 브릿지 → 아웃트로 순으로, "
    "보통 후렴구에서 에너지가 가장 커집니다(클라이맥스)."
)

st.divider()


# =========================================================
# [레이아웃 5] 감상 비평지 작성 및 저장
# =========================================================
st.subheader("✍️ 나만의 감상 비평지 작성하기")
st.caption("아래 단어 칩을 누르면 감상평에 자동으로 단어가 추가돼요. 자유롭게 이어 써보세요!")

# 도우미 단어 칩
chip_words = [
    "웅장한", "경쾌한", "우아한", "쓸쓸한",
    "몽환적인", "박진감 있는", "포근한", "신비로운",
    "역동적인", "잔잔한", "감미로운", "긴장감 있는",
]
chip_cols = st.columns(4)
for i, word in enumerate(chip_words):
    with chip_cols[i % 4]:
        st.button(
            f"+ {word}",
            key=f"chip_{i}",
            on_click=append_chip,
            args=(word,),
            use_container_width=True,
        )

# 감상평 입력창
st.text_area(
    "📝 감상평 (자유롭게 작성)",
    key="review_text",
    height=150,
    placeholder="예) 후렴구에서 에너지가 확 올라가서 가슴이 두근거렸다. 장조의 밝은 느낌이 봄과 잘 어울린다...",
)

# 감상 비평지 완성 버튼
if st.button("📜 감상 비평지 완성하기", type="primary"):
    chart_summary_lines = [
        f"- 도입부 에너지: {selected_song['timeline']['energy'][0]}",
        f"- 후렴구 에너지: {selected_song['timeline']['energy'][2]} (최고점)",
        f"- 아웃트로 에너지: {selected_song['timeline']['energy'][-1]}",
    ]
    chart_summary = "\n".join(chart_summary_lines)

    report = f"""🎵 감상 비평지
====================================

[선택한 음악 정보]
- 제목      : {selected_song['title']}
- 아티스트  : {selected_song['artist']}
- 장르      : {selected_song['genre']}

[데이터 차트 요약]
- 밝기(Valence) : {selected_song['valence']} / 100  → {valence_term(selected_song['valence'])}
- 에너지(Energy): {selected_song['energy']} / 100  → {energy_term(selected_song['energy'])}
- 템포(Tempo)   : {selected_song['tempo']} / 100  → {tempo_term(selected_song['tempo'])}

[섹션별 에너지 변화]
{chart_summary}

[내 기분 슬라이더 값]
- 밝기 {user_valence} / 에너지 {user_energy} / 빠르기 {user_tempo}

[내가 쓴 감상평]
{st.session_state.review_text.strip() or "(아직 작성되지 않았습니다)"}

====================================
"""
    st.session_state.final_report = report

# 비평지 미리보기 + 다운로드
if st.session_state.final_report:
    st.markdown("### 📄 완성된 감상 비평지 미리보기")
    st.code(st.session_state.final_report, language="markdown")

    # TXT 다운로드
    st.download_button(
        label="⬇️ 텍스트(.txt)로 저장하기",
        data=st.session_state.final_report.encode("utf-8"),
        file_name=f"감상비평지_{selected_song['title']}.txt",
        mime="text/plain",
    )

    # CSV 다운로드
    csv_df = pd.DataFrame([{
        "제목": selected_song["title"],
        "아티스트": selected_song["artist"],
        "장르": selected_song["genre"],
        "밝기(Valence)": selected_song["valence"],
        "에너지(Energy)": selected_song["energy"],
        "템포(Tempo)": selected_song["tempo"],
        "내_기분_밝기": user_valence,
        "내_기분_에너지": user_energy,
        "내_기분_빠르기": user_tempo,
        "감상평": st.session_state.review_text.strip(),
    }])
    csv_buf = io.StringIO()
    csv_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ 표 데이터(.csv)로 저장하기",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"감상비평지_{selected_song['title']}.csv",
        mime="text/csv",
    )

st.divider()
st.caption("© 음악 수업용 데모 — Streamlit으로 만든 감상 대시보드")
