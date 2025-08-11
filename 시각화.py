import pandas as pd
import plotly.express as px

# 업로드된 파일 경로 사용
file_path = "7월 인구현황(게시용).xlsx"

# 원본 읽기
raw = pd.read_excel(file_path,header=None)

# 헤더 행 결합
h2 = raw.iloc[2].astype(str).fillna('')
h3 = raw.iloc[3].astype(str).fillna('')

cols = []
for a, b in zip(h2, h3):
    a = '' if a == 'nan' else a.strip()
    b = '' if b == 'nan' else b.strip()
    name = "_".join([x for x in [a, b] if x])
    cols.append(name if name else 'col')

# 데이터 시작
df = raw.iloc[4:].copy()
df.columns = cols
# 필요한 컬럼 탐색
region_col = next((c for c in df.columns if ('구' in c and '분' in c)), df.columns[0])
foreign_col = next((c for c in df.columns if '외국인' in c and ('계' in c or c.endswith('_계'))), None)
rate_col = next((c for c in df.columns if ('증감' in c and ('율' in c or '률' in c))), None)

if foreign_col is None or rate_col is None:
    raise ValueError("필요한 컬럼(외국인 인구 '계' 또는 '증감율')을 찾지 못했습니다.")

# 전처리
out = df[[region_col, foreign_col, rate_col]].rename(
    columns={region_col: '구분', foreign_col: '외국인_계', rate_col: '외국인_증감율'}
).dropna(subset=['구분'])

for c in ['외국인_계', '외국인_증감율']:
    out[c] = pd.to_numeric(out[c].astype(str).str.replace(',', ''), errors='coerce')

out = out[~out['구분'].astype(str).str.contains('합계')]

# 그래프
fig = px.bar(
    out,
    x='구분',
    y='외국인_증감율',
    title='구분별 전월 대비 외국인 인구 증감율',
    text=out['외국인_증감율'].apply(lambda v: f"{v:.2f}%")
)
fig.update_traces(textposition='outside')
fig.show()
