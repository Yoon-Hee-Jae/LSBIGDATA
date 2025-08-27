import seaborn as sns
from shared import df
from shinyswatch import theme
from shiny.express import input, render, ui
from shiny import App, reactive
import matplotlib as mpl
import matplotlib.font_manager as fm
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False 

# 마이너스 깨짐 방지
mpl.rcParams["axes.unicode_minus"] = False

ui.page_opts(title="팔머펭귄 데이터 대시보드", theme=theme.darkly)

with ui.sidebar(open='desktop'):
    "변수를 선택해주세요"
    ui.input_select('var1', 'x축', choices={"bill_length_mm":"부리길이",
    "bill_depth_mm": "부리깊이",
    "flipper_length_mm": "날개길이",
    "body_mass_g": "몸무게"
    },
    selected="bill_length_mm"
    )

    ui.input_select(
        'var2', 'y축', choices={"bill_length_mm":"부리길이",
    "bill_depth_mm": "부리깊이",
    "flipper_length_mm": "날개길이",
    "body_mass_g": "몸무게"
    },
    selected="bill_depth_mm"
    )

    ui.input_checkbox_group(
        "var3", "종 선택", choices={"Adelie":"아델리",
    "Gentoo": "젠투",
    "Chinstrap": "친스트랩"
    }, selected=list(df["species"].unique())
    )

    ui.input_action_button("apply", "Apply")

# 사용자가 Apply를 눌렀을 때만 반영
filtered_df = reactive.Value(df)
input_1 = reactive.Value('bill_length_mm')
input_2 = reactive.Value('bill_depth_mm')

@reactive.effect
@reactive.event(input.apply)

def _():
    filtered_df.set(df[df["species"].isin(input.var3())])
    input_1.set(input.var1())
    input_2.set(input.var2())

with ui.nav_panel("Page 1"):
    with ui.layout_columns():
        with ui.card():
            @render.plot
            def hist():
                species_labels = {
                "Adelie": "아델리",
                "Chinstrap": "친스트랩",
                "Gentoo": "젠투"
                }
                spe_palette = {
                "Adelie": "#FF7F0E",    # 주황
                "Chinstrap": "#1F77B4", # 파랑
                "Gentoo": "#2CA02C"     # 초록
                }
                p = sns.scatterplot(
                                data=filtered_df(),
                                x=input_1(),
                                y=input_2(),
                                hue='species',
                                palette=spe_palette
                                )
                # legend 한글로 바꾸기
                handles, labels = p.get_legend_handles_labels()
                # labels는 영어 species list
                labels = [species_labels.get(l, l) for l in labels]  # 없는 건 그대로
                p.legend(handles=handles, labels=labels, title='종 이름')
                x_choices = {
                    "bill_length_mm": "부리길이",
                    "bill_depth_mm": "부리깊이",
                    "flipper_length_mm": "날개길이",
                    "body_mass_g": "몸무게"
                }
                y_choices = x_choices  # 동일 구조

                p.set_xlabel(x_choices[input_1()])  # friendly label
                p.set_ylabel(y_choices[input_2()])                
                return p
            
        with ui.card():
            @render.data_frame
            def data():
                dff = filtered_df()
                return dff[['species','island',input.var1()]]

with ui.nav_panel("Page 2"):
    "두번째 페이지"

                

