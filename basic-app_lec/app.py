from shiny.express import ui

ui.page_opts(fillable=True)

"ui.layout_columns()"

with ui.layout_columns(col_widths=(4,4,4)):
    with ui.card():  
        ui.card_header("Card with sidebar")
        with ui.layout_sidebar():  
            with ui.sidebar(bg="#d02828"):  
                "Sidebar"  
            "Card content"
    with ui.card():
        ui.card_header("Card with sidebar")
        with ui.layout_sidebar():  
            with ui.sidebar(bg="#514cea"):  
                "Sidebar"  
            "Card content"
    with ui.card():
        ui.card_header("Card with sidebar")
        with ui.layout_sidebar():  
            with ui.sidebar(bg="#a2e24e"):  
                "Sidebar"  
            "Card content"