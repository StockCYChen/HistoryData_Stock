import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import calendar
import os
import re
from datetime import datetime
import threading

# 改用支援中文字體（SimHei , Microsoft YaHei, Microsoft JhengHei, MingLiU）
font_tkinter = 'Microsoft YaHei'
plt.rcParams['font.family'] = font_tkinter

# 計算技術指標 (KD, 移動平均線(MA5, MA10, MA20, MA60))
def compute_indicators(df):
    # DEBUG
    print("執行ftn call \"compute_indicators\"")
    # DEBUG
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()

    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    return df

# 繪圖功能（含K棒）
def plot_chart(df, symbol):
    # DEBUG
    print("執行ftn call \"plot_chart\"")
    # DEBUG
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(f'{symbol} 股價走勢圖', fontsize=18)

    # 畫出K棒
    width = 0.6
    color_up = 'red'
    color_down = 'green'
    for i, (date, row) in enumerate(df.iterrows()):
        color = color_up if row['Close'] >= row['Open'] else color_down
        ax1.plot([date, date], [row['Low'], row['High']], color=color)
        ax1.add_patch(Rectangle((mdates.date2num(date) - width/2, min(row['Open'], row['Close'])), width, abs(row['Close'] - row['Open']), color=color))

    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        ax1.plot(df.index, df[ma], label=ma)
    ax1.set_ylabel('股價')
    ax1.legend()
    ax1.grid(True)

    colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
    ax2.bar(df.index, df['Volume'], color=colors, label='Volume')
    ax2.set_ylabel('成交量')
    ax2.grid(True)

    ax3.plot(df.index, df['K'], label='K', color='blue')
    ax3.plot(df.index, df['D'], label='D', color='orange')
    ax3.set_ylabel('KD 指標')
    ax3.set_xlabel('日期')
    ax3.legend()
    ax3.grid(True)

    fig.autofmt_xdate()
    return fig

# 主下載與圖表邏輯，圖表處理搬至主執行緒中顯示
def fetch_and_prepare_data():
    # DEBUG
    print("執行ftn call \"fetch_and_prepare_data\"")
    # DEBUG
    symbol_input = symbol_entry.get().strip().upper()
    # DEBUG
    print(f"輸入的股票代號: {symbol_input}")
    # DEBUG
    if not symbol_input.endswith(".TW") and symbol_input.isdigit():
        symbol = symbol_input + ".TW"
    else:
        symbol = symbol_input

    start_ym = start_entry.get().strip()
    end_ym = end_entry.get().strip()
    # DEBUG
    print(f"輸入的起始年月: {start_ym}")
    print(f"輸入的終止年月: {end_ym}")
    # DEBUG

    if not re.match(r"^\d{4}-\d{2}$", start_ym) or not re.match(r"^\d{4}-\d{2}$", end_ym):
        raise ValueError("請輸入有效的年月格式 yyyy-mm")
    
    try:
        start_dt = datetime.strptime(start_ym, "%Y-%m")
        end_dt = datetime.strptime(end_ym, "%Y-%m")
    except ValueError:
        raise ValueError("年月格式錯誤，請輸入 yyyy-mm")

    if start_dt > end_dt:
        raise ValueError("起始年月不能大於終止年月")

    today = datetime.today()
    if start_dt > today or end_dt > today:
        raise ValueError("年月不可超過當前日期")

    start_date = f"{start_ym}-01"
    y, m = map(int, end_ym.split("-"))
    end_date = f"{end_ym}-{calendar.monthrange(y, m)[1]:02d}"

    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        raise ValueError(f"查無 {symbol} 的資料")

    # 判斷 Pandas DataFrame 的欄位 df.columns 是否是 MultiIndex（多層索引）
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(1):
            # DEBUG
            print(f"df: \n {df}")
            # DEBUG
            df = df.xs(symbol, axis=1, level=1)
            # DEBUG
            print(f"df.xs(symbol, axis=1, level=1): \n {df}")
            # DEBUG
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, axis=1, level=0)
        else:
            available = df.columns.get_level_values(1).unique().tolist()
            raise ValueError(f"未在下載結果中找到股票代號 {symbol}\n可用代號: {available}")

    # 將欄位名稱做以下處理: (1) 轉成字串型態 (2) 移除字串前後的空白字元 (3) 轉成「每個單字首字母大寫」的格式
    df.columns = df.columns.astype(str).str.strip().str.title()

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"資料中缺少以下欄位：{missing_cols}\n實際欄位為：{list(df.columns)}\n請確認股票代號與時間範圍是否正確")

    # 從 DataFrame df 中刪除在指定欄位（由 required_cols 定義）中，有缺失值（NaN）的所有列（rows）
    df = df.dropna(subset=required_cols)
    df = compute_indicators(df)
    return df, symbol

def on_data_ready(df, symbol):
    # DEBUG
    print("執行ftn call \"on_data_ready\"")
    # DEBUG
    fig = plot_chart(df, symbol)
    for widget in plot_frame.winfo_children():
        widget.destroy()

    btn_frame = tk.Frame(plot_frame)
    btn_frame.pack(pady=5)

    def save_chart():
        # 彈出一個「另存為檔案」對話框，讓使用者選擇存檔位置與類型
        # filedialog.asksaveasfilename: 讓使用者能選擇要把檔案儲存在哪個資料夾、命名為什麼檔案名，以及存成什麼檔案格式
        # defaultextension=".png": 如果使用者沒在檔名後寫副檔名，預設會自動加上 .png 副檔名
        # filetypes=[("PNG 圖片", "*.png"), ("PDF", "*.pdf")]: 設定檔案類型過濾器，下拉選單允許使用者選擇「只顯示 PNG 圖檔」或「只顯示 PDF」
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 圖片", "*.png"), ("PDF", "*.pdf")])
        if file_path:
            fig.savefig(file_path)
            messagebox.showinfo("匯出成功", f"已儲存為：{file_path}")
        else:
            messagebox.showinfo("取消匯出", "使用者取消了儲存動作，尚未匯出任何檔案。")

    export_btn = tk.Button(btn_frame, text="匯出圖表", command=save_chart, font=(font_tkinter, 12), bg="yellow")
    export_btn.pack()

    # 將 matplotlib 的 fig 包裝成 Tkinter 可用的 canvas
    # 此做法是 Tkinter 與 Matplotlib 整合的標準流程，廣泛用於需要在 GUI 內嵌入圖表的應用程序中
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    # draw() 是讓 Matplotlib 在 Tkinter 的畫布上完成圖形繪製動作
    canvas.draw()
    # pack() 則是 Tkinter 佈局管理的步驟，讓畫布元件出現於視窗中
    canvas.get_tk_widget().pack()

def threaded_fetch():
    def task():
        try:
            df, symbol = fetch_and_prepare_data()
            # 在 Tkinter 的事件循環空閒時，立刻呼叫 on_data_ready(df, symbol) 函式，而不是立即在目前程式碼中直接執行
            # root.after(delay, callback) 是 Tkinter 用來延遲執行某個函式或任務的方法
            root.after(0, lambda: on_data_ready(df, symbol))
        except Exception as e:
            root.after(0, lambda e=e: messagebox.showerror("錯誤", f"執行時發生錯誤：\n{str(e)}"))
        finally:
            root.after(0, lambda: loading_label.config(text=""))

    loading_label.config(text="資料下載中...")
    # 建立並啟動一個新的「執行緒」（Thread），讓函式 task 在背景中並行執行，而不會阻塞主程式執行流程
    # daemon=True，當主程式結束時，該執行緒會被強制終止，不會再繼續執行
    threading.Thread(target=task, daemon=True).start()

# 開始下載

def start_download():
    # DEBUG
    print("執行ftn call \"start_download\"")
    # DEBUG
    loading_label.config(text="")
    # 清空框架內容：每次需要重新顯示、繪製新的圖表或資訊前，先用這段程式把舊的元件全部移除，避免畫面重疊或殘留先前內容。
    # 動態更新介面：當程式需要讓一個區塊顯示不同內容時（例如重新繪圖、載入不同圖片或表格），可以先清除舊的內容，再加入新的元件
    for widget in plot_frame.winfo_children():
        widget.destroy()
    threaded_fetch()

def on_close():
    # DEBUG
    print("執行ftn call \"on_close\"")
    # DEBUG
    # Tkinter 中用來結束主事件循環（mainloop）的方法，但視窗仍可能存在
    root.quit()
    # Tkinter 中用來完全銷毀主視窗及其所有子元件，並釋放相關資源的方法
    root.destroy()

# GUI
# 用來建立主視窗（也就是最上層窗口）
root = tk.Tk()
#DEBUG
print("建立主視窗")
#DEBUG
root.title("Yahoo 股價圖表工具")
root.geometry("900x800")
# 用來設定全局預設字體和字體大小
root.option_add("*Font", (font_tkinter, 14)) 

# 建立一個 Frame 元件（框架容器）並且設置它為主視窗 root 的子元件
frame = tk.Frame(root) 
#DEBUG
print("建立一個 Frame 元件")
#DEBUG
# 將 frame 放置在視窗中，並在它的上下各加 10 像素的空白，避免元件和其他元件黏得太近，讓視覺更整齊
frame.pack(pady=10) 

# 介面上放一個顯示「股票代號 (例如2330.TW):」文字的標籤，位置在框架的第一行第一格，且文字靠右對齊
tk.Label(frame, text="股票代號 (例如2330.TW):").grid(row=0, column=0, sticky="e") 
#DEBUG
print("介面上放一個顯示「股票代號 (例如2330.TW):」文字的標籤")
#DEBUG
# 在 Tkinter GUI 中，建立一個單行輸入框 (Entry widget)，並且將這個輸入框設置在名為 frame 的容器內
# tk.Entry() 是 Tkinter 中用來讓使用者輸入單行文本的控件，也就是常見的文字輸入框。使用者可以在這個框裡鍵入字串，用來收集用戶輸入資料
symbol_entry = tk.Entry(frame) 
# 在 Tkinter 中對先前建立的輸入框 symbol_entry 使用 Grid 佈局管理器，將它放置在其父容器（例如之前提到的 frame）的第0行第1列位置
symbol_entry.grid(row=0, column=1) 

tk.Label(frame, text="起始年月 (yyyy-mm):").grid(row=1, column=0, sticky="e")
#DEBUG
print("介面上放一個顯示「起始年月 (yyyy-mm):」文字的標籤")
#DEBUG
start_entry = tk.Entry(frame)
start_entry.grid(row=1, column=1)

tk.Label(frame, text="終止年月 (yyyy-mm):").grid(row=2, column=0, sticky="e")
#DEBUG
print("介面上放一個顯示「終止年月 (yyyy-mm):」文字的標籤")
#DEBUG
end_entry = tk.Entry(frame)
end_entry.grid(row=2, column=1)

# 用 Tkinter 在 frame 容器裡建立一個按鈕（Button），並且立刻透過 .grid() 將它放置在指定的位置
# start_download: 按鈕被按下時要呼叫的函式
tk.Button(frame, text="下載並繪圖", command=start_download, bg="lightblue").grid(row=3, column=0, columnspan=2, pady=10)
#DEBUG
print("建立一個按鈕Button \"下載並繪圖\"")
#DEBUG

loading_label = tk.Label(root, text="", fg="blue")
loading_label.pack()

plot_frame = tk.Frame(root)
# fill="both": 這個 plot_frame 要同時在水平方向（x 軸）和垂直方向（y 軸）填滿其父容器剩餘的空間。
# 換句話說，plot_frame 會撐滿父容器的整個寬度和高度。
# expand=True: 當父容器的大小改變時，plot_frame 會擴展以佔據更多可用空間，而不僅僅是維持元件的原始大小。
# 這可以讓元件會隨著視窗或容器的大小動態調整尺寸。
plot_frame.pack(fill="both", expand=True)

# 在 Tkinter GUI 程式中，讓我們自訂當使用者點擊視窗右上角關閉（×）按鈕時的行為
# "WM_DELETE_WINDOW" 是系統視窗管理器傳來的「視窗關閉事件」信號，加上 "on_close"，代表改用自訂的函式 on_close 來處理
root.protocol("WM_DELETE_WINDOW", on_close)
# 執行時，程式會進入一個 事件循環（event loop），也就是不停等待並處理使用者和系統產生的事件（像是按鈕點擊、鍵盤輸入、視窗調整大小等）。
# 這個函式會阻塞程式的後續執行，直到視窗被關閉（例如呼叫 root.destroy() 或使用者按下視窗的關閉按鈕）為止
root.mainloop()
