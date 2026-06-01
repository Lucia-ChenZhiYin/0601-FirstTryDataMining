import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 資料探勘與機器學習套件
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori, association_rules

# 讓圖表可以正常顯示中文
plt.rcParams['font.family'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 功能一：資料載入與多變數初步觀測
# ==========================================
def load_and_explore_data(file_path):
    print("--- [步驟 1] 資料載入與多變數觀測 ---")
    df = pd.read_excel(file_path)
    print(f"資料筆數: {df.shape[0]} 筆 | 變數數量: {df.shape[1]} 個\n")
    print("【欄位型態資訊】")
    print(df.info())
    print("\n【敘述性統計摘要】")
    print(df.describe(include='all'))
    
    # 自動繪製數值變數的相關性熱圖
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("多變數相關性熱圖 (Correlation Heatmap)")
        plt.savefig("correlation_heatmap.png")
        print("\n[提示] 已自動生成多變數相關性熱圖：correlation_heatmap.png")
    
    return df

# ==========================================
# 功能二：關聯規則分析 (Association Rules)
# ==========================================
def run_association_rules(df, min_support=0.1, min_confidence=0.7):
    print("\n--- [核心模型] 關聯規則分析 (Apriori) ---")
    # 關聯規則需要將資料轉換為 0/1 矩陣
    basket_df = df.select_dtypes(include=[np.number, bool]).copy()
    basket_df = basket_df.applymap(lambda x: 1 if x > 0 else 0)
    
    # 尋找頻繁項集
    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("【警告】找不到符合該支持度門檻的頻繁項集，請嘗試調低 min_support。")
        return
        
    # 計算關聯規則
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    print(f"依據參數 (最小支持度={min_support}, 最小信心度={min_confidence}) 篩選出的前 5 條規則：")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    rules.to_csv("association_rules_results.csv", index=False)

# ==========================================
# 功能三：分類分析 (Classification)
# ==========================================
def run_classification(df, target_column, feature_columns, n_estimators=100, max_depth=None):
    print("\n--- [核心模型] 分類分析 (隨機森林) ---")
    X = df[feature_columns]
    y = df[target_column]
    
    # 劃分 80% 訓練集、20% 測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 建立模型與設定參數
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # 輸出評估報告
    y_pred = clf.predict(X_test)
    print(f"模型參數：樹木數量={n_estimators}, 最大深度={max_depth}")
    print("\n【分類模型評估報告】")
    print(classification_report(y_test, y_pred))
    
    # 特徵重要性（讓老師看到多變數的觀測影響力）
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("【多變數特徵重要性排序】")
    for f in range(X.shape[1]):
        print(f"{f + 1}. 變數 '{X.columns[indices[f]]}' 貢獻度: {importances[indices[f]]:.4f}")

# ==========================================
# 功能四：群集分析 (Clustering)
# ==========================================
def run_clustering(df, feature_columns, n_clusters=3):
    print("\n--- [核心模型] 群集分析 (K-Means) ---")
    X = df[feature_columns].select_dtypes(include=[np.number])
    
    # 建立 K-Means 模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    print(f"模型參數：設定分群數 K={n_clusters}")
    
    # 分析各群特徵均值（客戶畫像/資料特徵觀測）
    df_analyzed = X.copy()
    df_analyzed['Cluster_Label'] = labels
    print("\n【各群集的多變數特徵均值觀測】")
    print(df_analyzed.groupby('Cluster_Label').mean())
    
    # 視覺化（若選取兩個以上的變數，畫出前兩個變數的分群散佈圖）
    if len(feature_columns) >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(f"K-Means 群集結果視覺化 (K={n_clusters})")
        plt.xlabel(feature_columns[0])
        plt.ylabel(feature_columns[1])
        plt.savefig("clustering_result.png")
        print("\n[提示] 已自動生成群集視覺化圖表：clustering_result.png")

# ==========================================
# 主程式執行入口
# ==========================================
if __name__ == "__main__":
    # 【請確保你的資料夾內有這個 Excel 檔，或把下方改成你的真實檔名】
    file_path = "data.xlsx" 
    
    try:
        # 1. 載入並觀測多變數
        dataset = load_and_explore_data(file_path)
        
        # 2. 提供互動選單供展示
        print("\n==================================")
        print(" 選擇你想建立的資料探勘模型：")
        print(" 1: 關聯規則分析 (Apriori)")
        print(" 2: 分類分析 (Random Forest)")
        print(" 3: 群集分析 (K-Means)")
        print("==================================")
        mode = input("請輸入數字 (1/2/3): ")
        
        if mode == "1":
            supp = float(input("請設定最小支持度 min_support (例如 0.05): ") or 0.05)
            conf = float(input("請設定最小信心度 min_confidence (例如 0.6): ") or 0.6)
            run_association_rules(dataset, min_support=supp, min_confidence=conf)
            
        elif mode == "2":
            print("\n可用欄位名稱：", list(dataset.columns))
            target = input("請輸入『目標標籤欄位 (Y)』: ")
            features = input("請輸入『特徵變數欄位 (X)』(多個請用英文逗號隔開): ").split(',')
            features = [f.strip() for f in features]
            
            run_classification(dataset, target_column=target, feature_columns=features, n_estimators=100, max_depth=5)
            
        elif mode == "3":
            print("\n可用欄位名稱：", list(dataset.columns))
            features = input("請輸入要納入分群的『數值變數』(多個請用英文逗號隔開): ").split(',')
            features = [f.strip() for f in features]
            k_num = int(input("請設定群集數量 K (例如 3): ") or 3)
            
            run_clustering(dataset, feature_columns=features, n_clusters=k_num)
        else:
            print("輸入無效，程式結束。")
            
    except FileNotFoundError:
        print(f"【錯誤】在當前資料夾找不到 '{file_path}' 檔案，請確認檔名並確認檔案已放入專案目錄中。")