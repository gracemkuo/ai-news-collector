#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版手動執行腳本 - 解決import問題
"""

import os
import sys
import csv
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import logging

# 設定logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataManager:
    """簡化版資料管理器"""
    
    def __init__(self, csv_file: str = "ai_news_data.csv"):
        self.csv_file = csv_file
        self.fieldnames = [
            'hash_id', 'title', 'url', 'summary', 'content', 
            'published_date', 'source', 'category', 'relevance_score',
            'sentiment_score', 'ai_summary', 'collected_date'
        ]
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """確保CSV檔案存在"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            print(f"✅ 建立了新的資料檔案: {self.csv_file}")
    
    def get_statistics(self) -> Dict:
        """取得資料統計"""
        try:
            if not os.path.exists(self.csv_file):
                return {"error": "資料檔案不存在"}
            
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return {"error": "檔案中無資料"}
            
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
            df['collected_date'] = pd.to_datetime(df['collected_date'], errors='coerce')
            
            # 移除日期解析失敗的行
            df = df.dropna(subset=['published_date'])
            
            if df.empty:
                return {"error": "沒有有效的日期資料"}
            
            stats = {
                "總文章數": len(df),
                "資料收集期間": {
                    "最早": df['published_date'].min().strftime('%Y-%m-%d'),
                    "最新": df['published_date'].max().strftime('%Y-%m-%d')
                },
                "來源統計": df['source'].value_counts().head(10).to_dict(),
                "分類統計": df['category'].value_counts().to_dict(),
                "平均相關性分數": {
                    category: round(group['relevance_score'].mean(), 3)
                    for category, group in df.groupby('category') if len(group) > 0
                },
                "情感分析": {
                    "正面文章": len(df[df['sentiment_score'] > 0.3]),
                    "中性文章": len(df[(df['sentiment_score'] >= -0.3) & (df['sentiment_score'] <= 0.3)]),
                    "負面文章": len(df[df['sentiment_score'] < -0.3])
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"統計資料時發生錯誤: {e}")
            return {"error": str(e)}
    
    def search_articles(self, 
                       keyword: str = None,
                       source: str = None,
                       category: str = None,
                       date_from: str = None,
                       date_to: str = None,
                       min_relevance: float = None,
                       limit: int = 100) -> List[Dict]:
        """搜尋文章"""
        try:
            if not os.path.exists(self.csv_file):
                return []
            
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return []
            
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
            df = df.dropna(subset=['published_date'])
            
            # 關鍵字搜尋
            if keyword:
                mask = (df['title'].str.contains(keyword, case=False, na=False) |
                       df['summary'].str.contains(keyword, case=False, na=False) |
                       df['ai_summary'].str.contains(keyword, case=False, na=False))
                df = df[mask]
            
            # 來源過濾
            if source:
                df = df[df['source'].str.contains(source, case=False, na=False)]
            
            # 分類過濾
            if category:
                df = df[df['category'] == category]
            
            # 日期範圍過濾
            if date_from:
                df = df[df['published_date'] >= pd.to_datetime(date_from)]
            if date_to:
                df = df[df['published_date'] <= pd.to_datetime(date_to)]
            
            # 相關性過濾
            if min_relevance is not None:
                df = df[df['relevance_score'] >= min_relevance]
            
            # 按相關性排序並限制數量
            df = df.sort_values('relevance_score', ascending=False).head(limit)
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"搜尋文章時發生錯誤: {e}")
            return []

def show_statistics():
    """顯示統計資訊"""
    data_manager = SimpleDataManager()
    
    print("📊 AI新聞資料統計分析")
    print("=" * 50)
    
    stats = data_manager.get_statistics()
    
    if "error" in stats:
        print(f"❌ {stats['error']}")
        print("\n💡 提示:")
        print("1. 請先執行收集程序來產生資料")
        print("2. 或者檢查 ai_news_data.csv 檔案是否存在")
        return
    
    print(f"📈 總文章數: {stats['總文章數']}")
    print(f"📅 資料期間: {stats['資料收集期間']['最早']} ~ {stats['資料收集期間']['最新']}")
    
    print(f"\n📰 來源統計 (前10名):")
    for source, count in stats['來源統計'].items():
        print(f"   • {source}: {count} 篇")
    
    print(f"\n🏷️  分類統計:")
    for category, count in stats['分類統計'].items():
        avg_score = stats['平均相關性分數'].get(category, 0)
        print(f"   • {category}: {count} 篇 (平均相關性: {avg_score:.3f})")
    
    print(f"\n😊 情感分析:")
    emotion = stats['情感分析']
    total = emotion['正面文章'] + emotion['中性文章'] + emotion['負面文章']
    if total > 0:
        print(f"   • 正面: {emotion['正面文章']} 篇 ({emotion['正面文章']/total*100:.1f}%)")
        print(f"   • 中性: {emotion['中性文章']} 篇 ({emotion['中性文章']/total*100:.1f}%)")
        print(f"   • 負面: {emotion['負面文章']} 篇 ({emotion['負面文章']/total*100:.1f}%)")

def search_articles_cmd(args):
    """搜尋文章命令"""
    data_manager = SimpleDataManager()
    
    print(f"🔍 搜尋條件:")
    if args.keyword:
        print(f"   關鍵字: {args.keyword}")
    if args.source:
        print(f"   來源: {args.source}")
    if args.category:
        print(f"   分類: {args.category}")
    if args.date_from:
        print(f"   開始日期: {args.date_from}")
    if args.date_to:
        print(f"   結束日期: {args.date_to}")
    if args.min_relevance:
        print(f"   最低相關性: {args.min_relevance}")
    
    articles = data_manager.search_articles(
        keyword=args.keyword,
        source=args.source,
        category=args.category,
        date_from=args.date_from,
        date_to=args.date_to,
        min_relevance=args.min_relevance,
        limit=args.limit
    )
    
    print(f"\n📄 找到 {len(articles)} 篇文章:")
    print("-" * 80)
    
    if not articles:
        print("沒有找到符合條件的文章")
        return
    
    for i, article in enumerate(articles, 1):
        print(f"{i}. 【{article.get('category', 'Unknown')}】{article.get('title', 'No Title')}")
        print(f"   來源: {article.get('source', 'Unknown')} | 相關性: {article.get('relevance_score', 0):.3f} | 情感: {article.get('sentiment_score', 0):.3f}")
        print(f"   連結: {article.get('url', 'No URL')}")
        
        summary = article.get('ai_summary', article.get('summary', 'No Summary'))
        if len(summary) > 100:
            summary = summary[:100] + "..."
        print(f"   摘要: {summary}")
        print()

def create_sample_data():
    """建立範例資料"""
    data_manager = SimpleDataManager()
    
    print("📝 建立範例資料...")
    
    sample_articles = [
        {
            'hash_id': 'sample1',
            'title': 'OpenAI發布ChatGPT-5：突破性AI模型的新里程碑',
            'url': 'https://example.com/chatgpt5',
            'summary': 'OpenAI今日宣布推出ChatGPT-5，這是該公司迄今為止最先進的語言模型...',
            'content': '',
            'published_date': datetime.now().isoformat(),
            'source': 'OpenAI Blog',
            'category': 'AI Research',
            'relevance_score': 0.95,
            'sentiment_score': 0.8,
            'ai_summary': 'OpenAI發布了新一代ChatGPT-5模型，在推理能力和多模態處理上有重大突破。',
            'collected_date': datetime.now().isoformat()
        },
        {
            'hash_id': 'sample2',
            'title': 'AI輔助藥物發現：DeepMind在蛋白質結構預測上的新進展',
            'url': 'https://example.com/protein-folding',
            'summary': 'DeepMind的AlphaFold技術在藥物發現領域取得新突破...',
            'content': '',
            'published_date': (datetime.now() - timedelta(days=1)).isoformat(),
            'source': 'DeepMind Blog',
            'category': 'Biotech',
            'relevance_score': 0.88,
            'sentiment_score': 0.6,
            'ai_summary': 'DeepMind利用AI技術在蛋白質結構預測上取得突破，有助於加速新藥開發。',
            'collected_date': datetime.now().isoformat()
        },
        {
            'hash_id': 'sample3',
            'title': 'Tesla機器人Optimus開始商業化部署',
            'url': 'https://example.com/tesla-robot',
            'summary': 'Tesla宣布其人形機器人Optimus將開始在工廠環境中進行商業化測試...',
            'content': '',
            'published_date': (datetime.now() - timedelta(days=2)).isoformat(),
            'source': 'TechCrunch AI',
            'category': 'Robotics',
            'relevance_score': 0.82,
            'sentiment_score': 0.4,
            'ai_summary': 'Tesla的人形機器人Optimus開始商業化測試，標誌著機器人技術的重要進展。',
            'collected_date': datetime.now().isoformat()
        }
    ]
    
    # 寫入CSV檔案
    with open(data_manager.csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data_manager.fieldnames)
        for article in sample_articles:
            writer.writerow(article)
    
    print(f"✅ 已建立 {len(sample_articles)} 筆範例資料")

def main():
    parser = argparse.ArgumentParser(description='AI新聞收集器 - 資料分析工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 統計命令
    stats_parser = subparsers.add_parser('stats', help='顯示統計資訊')
    
    # 搜尋命令
    search_parser = subparsers.add_parser('search', help='搜尋文章')
    search_parser.add_argument('--keyword', help='關鍵字')
    search_parser.add_argument('--source', help='來源')
    search_parser.add_argument('--category', help='分類')
    search_parser.add_argument('--date-from', help='開始日期 (YYYY-MM-DD)')
    search_parser.add_argument('--date-to', help='結束日期 (YYYY-MM-DD)')
    search_parser.add_argument('--min-relevance', type=float, help='最低相關性分數')
    search_parser.add_argument('--limit', type=int, default=20, help='最大結果數')
    
    # 建立範例資料命令
    sample_parser = subparsers.add_parser('sample', help='建立範例資料')
    
    args = parser.parse_args()
    
    if args.command == 'stats':
        show_statistics()
    elif args.command == 'search':
        search_articles_cmd(args)
    elif args.command == 'sample':
        create_sample_data()
    else:
        parser.print_help()
        print("\n💡 使用範例:")
        print("python manual_trigger.py sample    # 建立範例資料")
        print("python manual_trigger.py stats     # 查看統計")
        print("python manual_trigger.py search --keyword ChatGPT  # 搜尋文章")

if __name__ == "__main__":
    main()