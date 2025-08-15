#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試版AI News Collector - 用於本地測試
"""

import os
import csv
import json
import hashlib
import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# 設定logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Article:
    """文章資料結構"""
    title: str
    url: str
    summary: str
    content: str
    published_date: str
    source: str
    category: str
    relevance_score: float
    sentiment_score: float
    hash_id: str
    ai_summary: str = ""

class TestDataManager:
    """測試用資料管理器"""
    
    def __init__(self, csv_file: str = "test_ai_news_data.csv"):
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
            logger.info(f"建立新的測試資料檔案: {self.csv_file}")
    
    def article_exists(self, hash_id: str) -> bool:
        """檢查文章是否已存在"""
        try:
            df = pd.read_csv(self.csv_file)
            return hash_id in df['hash_id'].values
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return False
    
    def save_article(self, article: Article):
        """儲存文章"""
        if self.article_exists(article.hash_id):
            logger.info(f"文章已存在: {article.title[:50]}...")
            return
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'hash_id': article.hash_id,
                'title': article.title,
                'url': article.url,
                'summary': article.summary,
                'content': article.content,
                'published_date': article.published_date,
                'source': article.source,
                'category': article.category,
                'relevance_score': article.relevance_score,
                'sentiment_score': article.sentiment_score,
                'ai_summary': article.ai_summary,
                'collected_date': datetime.now().isoformat()
            })
        logger.info(f"✅ 儲存新文章: {article.title[:50]}...")

class TestRSSCollector:
    """測試用RSS收集器 - 只取少量文章"""
    
    # 精選測試用RSS來源
# 在 test_collector.py 中，替換 TestRSSCollector 的 RSS_FEEDS
    RSS_FEEDS = {
    # 主要科技媒體
    'MIT Technology Review': 'https://www.technologyreview.com/feed/',
    'Nature AI': 'https://www.nature.com/subjects/machine-learning.rss',
    'IEEE Spectrum': 'https://spectrum.ieee.org/rss/fulltext',
    'TechCrunch': 'https://feeds.feedburner.com/TechCrunch/',    
    # AI公司官方部落格
    'OpenAI Blog': 'https://openai.com/blog/rss.xml',
    'Google AI Research': 'https://research.google/blog/rss/',
    'Meta AI Blog': 'https://research.facebook.com/feed/',
    'Anthropic Blog': 'https://www.anthropic.com/feed.xml',
    'Amazon AI Blog': 'https://aws.amazon.com/blogs/machine-learning/feed/',
    'Microsoft AI Blog': 'https://blogs.microsoft.com/ai/feed/',
    
    # 學術期刊
    'arXiv AI': 'http://export.arxiv.org/rss/cs.AI',
    'arXiv ML': 'http://export.arxiv.org/rss/cs.LG',
    'arXiv Bio': 'http://export.arxiv.org/rss/q-bio',
    
    # 科技新聞
    'TechCrunch AI': 'https://techcrunch.com/category/artificial-intelligence/feed/',
    'The Verge AI': 'https://www.theverge.com/rss/ai-artificial-intelligence/index.xml',
    'Wired AI': 'https://www.wired.com/feed/tag/ai/latest/rss',
    'VentureBeat AI': 'https://venturebeat.com/category/ai/feed/',
    
    # 生物科技
    'Nature Biotechnology': 'https://www.nature.com/nbt.rss',
    'BioPharma Dive': 'https://www.biopharmadive.com/feeds/news/',
    'GenomeWeb': 'https://www.genomeweb.com/section/rss/news?access_control=46',
    
    # 額外來源
    'AI Business': 'https://aibusiness.com/rss.xml',
    'AI News': 'https://www.artificialintelligence-news.com/feed/rss/',
    'Towards Data Science': 'https://towardsdatascience.com/feed',
    'Machine Learning Mastery': 'https://machinelearningmastery.com/blog/feed/',
    'BAIR Blog': 'https://bair.berkeley.edu/blog/feed.xml',
    'NVIDIA AI Blog': 'https://feeds.feedburner.com/nvidiablog',
}
    
    def collect_rss_articles(self, max_articles_per_feed: int = 3) -> List[Article]:
        """收集RSS文章 - 測試版本"""
        articles = []
        
        for source_name, feed_url in self.RSS_FEEDS.items():
            try:
                logger.info(f"📡 正在測試收集 {source_name}...")
                feed = feedparser.parse(feed_url)
                
                if not feed.entries:
                    logger.warning(f"⚠️  {source_name} 沒有文章或無法存取")
                    continue
                
                for entry in feed.entries[:max_articles_per_feed]:
                    try:
                        title = entry.get('title', '').strip()
                        url = entry.get('link', '')
                        summary = entry.get('summary', entry.get('description', ''))
                        
                        if not title or not url:
                            continue
                        
                        # 處理發佈日期
                        published_date = self._parse_date(entry)
                        
                        # 生成唯一ID
                        hash_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
                        
                        # 簡單的AI相關過濾
                        if self._is_ai_related(title, summary):
                            article = Article(
                                title=title,
                                url=url,
                                summary=self._clean_text(summary),
                                content="",
                                published_date=published_date,
                                source=source_name,
                                category="AI",
                                relevance_score=0.0,
                                sentiment_score=0.0,
                                hash_id=hash_id
                            )
                            articles.append(article)
                            logger.info(f"   📄 找到AI相關文章: {title[:50]}...")
                    
                    except Exception as e:
                        logger.error(f"處理 {source_name} 文章時發生錯誤: {e}")
                        continue
                
                # 避免請求過於頻繁
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"收集 {source_name} RSS時發生錯誤: {e}")
                continue
        
        logger.info(f"🎯 總共收集到 {len(articles)} 篇AI相關文章")
        return articles
    
    def _parse_date(self, entry) -> str:
        """解析發佈日期"""
        date_fields = ['published', 'updated', 'pubDate']
        for field in date_fields:
            if hasattr(entry, field + '_parsed'):
                try:
                    date_tuple = getattr(entry, f"{field}_parsed", None)
                    if date_tuple:
                        return datetime(*date_tuple[:6]).isoformat()
                except:
                    continue
        
        return datetime.now().isoformat()
    
    def _is_ai_related(self, title: str, summary: str) -> bool:
        """判斷是否為AI相關內容"""
        ai_keywords = [
            'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
            'neural network', 'chatgpt', 'gpt', 'llm', 'large language model',
            'computer vision', 'natural language', 'nlp', 'robotics', 'automation',
            'biotech', 'biotechnology', 'bioinformatics', 'computational biology'
        ]
        
        text = f"{title} {summary}".lower()
        return any(keyword in text for keyword in ai_keywords)
    
    def _clean_text(self, text: str) -> str:
        """清理文字內容"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class TestClaudeProcessor:
    """測試用Claude處理器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def classify_and_summarize(self, article: Article) -> Article:
        """測試Claude分類和摘要功能"""
        try:
            logger.info(f"🤖 正在用Claude處理: {article.title[:30]}...")
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            prompt = f"""
            請分析以下AI相關文章並提供JSON格式回答：
            
            文章標題: {article.title}
            文章摘要: {article.summary[:500]}
            
            請回答：
            1. 內容分類 (AI Research, Industry News, Biotech, Robotics, Other)
            2. 相關性評分 (0-1, 1為最相關AI主題)
            3. 情感分析評分 (-1到1, -1負面, 0中性, 1正面)
            4. 繁體中文摘要 (80字以內)
            
            回答格式：
            {{
                "category": "分類",
                "relevance_score": 0.8,
                "sentiment_score": 0.2,
                "ai_summary": "繁體中文摘要"
            }}
            """
            
            data = {
                "model": "claude-3-haiku-20240307",  # 使用較便宜的模型測試
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text']
                
                try:
                    # 解析JSON回應
                    parsed_result = json.loads(content)
                    article.category = parsed_result.get('category', 'Other')
                    article.relevance_score = float(parsed_result.get('relevance_score', 0.5))
                    article.sentiment_score = float(parsed_result.get('sentiment_score', 0.0))
                    article.ai_summary = parsed_result.get('ai_summary', article.summary)
                    
                    logger.info(f"   ✅ Claude處理完成 - 分類: {article.category}, 相關性: {article.relevance_score:.2f}")
                    
                except json.JSONDecodeError:
                    logger.error(f"   ❌ 無法解析Claude回應: {content}")
                    # 使用預設值
                    article.category = "Other"
                    article.relevance_score = 0.5
                    article.sentiment_score = 0.0
                    article.ai_summary = article.summary
            else:
                logger.error(f"   ❌ Claude API錯誤: {response.status_code} - {response.text}")
                # 使用預設值
                article.category = "Other"
                article.relevance_score = 0.5
                article.sentiment_score = 0.0
                article.ai_summary = article.summary
            
            # 避免API rate limit
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Claude處理時發生錯誤: {e}")
            # 使用預設值
            article.category = "Other"
            article.relevance_score = 0.5
            article.sentiment_score = 0.0
            article.ai_summary = article.summary
        
        return article

def test_rss_only():
    """只測試RSS收集功能"""
    print("🧪 測試RSS收集功能...")
    
    collector = TestRSSCollector()
    articles = collector.collect_rss_articles(max_articles_per_feed=2)
    
    print(f"\n📊 收集結果:")
    print(f"總文章數: {len(articles)}")
    
    for i, article in enumerate(articles[:5], 1):
        print(f"\n{i}. {article.title}")
        print(f"   來源: {article.source}")
        print(f"   摘要: {article.summary[:100]}...")
        print(f"   連結: {article.url}")

def test_claude_api():
    """測試Claude API"""
    api_key = os.environ.get('CLAUDE_API_KEY')
    if not api_key:
        print("❌ 請設定 CLAUDE_API_KEY 環境變數")
        return
    
    print("🧪 測試Claude API...")
    
    processor = TestClaudeProcessor(api_key)
    
    # 建立測試文章
    test_article = Article(
        title="OpenAI Launches GPT-4 Turbo with Enhanced Capabilities",
        url="https://example.com/test",
        summary="OpenAI announced the release of GPT-4 Turbo, featuring improved reasoning capabilities and reduced costs for developers.",
        content="",
        published_date=datetime.now().isoformat(),
        source="Test Source",
        category="",
        relevance_score=0.0,
        sentiment_score=0.0,
        hash_id="test123"
    )
    
    processed_article = processor.classify_and_summarize(test_article)
    
    print(f"\n📊 Claude處理結果:")
    print(f"標題: {processed_article.title}")
    print(f"分類: {processed_article.category}")
    print(f"相關性: {processed_article.relevance_score}")
    print(f"情感分析: {processed_article.sentiment_score}")
    print(f"AI摘要: {processed_article.ai_summary}")

def test_full_pipeline():
    """測試完整流程"""
    print("🧪 測試完整AI新聞收集流程...")
    
    # 檢查環境變數
    claude_api_key = os.environ.get('CLAUDE_API_KEY')
    if not claude_api_key:
        print("❌ 請設定 CLAUDE_API_KEY 環境變數")
        return
    
    # 初始化組件
    data_manager = TestDataManager()
    rss_collector = TestRSSCollector()
    claude_processor = TestClaudeProcessor(claude_api_key)
    
    # 收集文章
    print("\n📡 步驟1: 收集RSS文章...")
    articles = rss_collector.collect_rss_articles(max_articles_per_feed=2)
    
    # 處理文章
    print(f"\n🤖 步驟2: 用Claude處理 {len(articles)} 篇文章...")
    new_articles = []
    
    for i, article in enumerate(articles, 1):
        if not data_manager.article_exists(article.hash_id):
            print(f"   處理 {i}/{len(articles)}: {article.title[:40]}...")
            processed_article = claude_processor.classify_and_summarize(article)
            data_manager.save_article(processed_article)
            new_articles.append(processed_article)
        else:
            print(f"   跳過已存在文章: {article.title[:40]}...")
    
    print(f"\n✅ 完成！處理了 {len(new_articles)} 篇新文章")
    
    # 顯示結果
    if new_articles:
        print(f"\n📊 處理結果摘要:")
        categories = {}
        total_relevance = 0
        
        for article in new_articles:
            categories[article.category] = categories.get(article.category, 0) + 1
            total_relevance += article.relevance_score
        
        print(f"分類統計: {categories}")
        print(f"平均相關性: {total_relevance/len(new_articles):.2f}")
        
        print(f"\n📰 最相關的文章:")
        sorted_articles = sorted(new_articles, key=lambda x: x.relevance_score, reverse=True)
        for article in sorted_articles[:3]:
            print(f"• {article.title[:60]}...")
            print(f"  相關性: {article.relevance_score:.2f} | {article.ai_summary[:80]}...")

def main():
    print("🧪 AI News Collector 測試工具")
    print("=" * 50)
    
    while True:
        print("\n選擇測試項目:")
        print("1. 只測試RSS收集")
        print("2. 只測試Claude API")
        print("3. 測試完整流程")
        print("4. 查看測試資料")
        print("5. 退出")
        
        choice = input("\n請選擇 (1-5): ").strip()
        
        if choice == '1':
            test_rss_only()
        elif choice == '2':
            test_claude_api()
        elif choice == '3':
            test_full_pipeline()
        elif choice == '4':
            os.system("python manual_trigger.py stats")
        elif choice == '5':
            print("👋 測試結束")
            break
        else:
            print("❌ 無效選擇，請重新輸入")

if __name__ == "__main__":
    main()