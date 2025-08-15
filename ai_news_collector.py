#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版AI News Collector - 長摘要版本
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
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

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

class DataManager:
    """資料管理器 - 使用CSV儲存"""
    
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
    
    def get_articles_by_date(self, date: str) -> List[Dict]:
        """根據日期取得文章"""
        try:
            df = pd.read_csv(self.csv_file)
            # 👇 改善日期解析
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce', format='mixed')
            target_date = pd.to_datetime(date)
            
            # 移除無法解析的日期
            df = df.dropna(subset=['published_date'])
            
            filtered_df = df[df['published_date'].dt.date == target_date.date()]
            return filtered_df.to_dict('records')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return []
        except Exception as e:
            logger.error(f"取得文章時發生錯誤: {e}")
            return []

class RSSCollector:
    """RSS Feed收集器"""
    
    # 完整RSS來源清單 - 你指定的版本
    RSS_FEEDS = {
        # 主要科技媒體
        'MIT Technology Review': 'https://www.technologyreview.com/feed/',
        'Nature AI': 'https://www.nature.com/subjects/machine-learning.rss',
        'IEEE Spectrum': 'https://spectrum.ieee.org/rss/fulltext',
        
        # AI公司官方部落格
        'OpenAI Blog': 'https://openai.com/blog/rss.xml',
        'Google AI Research': 'https://research.google/blog/rss/',
        'Anthropic Blog': 'https://www.anthropic.com/feed.xml',
        'Amazon AI Blog': 'https://aws.amazon.com/blogs/machine-learning/feed/',
        
        # 學術期刊
        'arXiv AI': 'http://export.arxiv.org/rss/cs.AI',
        'arXiv ML': 'http://export.arxiv.org/rss/cs.LG',
        'arXiv Bio': 'http://export.arxiv.org/rss/q-bio',
        
        # 科技新聞
        'TechCrunch AI': 'https://techcrunch.com/category/artificial-intelligence/feed/',
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
        'NVIDIA AI Blog': 'https://feeds.feedburner.com/nvidiablog',
    }
    
    def collect_rss_articles(self, max_articles_per_feed: int = 10, max_total_articles: int = 100) -> List[Article]:
        """收集RSS文章"""
        articles = []
        
        for source_name, feed_url in self.RSS_FEEDS.items():
            if len(articles) >= max_total_articles:
                logger.info(f"已達到最大文章數限制 ({max_total_articles})，停止收集")
                break
            try:
                logger.info(f"正在收集 {source_name} 的RSS...")
                feed = feedparser.parse(feed_url)
                source_articles = 0
                for entry in feed.entries[:max_articles_per_feed]:
                    if source_articles >= max_articles_per_feed or len(articles) >= max_total_articles:
                        break
                    try:
                        # 提取文章資訊
                        title = entry.get('title', '').strip()
                        url = entry.get('link', '')
                        summary = entry.get('summary', entry.get('description', ''))
                        
                        # 處理發佈日期
                        published_date = self._parse_date(entry)
                        
                        # 生成唯一ID
                        hash_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
                        
                        # 過濾AI相關內容
                        # 更嚴格的AI相關內容過濾
                        if self._is_ai_related(title, summary):
                            published_date = self._parse_date(entry)
                            hash_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
                            
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
                            source_articles += 1
                    
                    except Exception as e:
                        logger.error(f"處理文章時發生錯誤: {e}")
                        continue
                
                # 避免請求過於頻繁
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"收集 {source_name} RSS時發生錯誤: {e}")
                continue
        
        logger.info(f"共收集到 {len(articles)} 篇文章")
        return articles
    
    def _parse_date(self, entry) -> str:
        """解析發佈日期"""
        date_fields = ['published', 'updated', 'pubDate']
        for field in date_fields:
            if hasattr(entry, field):
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
            'biotech', 'biotechnology', 'bioinformatics', 'computational biology',
            'drug discovery', 'protein folding', 'genomics', 'crispr','DNA','RNA','protein'
        ]
        
        text = f"{title} {summary}".lower()
        return any(keyword in text for keyword in ai_keywords)
    
    def _clean_text(self, text: str) -> str:
        """清理文字內容"""
        # 移除HTML標籤
        text = re.sub(r'<[^>]+>', '', text)
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1000]

class ClaudeProcessor:
    """Claude API處理器 - 增強版長摘要"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.max_retries = 3
        self.base_delay = 2
    
    def classify_and_summarize(self, article: Article) -> Article:
        """使用Claude進行分類和摘要 - 生成更長的摘要"""
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                }
                
                    # 簡化prompt以提高成功率
                prompt = f"""
                    請分析以下文章並提供JSON格式回答：

                    標題: {article.title[:200]}
                    摘要: {article.summary[:300]}
                    來源: {article.source}

                    請提供JSON格式：
                    {{
                        "category": "AI Research/Industry News/Biotech/Other中選一",
                        "relevance_score": 0.85,
                        "sentiment_score": 0.0,
                        "ai_summary": "100-150字繁體中文摘要，包含主要內容和技術意義"
                    }}
                    """
                
                data = {
                    "model": "claude-3-5-sonnet-20241022",  # 使用更強的模型生成更好的摘要
                    "max_tokens": 500,  # 增加token數量支援更長摘要
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"role": "user", "content": prompt}]
                        }
                    ]
                }
                
                response = requests.post(self.base_url, headers=headers, json=data,timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['content'][0]['text']
                    
                    # 解析JSON
                    try:
                            # 嘗試從內容中提取JSON
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                parsed_result = json.loads(json_match.group())
                                
                                article.category = parsed_result.get('category', 'Other')
                                article.relevance_score = float(parsed_result.get('relevance_score', 0.5))
                                article.sentiment_score = float(parsed_result.get('sentiment_score', 0.0))
                                article.ai_summary = parsed_result.get('ai_summary', article.summary)
                                
                                logger.info(f"✅ Claude處理完成: {article.title[:30]}...")
                                return article
                            else:
                                raise json.JSONDecodeError("No JSON found", content, 0)
                                
                    except json.JSONDecodeError as e:
                            logger.warning(f"JSON解析失敗 (嘗試 {attempt + 1}/{self.max_retries}): {e}")
                            if attempt == self.max_retries - 1:
                                # 最後一次嘗試失敗，使用預設值
                                article.category = "Other"
                                article.relevance_score = 0.5
                                article.sentiment_score = 0.0
                                article.ai_summary = article.summary[:150]
                                return article
                    
                elif response.status_code == 529:
                        wait_time = self.base_delay * (2 ** attempt)
                        logger.warning(f"API繁忙 (529)，等待 {wait_time} 秒後重試...")
                        time.sleep(wait_time)
                        continue
                else:
                        logger.error(f"Claude API錯誤: {response.status_code}")
                        break
                        
            except requests.exceptions.Timeout:
                    logger.warning(f"請求超時 (嘗試 {attempt + 1}/{self.max_retries})")
                    time.sleep(self.base_delay)
                    continue
            except Exception as e:
                    logger.error(f"Claude處理錯誤: {e}")
                    break
        
        # 所有重試都失敗，使用預設值
        article.category = "Other"
        article.relevance_score = 0.5
        article.sentiment_score = 0.0
        article.ai_summary = article.summary[:150] if article.summary else "無法生成摘要"
        return article
    
    # def _extract_summary_from_text(self, text: str) -> Optional[str]:
    #     """從Claude回應文字中提取摘要"""
    #     try:
    #         # 尋找可能的摘要內容
    #         summary_patterns = [
    #             r'"ai_summary":\s*"([^"]+)"',
    #             r'摘要[：:]\s*(.+?)(?:\n|$)',
    #             r'總結[：:]\s*(.+?)(?:\n|$)'
    #         ]
            
    #         for pattern in summary_patterns:
    #             match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    #             if match:
    #                 return match.group(1).strip()
            
    #         return None
    #     except:
    #         return None

class EmailSender:
    """郵件發送器 - 增強版HTML格式"""
    
    def __init__(self, smtp_server: str, smtp_port: int, email: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
    
    def send_daily_report(self, articles: List[Dict], date: str):
        """發送每日報告"""
        if not articles:
            logger.info("今日無新文章，不發送郵件")
            return
        
        # 生成HTML郵件內容
        html_content = self._generate_enhanced_html_report(articles, date)
        
        # 建立郵件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'🤖 AI新聞日報 - {date} ({len(articles)}篇精選)'
        msg['From'] = self.email
        msg['To'] = self.email
        
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)
        
        # 發送郵件
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
            
            logger.info(f"郵件發送成功: {len(articles)} 篇文章")
            
        except Exception as e:
            logger.error(f"郵件發送失敗: {e}")
    
    def _generate_enhanced_html_report(self, articles: List[Dict], date: str) -> str:
        """生成增強版HTML報告 - 包含文章原始發布時間"""
        # 按相關性分數排序
        sorted_articles = sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # 分類統計
        categories = {}
        sentiment_stats = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for article in sorted_articles:
            category = article.get('category', 'Other')
            categories[category] = categories.get(category, 0) + 1
            
            sentiment = article.get('sentiment_score', 0)
            if sentiment > 0.3:
                sentiment_stats['positive'] += 1
            elif sentiment < -0.3:
                sentiment_stats['negative'] += 1
            else:
                sentiment_stats['neutral'] += 1
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>AI新聞日報 - {date}</title>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px;
                    line-height: 1.6; 
                    color: #333;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px 20px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 600;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                    font-size: 16px;
                }}
                .stats {{ 
                    background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
                    padding: 20px;
                    margin: 0;
                    border-bottom: 1px solid #e1e8ed;
                }}
                .stats h3 {{
                    color: #1a73e8;
                    margin-top: 0;
                    font-size: 18px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .stat-item {{
                    text-align: center;
                    padding: 10px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                .stat-number {{
                    display: block;
                    font-size: 24px;
                    font-weight: bold;
                    color: #1a73e8;
                }}
                .stat-label {{
                    font-size: 12px;
                    color: #666;
                    margin-top: 5px;
                }}
                .content {{
                    padding: 20px;
                }}
                .section-title {{
                    color: #1a73e8;
                    font-size: 20px;
                    font-weight: 600;
                    margin: 30px 0 15px 0;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #e8f4fd;
                }}
                .article {{ 
                    margin: 20px 0; 
                    padding: 20px;
                    border-left: 4px solid #1a73e8; 
                    background: linear-gradient(135deg, #fafbfc 0%, #f8f9fa 100%);
                    border-radius: 0 8px 8px 0;
                    transition: transform 0.2s ease;
                }}
                .article:hover {{
                    transform: translateX(5px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}
                .article h3 {{ 
                    color: #1a73e8; 
                    margin-top: 0; 
                    font-size: 18px;
                    line-height: 1.4;
                }}
                .article h3 a {{
                    color: #1a73e8;
                    text-decoration: none;
                }}
                .article h3 a:hover {{
                    text-decoration: underline;
                }}
                .meta {{ 
                    color: #666; 
                    font-size: 14px; 
                    margin: 8px 0;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    align-items: center;
                }}
                .meta-item {{
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }}
                .summary {{ 
                    margin: 15px 0;
                    font-size: 15px;
                    line-height: 1.6;
                    color: #444;
                    text-align: justify;
                }}
                .score {{ font-weight: bold; }}
                .high-score {{ color: #34a853; }}
                .medium-score {{ color: #ea4335; }}
                .low-score {{ color: #fbbc05; }}
                .category-tag {{
                    background: #e8f4fd;
                    color: #1a73e8;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 500;
                }}
                .url-link {{
                    display: inline-block;
                    margin-top: 10px;
                    padding: 8px 16px;
                    background: #1a73e8;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    font-size: 14px;
                    transition: background 0.2s ease;
                }}
                .url-link:hover {{
                    background: #1557b0;
                    color: white;
                    text-decoration: none;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    background: #f8f9fa;
                    color: #666;
                    font-size: 14px;
                    border-top: 1px solid #e1e8ed;
                }}
                .divider {{
                    height: 1px;
                    background: linear-gradient(90deg, transparent, #e1e8ed, transparent);
                    margin: 25px 0;
                }}
                @media (max-width: 600px) {{
                    .stats-grid {{
                        grid-template-columns: repeat(2, 1fr);
                    }}
                    .meta {{
                        flex-direction: column;
                        align-items: flex-start;
                        gap: 8px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI新聞日報</h1>
                    <p>日期: {date} | 共收集 {len(sorted_articles)} 篇精選文章</p>
                </div>
                
                <div class="stats">
                    <h3>📊 今日摘要統計</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-number">{len(sorted_articles)}</span>
                            <div class="stat-label">總文章數</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">{sum(1 for a in sorted_articles if a.get('relevance_score', 0) > 0.8)}</span>
                            <div class="stat-label">高相關性</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">{sentiment_stats['positive']}</span>
                            <div class="stat-label">正面新聞</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">{sentiment_stats['neutral']}</span>
                            <div class="stat-label">中性報導</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">{sentiment_stats['negative']}</span>
                            <div class="stat-label">負面消息</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">{round(sum(a.get('relevance_score', 0) for a in sorted_articles) / len(sorted_articles), 2) if sorted_articles else 0}</span>
                            <div class="stat-label">平均相關性</div>
                        </div>
                    </div>
                    
                    <h4 style="margin-top: 20px; margin-bottom: 10px; color: #1a73e8;">📈 分類分布</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;">
        """
        
        # 加入分類統計
        category_icons = {
            'AI Research': '🔬',
            'Industry News': '🏢', 
            'Biotech': '🧬',
            'Robotics': '🤖',
            'Other': '📱'
        }
        
        for category, count in categories.items():
            icon = category_icons.get(category, '📄')
            html += f'<div>{icon} <strong>{category}:</strong> {count}篇</div>'
        
        html += """
                    </div>
                </div>

                <div class="content">
                    <h2 class="section-title">🌟 今日精選文章</h2>
        """
        
        # 加入文章內容
        for i, article in enumerate(sorted_articles[:20], 1):  # 限制最多20篇
            relevance = article.get('relevance_score', 0)
            sentiment = article.get('sentiment_score', 0)
            
            # 相關性顏色
            if relevance >= 0.8:
                score_class = "high-score"
            elif relevance >= 0.6:
                score_class = "medium-score"
            else:
                score_class = "low-score"
            
            # 情感符號
            if sentiment > 0.3:
                sentiment_icon = "😊"
            elif sentiment < -0.3:
                sentiment_icon = "😟"
            else:
                sentiment_icon = "😐"
            
            # 文章URL - 確保有效連結
            article_url = article.get('url', '#')
            if not article_url.startswith('http'):
                article_url = '#'
            
            # 處理發布日期顯示
            published_date = article.get('published_date', '')
            date_display = ""
            if published_date:
                try:
                    from datetime import datetime
                    # 解析ISO格式日期
                    if 'T' in published_date:
                        pub_datetime = datetime.fromisoformat(published_date.replace('Z', '+00:00').replace('+00:00', ''))
                    else:
                        pub_datetime = datetime.fromisoformat(published_date)
                    
                    # 計算天數差
                    now = datetime.now()
                    days_diff = (now - pub_datetime).days
                    
                    if days_diff == 0:
                        date_display = f"📅 今天 {pub_datetime.strftime('%H:%M')}"
                    elif days_diff == 1:
                        date_display = f"📅 昨天 {pub_datetime.strftime('%H:%M')}"
                    elif days_diff <= 7:
                        date_display = f"📅 {days_diff}天前 ({pub_datetime.strftime('%m-%d %H:%M')})"
                    else:
                        date_display = f"📅 {pub_datetime.strftime('%Y-%m-%d %H:%M')}"
                except:
                    date_display = f"📅 {published_date[:10]}"
            
            html += f"""
            <div class="article">
                <h3>{i}. <a href="{article_url}" target="_blank">{article.get('title', '無標題')}</a></h3>
                <div class="meta">
                    <div class="meta-item">📰 來源: <strong>{article.get('source', '未知')}</strong></div>
                    <div class="meta-item">🏷️ <span class="category-tag">{article.get('category', 'Other')}</span></div>
                    <div class="meta-item"><span class="score {score_class}">⭐ 相關性: {relevance:.2f}</span></div>
                    <div class="meta-item">{sentiment_icon} 情感: {sentiment:.2f}</div>
                </div>
                <div class="publish-date">{date_display}</div>
                <div class="summary">
                    <strong>AI摘要:</strong> {article.get('ai_summary', article.get('summary', '無摘要'))}
                </div>
                <a href="{article_url}" target="_blank" class="url-link">📖 閱讀原文</a>
            </div>
            """
        
        html += """
                </div>

                <div class="footer">
                    <p><strong>📈 數據洞察</strong></p>
                    <p>本期AI新聞涵蓋了最新的技術突破、產業動態和研究進展</p>
                    <p>所有文章均附上原文連結，點擊即可深入閱讀</p>
                    <hr style="margin: 20px 0; border: none; height: 1px; background: #e1e8ed;">
                    <p style="margin: 0;">
                        此報告由AI新聞收集系統自動生成 | Powered by Claude AI<br>
                        <small>資料來源涵蓋23個權威AI/科技媒體 | 每日台北時間8:00更新</small>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

def main():
    """主程式"""
    start_time = time.time()
    logger.info("開始執行AI新聞收集程序...")
    
    # 從環境變數取得設定
    claude_api_key = os.environ.get('CLAUDE_API_KEY')
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD')
    
    if not all([claude_api_key, gmail_user, gmail_password]):
        logger.error("缺少必要的環境變數")
        return
    
    # 初始化組件
    data_manager = DataManager()
    rss_collector = RSSCollector()
    claude_processor = ClaudeProcessor(claude_api_key)
    email_sender = EmailSender('smtp.gmail.com', 587, gmail_user, gmail_password)
    
    # 收集文章
    logger.info("正在收集RSS文章...")
    articles = rss_collector.collect_rss_articles()
    max_process_time = 30 * 60  
    
    # 處理新文章
    new_articles = []
    processed_count = 0
    
    for article in articles:
        if time.time() - start_time > max_process_time:
            logger.warning("達到時間限制，停止處理新文章")
            break
        if not data_manager.article_exists(article.hash_id):
            # 使用Claude處理 - 生成長摘要
            logger.info(f"正在處理文章: {article.title[:50]}...")
            processed_article = claude_processor.classify_and_summarize(article)
            data_manager.save_article(processed_article)
            new_articles.append(processed_article)
            processed_count += 1
            
            # 每處理5篇文章暫停一下，避免API rate limit
            if processed_count % 3 == 0:
                    logger.info(f"已處理 {processed_count} 篇文章，暫停3秒...")
                    time.sleep(3)
    
    logger.info(f"處理了 {len(new_articles)} 篇新文章")
    
    # 準備今日報告
    today = datetime.now().strftime('%Y-%m-%d')
    today_articles = data_manager.get_articles_by_date(today)
    
    # 如果今日沒有新文章，取得最近的文章
    if not today_articles:
        try:
            df = pd.read_csv(data_manager.csv_file)
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
            recent_df = df.dropna(subset=['published_date']).tail(20)  # 最近20篇
            today_articles = recent_df.to_dict('records')
        except:
            today_articles = []
    
    # 發送郵件
    if today_articles:
        email_sender.send_daily_report(today_articles, today)
    else:
        logger.info("沒有文章可發送")
    total_time = time.time() - start_time
    logger.info(f"AI新聞收集程序執行完成! 總耗時: {total_time:.1f}秒")

if __name__ == "__main__":
    main()