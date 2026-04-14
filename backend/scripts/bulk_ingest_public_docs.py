from __future__ import annotations

"""批量抓取公开政策网页并直接入库。

优先从上海市政府公开页面抓取政策类文章，不足时再补充 gov.cn 的政策正文页。
目标是快速扩充公开知识库，而不是构建一个通用搜索引擎，因此这里采用：

1. 少量 seed page + 轻量 BFS 的方式发现文章链接
2. 只抓取明确的正文页 URL 模式
3. 下载为本地 html 后复用现有 process_and_store_document() 入库
"""

import argparse
import json
import re
import time
from collections import deque
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from backend.retrieval.pipeline import has_document_source, process_and_store_document


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}


SHANGHAI_SEEDS = [
    "https://www.shanghai.gov.cn/",
    "https://www.shanghai.gov.cn/nw2315/index.html",
    "https://www.shanghai.gov.cn/nw2318/index.html",
    "https://www.shanghai.gov.cn/nw12344/index.html",
]

GOV_CN_SEEDS = [
    "https://www.gov.cn/zhengce/",
    "https://www.gov.cn/yaowen/",
    "https://www.gov.cn/fuwu/",
]


LINK_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)

SHANGHAI_ARTICLE_RE = re.compile(r"^https://www\.shanghai\.gov\.cn/nw\d+/\d{8}/[A-Za-z0-9]+\.html$")
GOV_CN_ARTICLE_RE = re.compile(
    r"^https://www\.gov\.cn/"
    r"(zhengce/content/\d{6}/content_\d+\.htm|"
    r"zhengce/\d{4}-\d{2}/\d{2}/content_\d+\.htm|"
    r"zhengce/\d{6}/content_\d+\.htm)$"
)

POLICY_KEYWORDS = (
    "通知",
    "政策",
    "申报",
    "实施",
    "办法",
    "意见",
    "方案",
    "指南",
    "扶持",
    "奖励",
    "资助",
    "专项",
    "若干",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _output_dir() -> Path:
    path = _project_root() / "data" / "test_docs" / "public_bulk"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path() -> Path:
    return _output_dir() / "manifest.jsonl"


def _fetch(url: str, timeout: int = 25) -> requests.Response:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    return response


def _extract_links(base_url: str, html: str) -> list[str]:
    links: list[str] = []
    for raw in LINK_RE.findall(html):
        url = urljoin(base_url, raw).split("#", 1)[0]
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        links.append(url)
    return links


def _extract_title(html: str, fallback: str) -> str:
    match = TITLE_RE.search(html)
    if not match:
        return fallback
    title = unescape(re.sub(r"\s+", " ", match.group(1))).strip()
    return title or fallback


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"</(p|div|li|tr|h1|h2|h3|h4|h5|h6|table|section|article)>", "\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_policy_like(title: str, text: str) -> bool:
    haystack = f"{title}\n{text[:4000]}"
    return any(keyword in haystack for keyword in POLICY_KEYWORDS)


def _crawl_article_urls(
    seeds: list[str],
    article_pattern: re.Pattern[str],
    allowed_domain: str,
    max_pages: int,
    max_urls: int,
) -> list[str]:
    queue = deque((seed, 0) for seed in seeds)
    seen_pages: set[str] = set()
    article_urls: list[str] = []
    seen_articles: set[str] = set()

    while queue and len(seen_pages) < max_pages and len(article_urls) < max_urls:
        current_url, depth = queue.popleft()
        if current_url in seen_pages:
            continue
        seen_pages.add(current_url)
        try:
            response = _fetch(current_url)
        except Exception:
            continue

        for link in _extract_links(current_url, response.text):
            parsed = urlparse(link)
            if parsed.netloc != allowed_domain:
                continue
            if article_pattern.match(link):
                if link not in seen_articles:
                    seen_articles.add(link)
                    article_urls.append(link)
                continue

            # 只在同站点 html 页面里做很浅的 BFS，避免抓太深。
            if depth >= 2:
                continue
            if not parsed.path.endswith((".html", ".htm", "/")):
                continue
            if link not in seen_pages:
                queue.append((link, depth + 1))

    return article_urls


def _iter_candidates(target_count: int) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []

    shanghai_urls = _crawl_article_urls(
        seeds=SHANGHAI_SEEDS,
        article_pattern=SHANGHAI_ARTICLE_RE,
        allowed_domain="www.shanghai.gov.cn",
        max_pages=140,
        max_urls=max(target_count * 2, 160),
    )
    for url in shanghai_urls:
        candidates.append({"url": url, "region": "上海"})

    gov_urls = _crawl_article_urls(
        seeds=GOV_CN_SEEDS,
        article_pattern=GOV_CN_ARTICLE_RE,
        allowed_domain="www.gov.cn",
        max_pages=120,
        max_urls=max(target_count * 2, 160),
    )
    for url in gov_urls:
        candidates.append({"url": url, "region": "全国"})

    return candidates


def _safe_name(url: str) -> str:
    parsed = urlparse(url)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", parsed.path.strip("/"))
    return slug.strip("_")[:120] or "doc"


def ingest_public_docs(target_count: int = 100, pause_seconds: float = 0.2) -> dict[str, Any]:
    output_dir = _output_dir()
    manifest = _manifest_path()

    candidates = _iter_candidates(target_count=target_count)
    ingested = 0
    skipped_existing = 0
    skipped_non_policy = 0
    failed: list[dict[str, str]] = []
    written_files: list[str] = []

    for item in candidates:
        if ingested >= target_count:
            break

        url = item["url"]
        region = item["region"]
        if has_document_source(url):
            skipped_existing += 1
            continue

        try:
            response = _fetch(url)
            html = response.text
            title = _extract_title(html, fallback=_safe_name(url))
            text = _html_to_text(html)
            if len(text) < 500 or not _is_policy_like(title, text):
                skipped_non_policy += 1
                continue

            file_name = f"{region}_{_safe_name(url)}.html"
            file_path = output_dir / file_name
            file_path.write_text(html, encoding="utf-8")

            chunk_count = process_and_store_document(
                str(file_path),
                metadata_overrides={
                    "source": url,
                    "upload_name": file_name,
                    "title": title,
                    "region": region,
                    "doc_category": "policy",
                    "source_type": "html",
                    "crawl_batch": "public_bulk_20260318",
                },
            )

            if chunk_count <= 0:
                skipped_existing += 1
                continue

            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "url": url,
                            "title": title,
                            "region": region,
                            "file_name": file_name,
                            "chunks_inserted": chunk_count,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            ingested += 1
            written_files.append(file_name)
            time.sleep(pause_seconds)
        except Exception as exc:
            failed.append({"url": url, "error": str(exc)})

    return {
        "target_count": target_count,
        "ingested": ingested,
        "skipped_existing": skipped_existing,
        "skipped_non_policy": skipped_non_policy,
        "failed": failed[:20],
        "output_dir": str(output_dir),
        "manifest": str(manifest),
        "written_files_sample": written_files[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="批量抓取公开政策网页并入库")
    parser.add_argument("--target-count", type=int, default=100, help="目标新增入库文档数")
    parser.add_argument("--pause-seconds", type=float, default=0.15, help="每次抓取后稍作停顿")
    args = parser.parse_args()
    result = ingest_public_docs(target_count=args.target_count, pause_seconds=args.pause_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
