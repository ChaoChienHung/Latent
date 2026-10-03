import argparse
import logging
import mimetypes
import os
import re
import time
from calendar import monthrange
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

ARCTIC_SHIFT_BASE = 'https://arctic-shift.photon-reddit.com/api'
MAX_BATCH_RETRIES = 5

_SAFE_ID_RE = re.compile(r'^[a-z0-9]+$')
_ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov'}


def download_media(url: str, filepath: str, max_retries: int = 3) -> tuple[bool, str]:
    """Download media file from URL with retry logic"""
    if os.path.exists(filepath):
        return True, "File already exists"

    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return True, "Downloaded successfully"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return False, "Failed after %d attempts: %s" % (max_retries, e)


def get_file_extension(url: str, content_type: str | None = None) -> str:
    """Get appropriate file extension from URL or content type"""
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Common Reddit media extensions
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov']:
        if path.endswith(ext):
            return ext

    # Fallback to content type
    if content_type:
        ext = mimetypes.guess_extension(content_type)
        if ext:
            return ext

    # Default fallback
    return '.jpg'


def fetch_posts_arctic(subreddit: str, after_ts: float, before_ts: float, limit: int = 100) -> list[dict]:
    """Fetch posts from Arctic Shift API for a given time window."""
    params = {
        'subreddit': subreddit,
        'after': int(after_ts),
        'before': int(before_ts),
        'limit': limit,
        'sort': 'asc',
    }
    resp = requests.get(f'{ARCTIC_SHIFT_BASE}/posts/search', params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get('data') or []


def fetch_comments_arctic(post_id: str, limit: int = 100, max_pages: int = 50) -> list[dict]:
    """Fetch all comments for a post from Arctic Shift API."""
    comments = []
    params = {
        'link_id': post_id,
        'limit': limit,
        'sort': 'asc',
    }
    page_count = 0

    while True:
        resp = requests.get(f'{ARCTIC_SHIFT_BASE}/comments/search', params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get('data') or []
        if not batch:
            break
        comments.extend(batch)
        page_count += 1
        if len(batch) < limit:
            break
        if page_count >= max_pages:
            log.warning("Reached max_pages=%d for post %s; truncating comment fetch", max_pages, post_id)
            break
        # Paginate using last comment's created_utc as the new 'after'
        params['after'] = batch[-1]['created_utc']
        time.sleep(0.3)

    return comments


def scrape_subreddit_month(
    subreddit: str,
    year: int = 2025,
    month: int = 1,
    download_media_files: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:

    # Calculate month boundaries (UTC timestamps)
    month_start = datetime(year, month, 1, tzinfo=timezone.utc).timestamp()
    last_day = monthrange(year, month)[1]
    month_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc).timestamp()

    month_name = datetime(year, month, 1).strftime('%B')
    log.info(
        "Scraping posts from %s %d via Arctic Shift (%s → %s)",
        month_name, year,
        datetime.fromtimestamp(month_start),
        datetime.fromtimestamp(month_end),
    )

    posts_data = []
    comments_data = []
    posts_processed = 0
    current_after = month_start
    retry_count = 0

    while True:
        try:
            batch = fetch_posts_arctic(subreddit, current_after, month_end, limit=100)
            retry_count = 0
        except requests.exceptions.RequestException as e:
            retry_count += 1
            log.error("Error fetching posts batch (attempt %d/%d): %s", retry_count, MAX_BATCH_RETRIES, e)
            if retry_count >= MAX_BATCH_RETRIES:
                log.error("Max retries exceeded for %s %d-%02d; aborting month", subreddit, year, month)
                break
            time.sleep(5)
            continue

        if not batch:
            break

        for post in batch:
            posts_processed += 1
            if posts_processed % 50 == 0:
                log.info("  Processed %d posts...", posts_processed)

            if posts_processed == 1:
                log.info(
                    "  First post: %s (id: %s)",
                    datetime.fromtimestamp(post['created_utc']),
                    post['id'],
                )

            # Download media if enabled
            media_files = []
            url = post.get('url', '')
            is_self = post.get('is_self', True)
            if download_media_files and url and not is_self:
                post_id = post.get('id', '')
                if not _SAFE_ID_RE.match(post_id):
                    log.warning("Skipping media download: unsafe post id %r", post_id)
                else:
                    ext = get_file_extension(url)
                    if ext not in _ALLOWED_EXTENSIONS:
                        log.warning("Skipping media download: disallowed extension %r for post %s", ext, post_id)
                    else:
                        media_folder = f"media/{year}-{month:02d}/{post_id}"
                        os.makedirs(media_folder, exist_ok=True)
                        filepath = os.path.join(media_folder, f"media_1{ext}")
                        success, _ = download_media(url, filepath)
                        if success:
                            media_files.append(filepath)
                        time.sleep(0.5)

            post_dict = {
                'id': post.get('id'),
                'title': post.get('title'),
                'selftext': post.get('selftext'),
                'author': post.get('author'),
                'score': post.get('score'),
                'upvote_ratio': post.get('upvote_ratio'),
                'num_comments': post.get('num_comments'),
                'created_utc': post.get('created_utc'),
                'edited': post.get('edited'),
                'distinguished': post.get('distinguished'),
                'stickied': post.get('stickied'),
                'over_18': post.get('over_18'),
                'spoiler': post.get('spoiler'),
                'locked': post.get('locked'),
                'archived': post.get('archived'),
                'is_original_content': post.get('is_original_content'),
                'is_self': post.get('is_self'),
                'permalink': post.get('permalink'),
                'url': post.get('url'),
                'domain': post.get('domain'),
                'subreddit': post.get('subreddit'),
                'subreddit_id': post.get('subreddit_id'),
                'link_flair_text': post.get('link_flair_text'),
                'link_flair_css_class': post.get('link_flair_css_class'),
                'author_flair_text': post.get('author_flair_text'),
                'author_flair_css_class': post.get('author_flair_css_class'),
                'gilded': post.get('gilded'),
                'total_awards_received': post.get('total_awards_received'),
                'is_video': post.get('is_video'),
                'media': str(post.get('media')) if post.get('media') else None,
                'thumbnail': post.get('thumbnail'),
                'post_hint': post.get('post_hint'),
                'downloaded_media': '|'.join(media_files) if media_files else None,
            }
            posts_data.append(post_dict)

            # Fetch comments for this post
            try:
                raw_comments = fetch_comments_arctic(post['id'])
                for comment in raw_comments:
                    comment_dict = {
                        'id': comment.get('id'),
                        'parent_id': comment.get('parent_id'),
                        'post_id': post['id'],
                        'body': comment.get('body'),
                        'author': comment.get('author'),
                        'score': comment.get('score'),
                        'created_utc': comment.get('created_utc'),
                        'edited': comment.get('edited'),
                        'distinguished': comment.get('distinguished'),
                        'stickied': comment.get('stickied'),
                        'is_submitter': comment.get('is_submitter'),
                        'author_flair_text': comment.get('author_flair_text'),
                        'author_flair_css_class': comment.get('author_flair_css_class'),
                        'gilded': comment.get('gilded'),
                        'total_awards_received': comment.get('total_awards_received'),
                        'permalink': comment.get('permalink'),
                        'subreddit': comment.get('subreddit'),
                        'subreddit_id': comment.get('subreddit_id'),
                        'depth': comment.get('depth'),
                        'controversiality': comment.get('controversiality'),
                        'collapsed': comment.get('collapsed'),
                        'locked': comment.get('locked'),
                    }
                    comments_data.append(comment_dict)
            except Exception as e:
                log.error("Error fetching comments for post %s: %s", post['id'], e)

        # Paginate: use the last post's created_utc as the new 'after' for the next page
        current_after = batch[-1]['created_utc']

        # If we got fewer than 100, we've reached the end
        if len(batch) < 100:
            break

        time.sleep(1)  # be polite between pages

    # Create DataFrames
    df_posts = pd.DataFrame(posts_data)
    df_comments = pd.DataFrame(comments_data)

    # Convert timestamps to datetime
    if not df_posts.empty:
        df_posts['created_utc'] = pd.to_datetime(df_posts['created_utc'], unit='s')
        df_posts['edited'] = df_posts['edited'].apply(
            lambda x: pd.to_datetime(x, unit='s') if x and x is not False else None
        )

    if not df_comments.empty:
        df_comments['created_utc'] = pd.to_datetime(df_comments['created_utc'], unit='s')
        df_comments['edited'] = df_comments['edited'].apply(
            lambda x: pd.to_datetime(x, unit='s') if x and x is not False else None
        )

    log.info("Completed %s %d: %d posts, %d comments", month_name, year, len(df_posts), len(df_comments))
    return df_posts, df_comments, month_name


def scrape_year_by_months(subreddit: str, year: int = 2025, download_media: bool = False) -> None:
    """Scrape an entire year by processing each month separately"""
    media_status = "WITH media download" if download_media else "WITHOUT media download"
    log.info("Starting to scrape r/%s for %d by months (%s)...", subreddit, year, media_status)

    total_posts = 0
    total_comments = 0

    for month in range(1, 13):
        try:
            log.info("=" * 60)
            log.info("Processing month %d/12 of %d", month, year)
            log.info("=" * 60)

            monthly_posts, monthly_comments, month_name = scrape_subreddit_month(
                subreddit, year=year, month=month, download_media_files=download_media
            )

            if not monthly_posts.empty:
                posts_filename = f'singapore_posts_{year}_{month:02d}_{month_name.lower()}.csv'
                monthly_posts.to_csv(posts_filename, index=False)
                log.info("Posts saved to %s", posts_filename)
                total_posts += len(monthly_posts)

            if not monthly_comments.empty:
                comments_filename = f'singapore_comments_{year}_{month:02d}_{month_name.lower()}.csv'
                monthly_comments.to_csv(comments_filename, index=False)
                log.info("Comments saved to %s", comments_filename)
                total_comments += len(monthly_comments)

            log.info("Month %d completed: %d posts, %d comments", month, len(monthly_posts), len(monthly_comments))
            log.info("Running total: %d posts, %d comments", total_posts, total_comments)

        except Exception as e:
            log.error("Error processing month %d: %s", month, e)
            continue

    log.info("=" * 60)
    log.info("SCRAPING COMPLETED FOR %d", year)
    log.info("Total: %d posts, %d comments", total_posts, total_comments)
    log.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape a subreddit year by months via Arctic Shift')
    parser.add_argument('--subreddit', default='singapore', help='Subreddit name to scrape')
    parser.add_argument('--year', type=int, default=2025, help='Year to scrape')
    parser.add_argument('--download-media', action='store_true', help='Download media files for non-self posts')
    args = parser.parse_args()
    scrape_year_by_months(args.subreddit, year=args.year, download_media=args.download_media)
