"""
Incremental Reddit scraper for r/singapore using PRAW.

Designed to be run as a cron job (e.g. hourly or daily).
Reads existing monthly CSVs to find the latest scraped timestamp,
then fetches only new posts/comments since then and appends them
to the appropriate monthly CSV files.

Usage:
    python scrape_incremental.py
    python scrape_incremental.py --subreddit singapore --limit 500
    python scrape_incremental.py --download-media

Cron example (run every 6 hours):
    0 */6 * * * /path/to/.venv/bin/python /path/to/scrape_incremental.py >> /path/to/scrape.log 2>&1
"""

import argparse
import glob
import logging
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import praw
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r'^[a-z0-9]+$')
_ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov'}


def _init_reddit() -> praw.Reddit:
    """Initialise a PRAW Reddit instance, validating required env vars."""
    missing = [k for k in ('REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT')
               if not os.environ.get(k)]
    if missing:
        raise ValueError(
            "Missing required environment variables: %s. "
            "Copy .env.example to .env and fill in the values." % ', '.join(missing)
        )
    return praw.Reddit(
        client_id=os.environ['REDDIT_CLIENT_ID'],
        client_secret=os.environ['REDDIT_CLIENT_SECRET'],
        user_agent=os.environ['REDDIT_USER_AGENT'],
    )


def get_file_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm', '.mov']:
        if path.endswith(ext):
            return ext
    return '.jpg'


def download_media_file(url: str, filepath: str, max_retries: int = 3) -> bool:
    if os.path.exists(filepath):
        return True
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=30, stream=True)
            resp.raise_for_status()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def get_latest_timestamp(subreddit: str, data_dir: str = '.') -> pd.Timestamp | None:
    """Find the most recent created_utc across all existing post CSVs."""
    pattern = os.path.join(data_dir, f'{subreddit}_posts_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None

    latest_ts = None
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['created_utc'])
            if df.empty:
                continue
            # Parse as datetime with UTC timezone
            ts = pd.to_datetime(df['created_utc'], utc=True).max()
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
        except Exception as e:
            log.warning("Warning: could not read %s: %s", f, e)

    return latest_ts


def get_existing_post_ids(subreddit: str, data_dir: str = '.') -> set[str]:
    """Return a set of all post IDs already saved, to avoid duplicates."""
    pattern = os.path.join(data_dir, f'{subreddit}_posts_*.csv')
    ids: set[str] = set()
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f, usecols=['id'])
            ids.update(df['id'].astype(str))
        except Exception as e:
            log.warning("Could not read %s: %s", f, e)
    return ids


def posts_csv_path(subreddit: str, dt: datetime, data_dir: str = '.') -> str:
    """Return the monthly posts CSV path for a given datetime."""
    month_name = dt.strftime('%B').lower()
    return os.path.join(data_dir, f'{subreddit}_posts_{dt.year}_{dt.month:02d}_{month_name}.csv')


def comments_csv_path(subreddit: str, dt: datetime, data_dir: str = '.') -> str:
    """Return the monthly comments CSV path for a given datetime."""
    month_name = dt.strftime('%B').lower()
    return os.path.join(data_dir, f'{subreddit}_comments_{dt.year}_{dt.month:02d}_{month_name}.csv')


def append_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Append DataFrame to CSV, writing header only if the file doesn't exist."""
    write_header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=write_header, index=False)


def fetch_new_posts(
    reddit: praw.Reddit,
    subreddit_name: str,
    since_ts: pd.Timestamp | None,
    existing_ids: set[str],
    limit: int = 1000,
) -> list:
    """
    Fetch new posts from subreddit.new() up to `limit`.
    Stops early when reaching posts at or before since_ts.
    Returns list of PRAW submission objects.
    """
    subreddit = reddit.subreddit(subreddit_name)
    new_posts = []

    since_epoch = since_ts.timestamp() if since_ts else 0

    for submission in subreddit.new(limit=limit):
        if submission.created_utc <= since_epoch:
            break
        if str(submission.id) in existing_ids:
            continue
        new_posts.append(submission)

    return new_posts


def submission_to_dict(submission, media_files: list[str] | None = None) -> dict:
    return {
        'id': submission.id,
        'title': submission.title,
        'selftext': submission.selftext,
        'author': str(submission.author) if submission.author else None,
        'score': submission.score,
        'upvote_ratio': submission.upvote_ratio,
        'num_comments': submission.num_comments,
        'created_utc': submission.created_utc,
        'edited': submission.edited,
        'distinguished': submission.distinguished,
        'stickied': submission.stickied,
        'over_18': submission.over_18,
        'spoiler': submission.spoiler,
        'locked': submission.locked,
        'archived': submission.archived,
        'is_original_content': submission.is_original_content,
        'is_self': submission.is_self,
        'permalink': submission.permalink,
        'url': submission.url,
        'domain': submission.domain,
        'subreddit': str(submission.subreddit),
        'subreddit_id': submission.subreddit_id,
        'link_flair_text': submission.link_flair_text,
        'link_flair_css_class': submission.link_flair_css_class,
        'author_flair_text': submission.author_flair_text,
        'author_flair_css_class': submission.author_flair_css_class,
        'gilded': submission.gilded,
        'total_awards_received': submission.total_awards_received,
        'treatment_tags': getattr(submission, 'treatment_tags', None),
        'is_video': submission.is_video,
        'media': str(submission.media) if submission.media else None,
        'secure_media': str(submission.secure_media) if submission.secure_media else None,
        'thumbnail': submission.thumbnail,
        'post_hint': getattr(submission, 'post_hint', None),
        'preview': str(getattr(submission, 'preview', None)) if hasattr(submission, 'preview') else None,
        'downloaded_media': '|'.join(media_files) if media_files else None,
    }


def comment_to_dict(comment, post_id: str) -> dict:
    return {
        'id': comment.id,
        'parent_id': comment.parent_id,
        'post_id': post_id,
        'body': comment.body,
        'author': str(comment.author) if comment.author else None,
        'score': comment.score,
        'created_utc': comment.created_utc,
        'edited': comment.edited,
        'distinguished': comment.distinguished,
        'stickied': comment.stickied,
        'is_submitter': comment.is_submitter,
        'author_flair_text': comment.author_flair_text,
        'author_flair_css_class': comment.author_flair_css_class,
        'gilded': comment.gilded,
        'total_awards_received': comment.total_awards_received,
        'permalink': comment.permalink,
        'subreddit': str(comment.subreddit),
        'subreddit_id': comment.subreddit_id,
        'depth': comment.depth,
        'controversiality': comment.controversiality,
        'collapsed': comment.collapsed,
        'collapsed_reason': comment.collapsed_reason,
        'locked': comment.locked,
    }


def run(subreddit_name: str = 'singapore', limit: int = 1000, data_dir: str = '.', download_media: bool = False) -> None:
    reddit = _init_reddit()
    now = datetime.now(timezone.utc)
    log.info("[%s] Starting incremental scrape of r/%s", now.strftime('%Y-%m-%d %H:%M:%S UTC'), subreddit_name)

    # Find latest already-scraped timestamp and existing IDs
    latest_ts = get_latest_timestamp(subreddit_name, data_dir)
    existing_ids = get_existing_post_ids(subreddit_name, data_dir)

    if latest_ts is not None:
        log.info("Latest saved post: %s | Known posts: %d", latest_ts, len(existing_ids))
    else:
        log.info("No existing data found — will fetch up to limit from .new()")

    # Fetch new posts
    new_posts = fetch_new_posts(reddit, subreddit_name, latest_ts, existing_ids, limit=limit)
    log.info("New posts found: %d", len(new_posts))

    if not new_posts:
        log.info("Nothing to do.")
        return

    # Group posts by month and save
    posts_by_month: dict = {}
    comments_by_month: dict = {}

    for i, submission in enumerate(new_posts, 1):
        if i % 50 == 0:
            log.info("  Processing post %d/%d...", i, len(new_posts))

        post_dt = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        month_key = (post_dt.year, post_dt.month)

        media_files: list[str] = []
        if download_media and not submission.is_self and submission.url:
            sub_id = str(submission.id)
            if not _SAFE_ID_RE.match(sub_id):
                log.warning("Skipping media download: unsafe submission id %r", sub_id)
            else:
                ext = get_file_extension(submission.url)
                if ext not in _ALLOWED_EXTENSIONS:
                    log.warning("Skipping media download: disallowed extension %r for post %s", ext, sub_id)
                else:
                    media_folder = os.path.join(data_dir, 'media', subreddit_name,
                                                f'{post_dt.year}-{post_dt.month:02d}',
                                                sub_id)
                    filepath = os.path.join(media_folder, f'media_1{ext}')
                    if download_media_file(submission.url, filepath):
                        media_files.append(filepath)
                    time.sleep(0.5)

        post_dict = submission_to_dict(submission, media_files)
        posts_by_month.setdefault(month_key, []).append(post_dict)

        # Fetch comments
        try:
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                c = comment_to_dict(comment, submission.id)
                comments_by_month.setdefault(month_key, []).append(c)
        except Exception as e:
            log.warning("  Warning: could not fetch comments for %s: %s", submission.id, e)

    # Write to monthly CSVs
    total_posts = 0
    total_comments = 0

    for (year, month), post_list in sorted(posts_by_month.items()):
        df_posts = pd.DataFrame(post_list)
        df_posts['created_utc'] = pd.to_datetime(df_posts['created_utc'], unit='s')
        df_posts['edited'] = df_posts['edited'].apply(
            lambda x: pd.to_datetime(x, unit='s') if x and x is not False else None
        )

        posts_path = posts_csv_path(subreddit_name, datetime(year, month, 1), data_dir)
        append_to_csv(df_posts, posts_path)
        total_posts += len(df_posts)
        log.info("  Appended %d posts → %s", len(df_posts), os.path.basename(posts_path))

        comment_list = comments_by_month.get((year, month), [])
        if comment_list:
            df_comments = pd.DataFrame(comment_list)
            df_comments['created_utc'] = pd.to_datetime(df_comments['created_utc'], unit='s')
            df_comments['edited'] = df_comments['edited'].apply(
                lambda x: pd.to_datetime(x, unit='s') if x and x is not False else None
            )
            comments_path = comments_csv_path(subreddit_name, datetime(year, month, 1), data_dir)
            append_to_csv(df_comments, comments_path)
            total_comments += len(df_comments)
            log.info("  Appended %d comments → %s", len(df_comments), os.path.basename(comments_path))

    log.info("Done. Added %d posts and %d comments.", total_posts, total_comments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental Reddit scraper for cron use')
    parser.add_argument('--subreddit', default='singapore')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Max posts to fetch from .new() (Reddit cap: 1000)')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to read existing CSVs from and write new ones to (e.g. singapore_subreddit_scrapped)')
    parser.add_argument('--data-dir', default=None,
                        help='Alias for --output-dir (deprecated)')
    parser.add_argument('--download-media', action='store_true',
                        help='Download media files for non-self posts')
    args = parser.parse_args()

    data_dir = args.data_dir or args.output_dir
    run(args.subreddit, args.limit, data_dir, args.download_media)
