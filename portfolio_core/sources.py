"""Import configured, trusted dataset/feed sources; never arbitrary caller URLs."""
import io
import os
import httpx
import pandas as pd


def dataset(source):
    allowed={'who':'WHO_DATASET_URL','cdc':'CDC_DATASET_URL','sports':'SPORTS_FEED_URL','streaming':'STREAMING_DATASET_URL'}
    variable=allowed.get(source)
    url=os.environ.get(variable or '')
    if not url or not url.startswith('https://'):raise ValueError(f'Configure a trusted HTTPS {variable or "source"}')
    chunks=[];size=0
    with httpx.stream('GET',url,timeout=45,follow_redirects=False) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            size+=len(chunk)
            if size>10_000_000:raise ValueError('Dataset exceeds 10 MB')
            chunks.append(chunk)
    # Normalize the remote schema using a server-controlled mapping.
    import json
    mapping=json.loads(os.environ.get(source.upper()+'_COLUMN_MAPPING','{}'))
    df=pd.read_csv(io.BytesIO(b''.join(chunks))).rename(columns=mapping)
    return df.to_dict('records')


def browse_products(query,urls):
    """Extract public product pages using BrowserBase+Playwright; no checkout actions."""
    import json
    from playwright.sync_api import sync_playwright
    from .ai import key
    if len(urls)>3:raise ValueError('At most three product pages per run')
    from urllib.parse import urlparse
    allowed=set(os.environ.get('BROWSER_ALLOWED_HOSTS','').split(','))
    if any(urlparse(url).hostname not in allowed or urlparse(url).scheme!='https' for url in urls):raise ValueError('Product host must be in server-configured BROWSER_ALLOWED_HOSTS')
    token=key('BROWSERBASE_API_KEY')
    r=httpx.post('https://api.browserbase.com/v1/sessions',headers={'x-bb-api-key':token},json={'projectId':key('BROWSERBASE_PROJECT_ID')},timeout=30);r.raise_for_status()
    session=r.json();pages=[]
    with sync_playwright() as p:
        browser=p.chromium.connect_over_cdp(session['connectUrl'])
        try:
            for url in urls:
                page=browser.new_page();page.goto(url,timeout=30000,wait_until='domcontentloaded')
                pages.append({'url':url,'title':page.title(),'content':page.locator('body').inner_text()[:15000]});page.close()
        finally:browser.close()
    return pages
