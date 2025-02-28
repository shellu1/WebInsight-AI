import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
from datetime import datetime
import os
import random
import hashlib
from typing import Set, Dict, List, Optional
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
from PIL import Image as PILImage
from io import BytesIO
import re
from dataclasses import dataclass
from pathlib import Path
from collections import deque

@dataclass
class PageContent:
    url: str
    title: str
    content: List[Dict]
    level: int  # For hierarchy in PDF

class UnifiedWebsiteScraper:
    def __init__(
        self,
        start_url: str,
        output_dir: str = "inciem_website_content",
        max_pages: int = 100,
        concurrent_requests: int = 5,
        delay_range: tuple = (1, 3)
    ):
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.scheme = urlparse(start_url).scheme
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.concurrent_requests = concurrent_requests
        self.delay_range = delay_range
        
        # Initialize tracking
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.queue: deque = deque([(start_url, 0)])  # (url, level)
        self.content_store: List[PageContent] = []
        
        # Setup
        self._setup_directories()
        self._setup_logging()
        self.session = self._setup_session()

    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)

    def _setup_logging(self):
        """Configure logging"""
        log_filename = f'scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

    def _setup_session(self) -> requests.Session:
        """Set up requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by handling protocol-relative URLs"""
        if url.startswith('//'):
            return f"{self.scheme}:{url}"
        return url

    def _download_image(self, img_url: str, base_url: str) -> Optional[str]:
        """Download and process image"""
        try:
            # Normalize URL
            img_url = self._normalize_url(img_url)
            if not bool(urlparse(img_url).netloc):
                img_url = urljoin(base_url, img_url)

            img_hash = hashlib.md5(img_url.encode()).hexdigest()
            img_path = f"{self.output_dir}/images/{img_hash}.jpg"

            if os.path.exists(img_path):
                return img_path

            response = self.session.get(img_url, timeout=10)
            response.raise_for_status()

            img = PILImage.open(BytesIO(response.content))
            
            # Handle transparency and convert to RGB
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = PILImage.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                elif img.mode == 'LA':
                    background.paste(img, mask=img.split()[1])
                elif img.mode == 'P':
                    background.paste(img, mask=img.convert('RGBA').split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Save optimized image
            img.save(img_path, "JPEG", quality=85, optimize=True)
            return img_path

        except Exception as e:
            logging.error(f"Error downloading image {img_url}: {str(e)}")
            return None

    def _extract_content(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Extract content blocks from page"""
        content_blocks = []
        
        main_content = soup.find(['article', 'main']) or soup.find('div', class_=re.compile(r'content|post|article|entry'))
        if not main_content:
            main_content = soup
        
        for element in main_content.find_all(['p', 'img', 'h1', 'h2', 'h3', 'h4', 'blockquote']):
            if element.name == 'img':
                img_url = element.get('src')
                if img_url:
                    img_url = self._normalize_url(img_url)
                    local_path = self._download_image(img_url, url)
                    if local_path:
                        content_blocks.append({
                            'type': 'image',
                            'src': local_path,
                            'alt': element.get('alt', '')
                        })
            else:
                text = element.get_text(strip=True)
                if text:
                    content_blocks.append({
                        'type': 'text',
                        'content': text,
                        'tag': element.name
                    })
        
        return content_blocks

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize links"""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            href = self._normalize_url(href)
            absolute_url = urljoin(base_url, href)
            if urlparse(absolute_url).netloc == self.base_domain:
                links.append(absolute_url)
        return list(set(links))

    def _fetch_page(self, url: str, level: int) -> Optional[PageContent]:
        """Fetch and process a single page"""
        try:
            time.sleep(random.uniform(*self.delay_range))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.title.string if soup.title else url
            content_blocks = self._extract_content(soup, url)
            
            # Add new URLs to queue
            new_links = self._extract_links(soup, url)
            for link in new_links:
                if link not in self.visited_urls and link not in [u for u, _ in self.queue]:
                    self.queue.append((link, level + 1))
            
            return PageContent(
                url=url,
                title=title,
                content=content_blocks,
                level=level
            )
            
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            self.failed_urls.add(url)
            return None

    def _create_unified_pdf(self) -> str:
        """Create a single PDF containing all content"""
        try:
            pdf_path = f"{self.output_dir}/website_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Define styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30
            )
            subtitle_style = ParagraphStyle(
                'CustomSubTitle',
                parent=styles['Title'],
                fontSize=20,
                spaceAfter=20
            )
            h2_style = ParagraphStyle(
                'CustomH2',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=12
            )
            h3_style = ParagraphStyle(
                'CustomH3',
                parent=styles['Heading3'],
                fontSize=16,
                spaceAfter=12
            )
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )
            
            # Build PDF content
            elements = []
            
            # Add main title
            elements.append(Paragraph(f"Website Content: {self.base_domain}", title_style))
            elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
            elements.append(PageBreak())
            
            # Add table of contents
            elements.append(Paragraph("Table of Contents", subtitle_style))
            for page in self.content_store:
                indent = "&nbsp;" * (page.level * 4)
                elements.append(Paragraph(f"{indent}{page.title}", body_style))
            elements.append(PageBreak())
            
            # Add content pages
            for page in self.content_store:
                # Page title with level-based indentation
                elements.append(Paragraph(page.title, subtitle_style))
                elements.append(Paragraph(f"Source: {page.url}", styles['Italic']))
                elements.append(Spacer(1, 20))
                
                # Page content
                for block in page.content:
                    if block['type'] == 'text':
                        style = body_style
                        if block['tag'] == 'h2':
                            style = h2_style
                        elif block['tag'] == 'h3':
                            style = h3_style
                        elements.append(Paragraph(block['content'], style))
                    
                    elif block['type'] == 'image':
                        try:
                            with PILImage.open(block['src']) as pil_img:
                                img_width, img_height = pil_img.size
                            
                            # Scale image to fit page
                            max_width = 6 * inch
                            scale = min(1, max_width / img_width)
                            
                            img = Image(block['src'])
                            img.drawWidth = img_width * scale
                            img.drawHeight = img_height * scale
                            
                            elements.append(Spacer(1, 12))
                            elements.append(img)
                            if block['alt']:
                                elements.append(Paragraph(block['alt'], styles['Italic']))
                            elements.append(Spacer(1, 12))
                        except Exception as e:
                            logging.error(f"Error adding image to PDF: {str(e)}")
                
                elements.append(PageBreak())
            
            # Build PDF
            doc.build(elements)
            return pdf_path
            
        except Exception as e:
            logging.error(f"Error creating PDF: {str(e)}")
            return None

    def scrape_website(self):
        """Main method to scrape website and create unified PDF"""
        logging.info(f"Starting scrape of {self.start_url}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            while self.queue and len(self.visited_urls) < self.max_pages:
                current_batch = []
                for _ in range(min(self.concurrent_requests, len(self.queue))):
                    if self.queue:
                        url, level = self.queue.popleft()
                        if url not in self.visited_urls:
                            current_batch.append((url, level))
                
                if not current_batch:
                    continue
                
                # Process batch
                future_to_url = {
                    executor.submit(self._fetch_page, url, level): (url, level) 
                    for url, level in current_batch
                }
                
                for future in concurrent.futures.as_completed(future_to_url):
                    url, level = future_to_url[future]
                    self.visited_urls.add(url)
                    
                    try:
                        content = future.result()
                        if content:
                            self.content_store.append(content)
                            logging.info(f"Processed {url}")
                    except Exception as e:
                        logging.error(f"Error processing {url}: {str(e)}")
                        self.failed_urls.add(url)
        
        # Create unified PDF
        logging.info("Creating unified PDF...")
        pdf_path = self._create_unified_pdf()
        if pdf_path:
            logging.info(f"PDF created successfully: {pdf_path}")
        
        # Log summary
        logging.info(f"Scraping complete. Processed {len(self.visited_urls)} pages. "
                    f"Failed URLs: {len(self.failed_urls)}")

def main():
    try:
        scraper = UnifiedWebsiteScraper(
            start_url="https://inciem.com/",  # Replace with your target website
            max_pages=100,
            concurrent_requests=5,
            delay_range=(1, 3)
        )
        
        scraper.scrape_website()
        
    except Exception as e:
        logging.error(f"Scraping failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()