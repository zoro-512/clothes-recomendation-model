"""
Web search utility to find product URLs.
Now simplified to use product URLs directly from dataset.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_product_url(
    product_name: str, 
    article_id: Optional[str] = None,
    product_type: Optional[str] = None,
    color: Optional[str] = None,
    gender: Optional[str] = None,
    description: Optional[str] = None,
    product_url: Optional[str] = None
) -> Optional[str]:
    """
    Return product URL directly from dataset.
    
    Simplified function that uses the product_url from the dataset
    instead of performing web searches.
    """
    # If product_url is provided, use it directly
    if product_url and product_url.strip():
        logger.info(f"Using product URL from dataset: {product_url}")
        return product_url.strip()
    
    # Fallback: construct Myntra URL if article_id is available
    if article_id:
        fallback_url = f"https://www.myntra.com/product/{article_id}"
        logger.info(f"Constructed fallback URL: {fallback_url}")
        return fallback_url
    
    # Final fallback: return None
    logger.warning("No product URL available")
    return None
