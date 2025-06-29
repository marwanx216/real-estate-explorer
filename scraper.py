# scraper.py

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

BASE_URL = "https://aqarmap.com.eg/en/for-sale/property-type/cairo/?page="


def fetch_listings_from_page(page: int) -> list:
    url = BASE_URL + str(page)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to load page {page}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    listings = soup.find_all("div", class_="listing-card")

    result = []

    for listing in listings:
        data = extract_listing_data(listing)
        result.append(data)

    return result


def extract_listing_data(listing) -> dict:
    def safe_text(tag, default="N/A"):
        return tag.get_text(strip=True) if tag else default

    try:
        title = safe_text(listing.find("h2"))
        price = safe_text(listing.find("span", class_="text-title_4"))
        price_per_meter = safe_text(listing.find("p", class_="text-gray__dark_1"))

        location_tag = listing.find("p", class_="text-gray__dark_2", title=True)
        location = location_tag.get("title") if location_tag else "N/A"

        # Area fix
        area = "N/A"
        size_divs = listing.find_all("div", class_="flex items-center gap-x-x")
        for div in size_divs:
            icon = div.find("i", class_="inline-block bg-cover size-icon")
            if icon:
                area_tag = div.find("p", class_="text-gray__dark_2 text-caption")
                if area_tag:
                    area = area_tag.get_text(strip=True)
                break

        # Bedrooms and Bathrooms
        details = listing.find_all("p", class_="text-gray__dark_2 text-caption")
        bedrooms = safe_text(details[1]) if len(details) > 1 else "N/A"
        bathrooms = safe_text(details[2]) if len(details) > 2 else "N/A"

        image_tag = listing.find("img")
        image_url = image_tag['src'] if image_tag else "N/A"

        link_tag = listing.find("a", href=True)
        listing_url = "https://aqarmap.com.eg" + link_tag["href"] if link_tag else "N/A"

    except Exception:
        title = price = price_per_meter = location = area = bedrooms = bathrooms = image_url = listing_url = "N/A"

    return {
        "Title": title,
        "Price": price,
        "Price/m²": price_per_meter,
        "Location": location,
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Image URL": image_url,
        "Listing URL": listing_url
    }
