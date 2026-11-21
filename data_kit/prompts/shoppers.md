You create fictional shopper profiles for testing an Indian online shop. Every person is invented: do not use real or famous people, and do not invent phone numbers, emails or addresses.

You will receive a JSON list of shopper skeletons. Each has FIXED facts: customer_id, city, preferred_language, budget range in INR, interests, and the list of brands available for their interests. Do not change these facts.

For EACH skeleton write:
- "display_name": a plausible fictional Indian first name and last-name initial, e.g. "Asha K." (vary regions and genders).
- "fav_brands": 1 to 3 brands chosen ONLY from that skeleton's "available_brands" list.
- "sizes": an object with only the sizes that fit their interests, using "shoe" (Indian/UK size 5 to 11 as a string) and/or "clothing" (one of "S", "M", "L", "XL", "XXL"); use {} if none apply.

Return ONLY a JSON object of this exact shape, one entry per skeleton with the same customer_id:
{"shoppers": [{"customer_id": "...", "display_name": "...", "fav_brands": ["..."], "sizes": {"shoe": "9"}}]}

Shopper skeletons:
{skeletons_json}
