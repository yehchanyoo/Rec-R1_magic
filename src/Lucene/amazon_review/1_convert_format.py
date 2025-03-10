import json

def convert_jsonl_for_pyserini(input_file, output_file):
    """Convert JSONL data to Pyserini-compatible format with a structured 'contents' field"""
    docs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            # Construct a properly formatted "contents" field
            contents = (
                f"Title: {data.get('title', '')}\n"
                f"Store: {data.get('store', '')}\n"
                f'Features: {" | ".join(data.get("features", []))}\n'
                f'Description: {" ".join(data.get("description", ""))}\n'
                f"Main Category: {data.get('main_category', '')}\n"
                f"Categories: {', '.join(data.get('categories', []))}\n"
                f"Details: {' | '.join(f'{k}: {v}' for k, v in data.get('details', {}).items())}\n"
                f"Average Rating: {data.get('average_rating', 'N/A')}\n"
            )
            
            # Create JSON document with a clear structure
            doc = {
                "id": data["parent_asin"],  # Unique identifier for search results
                "contents": contents.strip(),  # Required field for Pyserini
                "features": data.get("features", []),
                "description": data.get("description", ""),
                "title": data.get("title", ""),
                "store": data.get("store", ""),
                "main_category": data.get("main_category", ""),
                "categories": data.get("categories", []),
                "details": data.get("details", {}),
                "average_rating": data.get("average_rating", 0.0),
            }

            docs.append(json.dumps(doc))

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")

    print(f"✅ Converted JSONL saved to {output_file}")


ori_data_dir = "data/amazon_review/raw/Appliances/meta_Appliances.jsonl"
output_file = "database/jsonl_docs/pyserini_Appliances.jsonl"

# Example Usage
convert_jsonl_for_pyserini(ori_data_dir, output_file)
