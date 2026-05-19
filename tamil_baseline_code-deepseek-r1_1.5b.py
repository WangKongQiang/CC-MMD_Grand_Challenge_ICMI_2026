from email.mime import image
import os
import ollama
import base64
import glob
import json
from dotenv import load_dotenv
from tqdm import tqdm
import time
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# 很好，这个需求很清晰：👉 从“图像 + prompt”改成“transcriptions（文本）+ prompt”做分类，任务保持不变（misogyny detection）。
#
# 我帮你做了最小侵入式修改，核心改动：
# ✅ 关键修改点
# ❌ 删除
# encode_image
# images=[base64_image]
# 所有 image 相关逻辑
# ✅ 新增
# 从 CSV 读取 transcriptions
# 用文本直接喂给 deepseek-r1-1.5b

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


def parse_model_response(result, image_path):
    """Parse model JSON response and fallback to heuristic classification."""
    try:
        if result is not None:
            if "```json" in result:
                json_start = result.find("```json") + 7
                json_end = result.find("```", json_start)
                json_str = result[json_start:json_end].strip()
            elif "```" in result:
                json_start = result.find("```") + 3
                json_end = result.find("```", json_start)
                json_str = result[json_start:json_end].strip()
            else:
                json_start = result.find("{")
                json_end = result.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result[json_start:json_end].strip()
                else:
                    json_str = result
        else:
            json_str = ""

        parsed_result = json.loads(json_str)
        classification = parsed_result.get("classification", "").lower()
        explanation = parsed_result.get("explanation", "")

        if "misogyny" in classification and not "not-misogyny" in classification:
            return "misogyny", explanation

        return "not-misogyny", explanation

    except Exception as json_error:
        print(f"JSON parsing failed for {image_path}: {str(json_error)}")
        print(f"Raw response: {result}")

        if (
            result is not None
            and "misogyny" in result.lower()
            and not ("not-misogyny" in result.lower() or "not misogyny" in result.lower())
        ):
            return "misogyny", result

        return "not-misogyny", (result if result is not None else "No response from model.")

# ✅ 修改后的核心代码（可直接用）
# 🔥 新函数：文本分类（替代 classify_image）

# def classify_image(image_path, model, country=None, language=None):
#     """
#     Classify a meme image as misogynistic or not using zero-shot prompting (Ollama Vision Model).
#
#     Args:
#         image_path: Path to the image file
#         model: The Ollama model name
#
#     Returns:
#         dict: Classification result with label and explanation
#     """
#     image_name = os.path.basename(image_path)
#     image_id = os.path.splitext(image_name)[0]
#
#     prompt = f"""
#     You will be provided with a {language} meme image. Your task is to analyze the meme for potential misogynistic content by examining both its visual and textual components. The meme is in {language} language and should be interpreted within the cultural, social, and linguistic context of {country}. Follow the steps below:
#
#     Step-by-Step Instructions:
#     - Extract and interpret all visible text from the meme with information such as caption text, top Text / bottom text, dialogues/speech, hashtags/phrases, labels, narrative text, emoji/symbols. The visible text may contain transliterations or code-mixed texts.
#     - Analyze the visual elements such as layout, characters, expressions, image manipulations, appearance, symbols, background, with the extracted informative texts.
#     - Evaluate the given meme according to the socio-cultural context of {country} by analysing the extracted text and visual analysis from the previous steps using the criteria below.
#
#     Classification Criteria:
#     - Classify the meme as “misogyny” if the content that directly or indirectly targets women or girls through stereotyping, hatred, discrimination, sexualization, objectification, gender-based violence, Insulting, disrespecting, devaluing, prejudice, Victim-blaming, Trivializing, Belittling, or mockery, even if it is in the form of sarcasm or humor.
#     - Classify the meme as "not-misogyny" if none of the above criteria apply.
#
#     Important Notes:
#     - Be objective, culturally aware and precise.
#     - Do not interpret humor or irony as neutral if it carries misogynistic meaning.
#
#     Respond only in the following JSON format with no additional text or commentary:
#     {{
#       "explanation": "Clearly explain your reasoning, referencing specific text (with translation if needed) and visual details. Include cultural context relevant to {country} where applicable.",
#       "classification": "misogyny" or "not-misogyny"
#     }}
#     """
#
#     # Encode image
#     base64_image = encode_image(image_path)
#     model = "gemma:2b"
#
#     try:
#         # Use ollama.chat() - the correct API call
#         response = ollama.chat(
#             model=model,
#             messages=[{"role": "user", "content": prompt, "images": [base64_image]}],
#             options={
#                 "temperature": 0.0,
#                 "num_predict": 2048,
#                 "num_thread": 12,
#                 "num_gpu": -1,
#             },
#         )
#         result = response["message"]["content"]
#     except Exception as e:
#         return {
#             "image_id": image_id,
#             "image": image_path,
#             "classification": "Error",
#             "explanation": f"Ollama API error: {str(e)}",
#             "full_response": f"Error: {str(e)}",
#         }
#
#     classification, explanation = parse_model_response(result, image_path)
#
#     return {
#         "image_id": image_id,
#         "image": image_path,
#         "classification": classification,
#         "explanation": explanation,
#         "full_response": result,
#     }

def classify_text(text, sample_id, model, country=None, language=None):
    """
    Classify meme transcription text as misogyny or not.
    """
    # model = "mistral:latest"
    # model = "qwen2.5:7b"
    model = "deepseek-r1:1.5b"
    
    prompt = f"""
    You will be provided with a {language} meme transcription. Your task is to analyze it for potential misogynistic content. The meme is in {language} and should be interpreted within the cultural, social, and linguistic context of {country}.

    Step-by-Step Instructions:
    - Carefully read and interpret the given text. It may contain slang, transliteration, sarcasm, or implicit meaning.
    - Analyze whether the text targets women or girls.

    Classification Criteria:
    - Classify as “misogyny” if it includes stereotyping, hatred, discrimination, sexualization, objectification, insults, or mockery toward women.
    - Otherwise classify as "not-misogyny".

    Respond ONLY in JSON:
    {{
      "explanation": "Explain reasoning with reference to text and cultural context",
      "classification": "misogyny" or "not-misogyny"
    }}

    Text:
    {text}
    """

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_predict": 1024,
            },
        )

        result = response["message"]["content"]

    except Exception as e:
        return {
            "id": sample_id,
            "text": text,
            "classification": "Error",
            "explanation": f"Ollama API error: {str(e)}",
            "full_response": str(e),
        }

    classification, explanation = parse_model_response(result, sample_id)

    return {
        "id": sample_id,
        "text": text,
        "classification": classification,
        "explanation": explanation,
        "full_response": result,
    }

# def parse_model_response(result, image_path):
#     """Parse model JSON response and fallback to heuristic classification."""
#     try:
#         if result is not None:
#             if "```json" in result:
#                 json_start = result.find("```json") + 7
#                 json_end = result.find("```", json_start)
#                 json_str = result[json_start:json_end].strip()
#             elif "```" in result:
#                 json_start = result.find("```") + 3
#                 json_end = result.find("```", json_start)
#                 json_str = result[json_start:json_end].strip()
#             else:
#                 json_start = result.find("{")
#                 json_end = result.rfind("}") + 1
#                 if json_start >= 0 and json_end > json_start:
#                     json_str = result[json_start:json_end].strip()
#                 else:
#                     json_str = result
#         else:
#             json_str = ""
#
#         parsed_result = json.loads(json_str)
#         classification = parsed_result.get("classification", "").lower()
#         explanation = parsed_result.get("explanation", "")
#
#         if "misogyny" in classification and not "not-misogyny" in classification:
#             return "misogyny", explanation
#
#         return "not-misogyny", explanation
#
#     except Exception as json_error:
#         print(f"JSON parsing failed for {image_path}: {str(json_error)}")
#         print(f"Raw response: {result}")
#
#         if (
#             result is not None
#             and "misogyny" in result.lower()
#             and not ("not-misogyny" in result.lower() or "not misogyny" in result.lower())
#         ):
#             return "misogyny", result
#
#         return "not-misogyny", (result if result is not None else "No response from model.")

# ✅ 批处理函数（改为读取 CSV）
#
# 假设你的 CSV 有：
#
# id,transcription
# 1,Some meme text
# 2,Another meme text

# 🔥 新 batch 函数：

# def batch_classify_images(
#     image_directory, model, output_base="results", country=None, language=None
# ):
#     """
#     Classify all meme images in a directory and save results to both JSON and TXT files.
#
#     Args:
#         image_directory: Directory containing images
#         model: The Ollama model name
#         output_base: Base name for output files (without extension)
#     """
#     # Get all image files
#     image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
#     image_files = []
#     for ext in image_extensions:
#         image_files.extend(glob.glob(os.path.join(image_directory, ext)))
#
#     results = []
#
#     # Create output filenames
#     json_output = f"{output_base}.json"
#     txt_output = f"{output_base}.txt"
#
#     # Clear the TXT output file before starting
#     with open(txt_output, "w") as f:
#         f.write("Misogyny Meme Classification Results\n")
#         f.write("=" * 50 + "\n\n")
#
#     # Process each image with progress bar
#     for img_path in tqdm(
#         image_files, desc="Classifying Memes", unit="meme", colour="green"
#     ):
#         result = classify_image(img_path, model, country=country, language=language)
#         results.append(result)
#
#         # Append to TXT file immediately
#         with open(txt_output, "a") as f:
#             f.write(f"Image ID: {result['image_id']}\n")
#             f.write(f"Image: {result['image']}\n")
#             f.write(f"Classification: {result['classification']}\n")
#             f.write(f"Explanation: {result['explanation']}\n")
#             f.write("-" * 50 + "\n\n")
#
#         # Add delay to avoid rate limiting
#         time.sleep(1)
#
#     # Save all results to JSON file at the end
#     with open(json_output, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#
#     print(f"\nResults saved to:")
#     print(f"JSON: {json_output}")
#     print(f"TXT: {txt_output}")
#
#     # Print summary
#     total = len(results)
#     misogyny = sum(1 for r in results if r["classification"] == "misogyny")
#     not_misogyny = sum(1 for r in results if r["classification"] == "not-misogyny")
#     errors = sum(1 for r in results if r["classification"] == "Error")
#
#     # Save error IDs to JSON
#     error_responses = [r for r in results if r["classification"] == "Error"]
#     if error_responses:
#         error_output = f"{json_output.replace('.json', '')}_errors.json"
#         print(error_output)
#         with open(error_output, "w", encoding="utf-8") as f:
#             json.dump(error_responses, f, indent=2, ensure_ascii=False)
#         print(f"Error responses saved to: {error_output}")
#
#     print(f"\nSummary:")
#     print(f"Total images: {total}")
#     print(f"Misogyny: {misogyny}")
#     print(f"Not-misogyny: {not_misogyny}")
#     print(f"Errors: {errors}")
#
#     return results

def batch_classify_texts(
    csv_path, model, output_base="results", country=None, language=None
):
    df = pd.read_csv(csv_path)

    results = []

    json_output = f"{output_base}.json"
    txt_output = f"{output_base}.txt"

    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("Misogyny Text Classification Results\n")
        f.write("=" * 50 + "\n\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying Texts"):
        sample_id = row["image_id"] if "image_id" in df.columns else _
        text = row["transcriptions"]

        result = classify_text(
            text=text,
            sample_id=sample_id,
            model=model,
            country=country,
            language=language,
        )

        results.append(result)

        with open(txt_output, "a", encoding="utf-8") as f:
            f.write(f"ID: {result['id']}\n")
            f.write(f"Text: {result['text']}\n")
            f.write(f"Classification: {result['classification']}\n")
            f.write(f"Explanation: {result['explanation']}\n")
            f.write("-" * 50 + "\n\n")

        time.sleep(0.5)

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {json_output} and {txt_output}")

    return results

# ✅ main函数修改

# if __name__ == "__main__":
#     # model_name = "gemma3-4b"
#     model_name = "gemma-2b"
#     image_directory = "MDMD"  # folder with meme images
#     language = "tamil"  # set according to your dataset language
#     country = "India"  # change the target culture as needed: "India", "China", "Ireland"
#
#     output_base = f"results/{model_name}/geo_instruction/{image_directory}"
#
#     print(f"Running classification with model={model_name}, country={country}, language={language}")
#     print(f"Image directory: {image_directory}")
#     print(f"Output base: {output_base}")
#
#     os.makedirs(os.path.dirname(output_base), exist_ok=True)
#
#     batch_classify_images(
#         image_directory=image_directory,
#         model=model_name,
#         output_base=output_base,
#         country=country,
#         language=language,
#     )
#
#     print("Batch classification complete.")

if __name__ == "__main__":
    # model_name = "mistral-latest"
    # model_name = "qwen2.5-7b"
    model_name = "deepseek-r1-1.5b"

    csv_path = "MDMD/test.csv"   # ← 你的transcription文件
    language = "tamil"
    country = "India"

    output_base = f"results/{model_name}/text_instruction/test"

    os.makedirs(os.path.dirname(output_base), exist_ok=True)

    batch_classify_texts(
        csv_path=csv_path,
        model=model_name,
        output_base=output_base,
        country=country,
        language=language,
    )

    print("Done.")

###运行脚本
# python tamil_baseline_code-deepseek-r1_1.5b.py