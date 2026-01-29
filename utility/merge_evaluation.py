import os
import json


def merge_evaluation_results():
    final_output = {}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    h1_key_map = {
        "Baseline": "H1_Baseline",
        "Method A": "H1_Method-A",
        "Method B": "H1_Method-B",
    }

    h2_key_map = {
        "Baseline": "H2_Baseline",
        "Method A": "H2_Method A",
        "Method B": "H2_Method B",
    }

    def extract_summary(data):
        if "summary" in data:
            return data["summary"]
        if "detailed_evaluation" in data:
            cleaned_data = data.copy()
            del cleaned_data["detailed_evaluation"]
            return cleaned_data
        return data

    h1_root = os.path.join(base_dir, "hypothesis_1", "results and evaluation")

    if os.path.exists(h1_root):
        print(f"Processing H1 directory: {h1_root}")
        for provider in os.listdir(h1_root):
            provider_path = os.path.join(h1_root, provider)
            if not os.path.isdir(provider_path):
                continue

            for method in os.listdir(provider_path):
                method_path = os.path.join(provider_path, method)
                if not os.path.isdir(method_path):
                    continue

                target_key = h1_key_map.get(method)
                if not target_key:
                    continue

                if target_key not in final_output:
                    final_output[target_key] = {}

                for file in os.listdir(method_path):
                    if file.endswith("_evaluation.json"):
                        model_key = file.replace("_evaluation.json", "")
                        file_path = os.path.join(method_path, file)

                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            final_output[target_key][model_key] = extract_summary(data)

    h2_root = os.path.join(base_dir, "hypothesis_2", "results and evaluation")

    if os.path.exists(h2_root):
        print(f"Processing H2 directory: {h2_root}")
        for provider in os.listdir(h2_root):
            provider_path = os.path.join(h2_root, provider)
            if not os.path.isdir(provider_path):
                continue

            for method in os.listdir(provider_path):
                method_path = os.path.join(provider_path, method)
                if not os.path.isdir(method_path):
                    continue

                target_key = h2_key_map.get(method)
                if not target_key:
                    continue

                if target_key not in final_output:
                    final_output[target_key] = {}

                if provider not in final_output[target_key]:
                    final_output[target_key][provider] = {}

                for file in os.listdir(method_path):
                    if file.endswith("_evaluation.json"):
                        raw_name = file.replace("_evaluation.json", "")
                        model_key = raw_name.replace("_h2", "")
                        file_path = os.path.join(method_path, file)

                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            final_output[target_key][provider][model_key] = (
                                extract_summary(data)
                            )

    output_filename = os.path.join(base_dir, "H1_H2_merged_evaluations.json")

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Process completed! Output file created: {output_filename}")


if __name__ == "__main__":
    merge_evaluation_results()
