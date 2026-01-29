import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

class ErrorAnalyzer:
    """Qualitative Error Analysis for Food Label Analysis System (H1 & H2)"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.h1_errors = []
        self.h2_errors = []
    
    def parse_h1_errors(self) -> None:
        """Parse H1 evaluation files for ingredient extraction errors"""
        h1_path = Path(self.base_dir) / "hypothesis_1" / "results and evaluation"
        
        if not h1_path.exists():
            print(f"WARNING: H1 directory not found at {h1_path}")
            return
        
        for json_file in h1_path.rglob("*_h1_evaluation.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model_name = json_file.stem.replace("_h1_evaluation", "")
                condition = json_file.parent.name
                provider = json_file.parent.parent.name
                
                detailed_evals = data.get("detailed_evaluations", [])
                
                for eval_item in detailed_evals:
                    product_id = eval_item.get("product_id", "UNKNOWN")
                    match_details = eval_item.get("ingredients", {}).get("match_details", [])
                    
                    for match in match_details:
                        match_type = match.get("match", "")
                        
                        if match_type == "FP":
                            pred_text = match.get("pred", "N/A")
                            score = match.get("score", None)
                            self.h1_errors.append({
                                "Model": model_name,
                                "Provider": provider,
                                "Condition": condition,
                                "Product_ID": product_id,
                                "Error_Type": "Hallucination",
                                "Text_Content": pred_text,
                                "Confidence_Score": score
                            })
                        
                        elif match_type == "FN":
                            gt_text = match.get("gt", "N/A")
                            self.h1_errors.append({
                                "Model": model_name,
                                "Provider": provider,
                                "Condition": condition,
                                "Product_ID": product_id,
                                "Error_Type": "Omission",
                                "Text_Content": gt_text,
                                "Confidence_Score": None
                            })
                
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"ERROR parsing {json_file}: {e}")
            except Exception as e:
                print(f"UNEXPECTED ERROR in {json_file}: {e}")
    
    def parse_h2_errors(self) -> None:
        """Parse H2 evaluation files for risk assessment errors
        
        H2 structure: detailed_evaluations is a DICT where keys are profile names,
        and values are LISTS of product evaluations.
        """
        h2_path = Path(self.base_dir) / "hypothesis_2" / "results and evaluation"
        
        if not h2_path.exists():
            print(f"WARNING: H2 directory not found at {h2_path}")
            return
        
        for json_file in h2_path.rglob("*_h2_evaluation.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model_name = json_file.stem.replace("_h2_evaluation", "")
                condition = json_file.parent.name
                provider = json_file.parent.parent.name
                
                detailed_evals = data.get("detailed_evaluations", {})
                
                # detailed_evals is a dictionary: {profile_name: [list of evaluations]}
                if isinstance(detailed_evals, dict):
                    for profile_name, eval_list in detailed_evals.items():
                        # eval_list is a list of product evaluations
                        if isinstance(eval_list, list):
                            for eval_item in eval_list:
                                self._extract_h2_error(eval_item, model_name, provider, 
                                                     condition, profile_name)
                        else:
                            print(f"  WARNING: Expected list for profile {profile_name} in {json_file}")
                
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"ERROR parsing {json_file}: {e}")
            except Exception as e:
                print(f"UNEXPECTED ERROR in {json_file}: {e}")
    
    def _extract_h2_error(self, eval_item: dict, model_name: str, provider: str, 
                         condition: str, profile_name: str) -> None:
        """Extract H2 errors from a single product evaluation"""
        try:
            product_id = eval_item.get("product_id", "UNKNOWN")
            risk_detection = eval_item.get("risk_detection", {})
            
            missed_high_risks = risk_detection.get("missed_high_risks", [])
            total_high_assigned = risk_detection.get("total_high_assigned", 0)
            true_positives = risk_detection.get("true_positives", 0)
            
            # Critical Missed Risks
            for missed_risk in missed_high_risks:
                self.h2_errors.append({
                    "Model": model_name,
                    "Provider": provider,
                    "Condition": condition,
                    "Product_ID": product_id,
                    "Profile_Name": profile_name,
                    "Error_Type": "Critical Missed Risk",
                    "Detail": missed_risk
                })
            
            # False Alarms
            if total_high_assigned > true_positives:
                false_alarms = total_high_assigned - true_positives
                self.h2_errors.append({
                    "Model": model_name,
                    "Provider": provider,
                    "Condition": condition,
                    "Product_ID": product_id,
                    "Profile_Name": profile_name,
                    "Error_Type": "False Alarm",
                    "Detail": f"{false_alarms} items incorrectly flagged as high risk"
                })
        except Exception as e:
            print(f"  ERROR extracting H2 data: {e}")
    
    def export_h1_csv(self, output_file: str = "H1_Error_Analysis.csv") -> None:
        """Export H1 errors to CSV"""
        if not self.h1_errors:
            print("No H1 errors found to export.")
            return
        
        df = pd.DataFrame(self.h1_errors)
        df = df[["Model", "Provider", "Condition", "Product_ID", "Error_Type", "Text_Content", "Confidence_Score"]]
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ H1 Error Analysis exported to {output_file}")
        print(f"  Total errors: {len(df)}")
        print(f"  Breakdown:\n{df['Error_Type'].value_counts()}\n")
    
    def export_h2_csv(self, output_file: str = "H2_Error_Analysis.csv") -> None:
        """Export H2 errors to CSV"""
        if not self.h2_errors:
            print("No H2 errors found to export.")
            return
        
        df = pd.DataFrame(self.h2_errors)
        df = df[["Model", "Provider", "Condition", "Product_ID", "Profile_Name", "Error_Type", "Detail"]]
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ H2 Error Analysis exported to {output_file}")
        print(f"  Total errors: {len(df)}")
        print(f"  Breakdown:\n{df['Error_Type'].value_counts()}\n")
    
    def print_summary(self) -> None:
        """Print a summary of errors by condition and model"""
        print("\n" + "="*80)
        print("H1 ERROR ANALYSIS SUMMARY")
        print("="*80)
        if self.h1_errors:
            df_h1 = pd.DataFrame(self.h1_errors)
            print(f"Total H1 Errors: {len(df_h1)}")
            print("\nErrors by Condition:")
            print(df_h1.groupby("Condition")["Error_Type"].value_counts())
            print("\nErrors by Model (Top 10):")
            print(df_h1.groupby("Model")["Error_Type"].value_counts().head(10))
        else:
            print("No H1 errors found.")
        
        print("\n" + "="*80)
        print("H2 ERROR ANALYSIS SUMMARY")
        print("="*80)
        if self.h2_errors:
            df_h2 = pd.DataFrame(self.h2_errors)
            print(f"Total H2 Errors: {len(df_h2)}")
            print("\nErrors by Condition:")
            print(df_h2.groupby("Condition")["Error_Type"].value_counts())
            print("\nErrors by Model (Top 10):")
            print(df_h2.groupby("Model")["Error_Type"].value_counts().head(10))
            print("\nErrors by Profile:")
            print(df_h2.groupby("Profile_Name")["Error_Type"].value_counts())
        else:
            print("No H2 errors found.")
        print("="*80 + "\n")
    
    def run(self, base_dir: str = ".", h1_output: str = "H1_Error_Analysis.csv", 
            h2_output: str = "H2_Error_Analysis.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full analysis pipeline"""
        print("Starting Qualitative Error Analysis...")
        print(f"Base directory: {base_dir}\n")
        
        self.parse_h1_errors()
        self.parse_h2_errors()
        
        self.print_summary()
        
        self.export_h1_csv(h1_output)
        self.export_h2_csv(h2_output)
        
        df_h1 = pd.DataFrame(self.h1_errors) if self.h1_errors else pd.DataFrame()
        df_h2 = pd.DataFrame(self.h2_errors) if self.h2_errors else pd.DataFrame()
        
        return df_h1, df_h2


if __name__ == "__main__":
    analyzer = ErrorAnalyzer(base_dir=".")
    df_h1, df_h2 = analyzer.run(
        base_dir=".",
        h1_output="statistical_test/H1_Error_Analysis.csv",
        h2_output="statistical_test/H2_Error_Analysis.csv"
    )
    
    print("\n✓ Analysis Complete!")
    print("Generated files:")
    print("  - H1_Error_Analysis.csv")
    print("  - H2_Error_Analysis.csv")