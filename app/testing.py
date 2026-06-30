"""
Testing and validation utilities
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
from fuzzywuzzy import fuzz


class Tester:
    """Handles testing and comparison with expected results"""
    
    def __init__(self, expected_dir: Path, output_dir: Path, min_score: int = 80):
        """
        Initialize tester
        
        Args:
            expected_dir: Directory containing expected results
            output_dir: Directory to save test results
            min_score: Minimum similarity score for passing
        """
        self.expected_dir: Path = expected_dir
        self.output_dir: Path = output_dir
        self.min_score: int = min_score
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_expected_data(self, image_name: str) -> Dict[str, str]:
        """
        Load expected data for an image
        
        Args:
            image_name: Name of the image (without extension)
            
        Returns:
            Dictionary of expected field values
        """
        json_path: Path = self.expected_dir / f"{image_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Expected results file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
            return {k: str(v) for k, v in data.items()}
    
    def compare_results(
        self, 
        actual: Dict[str, str], 
        expected: Dict[str, str]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Compare actual results with expected using fuzzy matching
        
        Args:
            actual: Actual extracted data
            expected: Expected data
            
        Returns:
            Tuple of (success, errors, details)
        """
        errors: List[str] = []
        details: Dict[str, Any] = {}
        
        for key in expected:
            if key not in actual:
                errors.append(f"Missing field: {key}")
                details[key] = {'status': 'missing'}
                continue
            
            exp: str = expected[key] or ""
            act: str = actual.get(key) or ""
            
            # If both are empty - ok
            if not exp and not act:
                details[key] = {'status': 'match', 'score': 100}
                continue
            
            # Calculate similarity using different methods
            score_ratio: int = fuzz.ratio(act, exp)
            score_partial: int = fuzz.partial_ratio(act, exp)
            score_token: int = fuzz.token_sort_ratio(act, exp)
            
            # Take maximum
            score: int = max(score_ratio, score_partial, score_token)
            
            details[key] = {
                'status': 'match' if score >= self.min_score else 'mismatch',
                'score': score,
                'expected': exp,
                'actual': act
            }
            
            if score < self.min_score:
                errors.append(
                    f"{key}: expected '{exp}', got '{act}' (similarity: {score}%)"
                )
        
        return len(errors) == 0, errors, details
    
    def run_tests(self, image_name: str, actual_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Run tests for an image
        
        Args:
            image_name: Name of the image
            actual_data: Extracted data
            
        Returns:
            Test results dictionary
        """
        try:
            expected_data: Dict[str, str] = self.load_expected_data(image_name)
            success: bool
            errors: List[str]
            details: Dict[str, Any]
            success, errors, details = self.compare_results(actual_data, expected_data)
            
            print(f"\n{'✅' if success else '❌'} Test {image_name}: {'PASSED' if success else 'FAILED'}")
            
            if errors:
                print("   ERRORS:")
                for error in errors:
                    print(f"   - {error}")
            
            # Statistics
            total: int = len(details)
            matches: int = sum(1 for d in details.values() if d.get('status') == 'match')
            mismatches: int = sum(1 for d in details.values() if d.get('status') == 'mismatch')
            
            print("\n   STATISTICS:")
            print(f"   - Total fields: {total}")
            print(f"   - Matches: {matches}")
            print(f"   - Mismatches: {mismatches}")
            
            if matches > 0:
                avg_score: float = sum(
                    d.get('score', 0) for d in details.values() 
                    if d.get('status') == 'match'
                ) / matches
                print(f"   - Average similarity: {avg_score:.1f}%")
            
            # Save result
            result: Dict[str, Any] = {
                "image": image_name,
                "success": success,
                "min_score": self.min_score,
                "errors": errors,
                "details": details,
                "stats": {
                    "total": total,
                    "matches": matches,
                    "mismatches": mismatches
                }
            }
            
            output_path: Path = self.output_dir / f"{image_name}_result.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\nTest result saved: {output_path}")
            
            return result
        
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {
                "image": image_name,
                "success": False,
                "error": str(e)
            }