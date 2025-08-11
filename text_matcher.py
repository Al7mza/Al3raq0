from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

class TextMatcher:
    def __init__(self):
        self.matcher = fuzz
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using multiple algorithms"""
        if not text1 or not text2:
            return 0
        
        # Normalize texts
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        # Calculate different similarity metrics
        ratio = fuzz.ratio(norm_text1, norm_text2)
        partial_ratio = fuzz.partial_ratio(norm_text1, norm_text2)
        token_sort_ratio = fuzz.token_sort_ratio(norm_text1, norm_text2)
        token_set_ratio = fuzz.token_set_ratio(norm_text1, norm_text2)
        
        # Weighted average of different metrics
        # Give more weight to partial ratio for better matching of partial text
        weighted_similarity = (
            ratio * 0.2 +
            partial_ratio * 0.4 +
            token_sort_ratio * 0.2 +
            token_set_ratio * 0.2
        )
        
        return round(weighted_similarity, 2)
    
    def normalize_text(self, text):
        """Normalize text for better comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters (keep Chinese characters)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def find_best_match(self, query_text, reference_texts, threshold=15):
        """Find the best matching reference text above the threshold"""
        if not query_text or not reference_texts:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for ref_id, ref_text in reference_texts:
            similarity = self.calculate_similarity(query_text, ref_text)
            
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = ref_id
        
        return best_match, best_similarity
    
    def find_all_matches(self, query_text, reference_texts, threshold=15):
        """Find all reference texts above the threshold, sorted by similarity"""
        if not query_text or not reference_texts:
            return []
        
        matches = []
        
        for ref_id, ref_text in reference_texts:
            similarity = self.calculate_similarity(query_text, ref_text)
            
            if similarity >= threshold:
                matches.append({
                    'id': ref_id,
                    'text': ref_text,
                    'similarity': similarity
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def extract_keywords(self, text):
        """Extract potential keywords from text for better matching"""
        if not text:
            return []
        
        # Normalize text
        norm_text = self.normalize_text(text)
        
        # Split into words
        words = norm_text.split()
        
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def keyword_based_similarity(self, text1, text2):
        """Calculate similarity based on keyword overlap"""
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        if union:
            return len(intersection) / len(union) * 100
        
        return 0
    
    def combined_similarity(self, text1, text2):
        """Calculate combined similarity using multiple methods"""
        fuzzy_similarity = self.calculate_similarity(text1, text2)
        keyword_similarity = self.keyword_based_similarity(text1, text2)
        
        # Combine both similarities with weights
        combined = (fuzzy_similarity * 0.7) + (keyword_similarity * 0.3)
        
        return round(combined, 2)