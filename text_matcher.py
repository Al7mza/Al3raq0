import re
from typing import List, Dict, Optional

# Try to import fuzzywuzzy, fallback to rapidfuzz or built-in
try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    try:
        from rapidfuzz import fuzz
        FUZZY_AVAILABLE = True
    except ImportError:
        FUZZY_AVAILABLE = False
        print("Warning: fuzzywuzzy and rapidfuzz not available, using built-in text matching")

class TextMatcher:
    def __init__(self):
        """Initialize text matcher"""
        pass
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using multiple algorithms
        Returns a weighted average of different similarity metrics
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        if FUZZY_AVAILABLE:
            # Use fuzzy matching if available
            ratio = fuzz.ratio(norm_text1, norm_text2)
            partial_ratio = fuzz.partial_ratio(norm_text1, norm_text2)
            token_sort_ratio = fuzz.token_sort_ratio(norm_text1, norm_text2)
            token_set_ratio = fuzz.token_set_ratio(norm_text1, norm_text2)
            
            # Weighted average
            weights = [0.3, 0.2, 0.25, 0.25]
            similarities = [ratio, partial_ratio, token_sort_ratio, token_set_ratio]
            weighted_similarity = sum(w * s for w, s in zip(weights, similarities))
        else:
            # Fallback to built-in similarity
            weighted_similarity = self.built_in_similarity(norm_text1, norm_text2)
        
        return weighted_similarity
    
    def built_in_similarity(self, text1: str, text2: str) -> float:
        """Built-in similarity calculation using difflib and other methods"""
        import difflib
        
        # Sequence matcher similarity
        seq_similarity = difflib.SequenceMatcher(None, text1, text2).ratio() * 100
        
        # Keyword overlap similarity
        keyword_sim = self.keyword_based_similarity(text1, text2)
        
        # Combined similarity
        combined = (seq_similarity * 0.7) + (keyword_sim * 0.3)
        
        return combined
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep Chinese characters
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def find_best_match(self, query_text: str, reference_texts: List[Dict], threshold: float = 15.0) -> Optional[Dict]:
        """
        Find the single best match above the threshold
        Returns the best match or None if no match above threshold
        """
        if not query_text or not reference_texts:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for ref in reference_texts:
            similarity = self.calculate_similarity(query_text, ref['text'])
            
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'id': ref['id'],
                    'similarity': similarity,
                    'text': ref['text']
                }
        
        return best_match
    
    def find_all_matches(self, query_text: str, reference_texts: List[Dict], threshold: float = 15.0) -> List[Dict]:
        """
        Find all matches above the threshold
        Returns list of matches sorted by similarity (highest first)
        """
        if not query_text or not reference_texts:
            return []
        
        matches = []
        
        for ref in reference_texts:
            similarity = self.calculate_similarity(query_text, ref['text'])
            
            if similarity >= threshold:
                matches.append({
                    'id': ref['id'],
                    'similarity': similarity,
                    'text': ref['text']
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for keyword-based matching"""
        if not text:
            return []
        
        # Normalize text
        normalized = self.normalize_text(text)
        
        # Split into words
        words = normalized.split()
        
        # Filter out short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def keyword_based_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on keyword overlap"""
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return 0.0
        
        return (intersection / union) * 100
    
    def combined_similarity(self, text1: str, text2: str, fuzzy_weight: float = 0.7) -> float:
        """
        Calculate combined similarity using both fuzzy matching and keyword matching
        fuzzy_weight: weight for fuzzy similarity (0.0 to 1.0)
        """
        fuzzy_sim = self.calculate_similarity(text1, text2)
        keyword_sim = self.keyword_based_similarity(text1, text2)
        
        # Combine with weights
        combined = (fuzzy_weight * fuzzy_sim) + ((1 - fuzzy_weight) * keyword_sim)
        
        return combined
    
    def find_matches_with_combined_similarity(self, query_text: str, reference_texts: List[Dict], threshold: float = 15.0) -> List[Dict]:
        """
        Find matches using combined similarity approach
        """
        if not query_text or not reference_texts:
            return []
        
        matches = []
        
        for ref in reference_texts:
            similarity = self.combined_similarity(query_text, ref['text'])
            
            if similarity >= threshold:
                matches.append({
                    'id': ref['id'],
                    'similarity': similarity,
                    'text': ref['text'],
                    'fuzzy_similarity': self.calculate_similarity(query_text, ref['text']),
                    'keyword_similarity': self.keyword_based_similarity(query_text, ref['text'])
                })
        
        # Sort by combined similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def get_similarity_breakdown(self, text1: str, text2: str) -> Dict:
        """Get detailed breakdown of similarity calculations"""
        if FUZZY_AVAILABLE:
            return {
                'ratio': fuzz.ratio(text1, text2),
                'partial_ratio': fuzz.partial_ratio(text1, text2),
                'token_sort_ratio': fuzz.token_sort_ratio(text1, text2),
                'token_set_ratio': fuzz.token_set_ratio(text1, text2),
                'keyword_similarity': self.keyword_based_similarity(text1, text2),
                'combined_similarity': self.combined_similarity(text1, text2)
            }
        else:
            return {
                'built_in_similarity': self.built_in_similarity(text1, text2),
                'keyword_similarity': self.keyword_based_similarity(text1, text2),
                'combined_similarity': self.combined_similarity(text1, text2)
            }