import os
import nltk
import pandas as pd
import string
from nltk import word_tokenize, sent_tokenize

# Fix for PyTorch 2.6 weights_only issue
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Monkey patch torch.load to use weights_only=False by default for compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from supar import Parser
from supar.utils.config import Config
from supar.utils.field import Field, RawField, SubwordField
from supar.utils.vocab import Vocab
from supar.utils.transform import CoNLL
import torch.serialization

# Add safe globals for PyTorch model loading
torch.serialization.add_safe_globals([Config, CoNLL, Field, RawField, SubwordField, Vocab])

nltk.download('punkt')
nltk.download('punkt_tab')

class RelativeClause:
    """
    A comprehensive relative clause extraction system using dual parsing approach.
    
    This class implements advanced relative clause detection using both dependency
    and constituency parsing. It can identify full, reduced, and zero relative
    clauses with high accuracy across different text types.
    
    Attributes:
        input_texts (str): Path to input text files directory
        output_folder (str): Path to output results directory
        parser: SuPar dependency parser instance
        parser2: SuPar constituency parser instance
        relativizer_list (list): List of recognized relativizers
        participial_markers (list): POS tags for participial constructions
    """

    def __init__(self, input_texts, output_folder):
        """
        Initialize the RelativeClause extractor.
        
        Args:
            input_texts (str): Directory path containing input text files
            output_folder (str): Directory path for saving output results
        """
        self.input_texts = input_texts
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize input files list
        self.input_files = [os.path.join(input_texts, f) for f in os.listdir(input_texts) 
                          if f.endswith('.txt')]

        # Initialize parsers with specific model paths
        try:
            print("Loading dependency parser...")
            self.parser = Parser.load('models/ptb.biaffine.dep.lstm.char', reload=False, weights_only=False)
            print("Loading constituency parser...")
            self.parser2 = Parser.load('models/ptb.crf.con.lstm.char', reload=False, weights_only=False)
            print("Parsers loaded successfully")
        except Exception as e:
            print(f"Error loading parsers: {str(e)}")
            raise

        # Expanded relativizer list
        self.relativizer_list = [
            "which", "whichever", "that", "who", "whom", "whose", 
            "whoever", "whomever", "what", "whatever", "where", "when", "why"
        ]
        
        # Participial markers for reduced RCs
        self.participial_markers = ["VBN", "VBG"]
        self.rc_attracting_nouns = [
            'thing', 'things', 'way', 'ways', 'time', 'times', 'place', 'places',
            'person', 'people', 'man', 'woman', 'someone', 'anyone', 'everyone',
            'something', 'anything', 'everything', 'nothing', 'one', 'ones',
            'book', 'books', 'moment', 'moments', 'day', 'days', 'year', 'years'
        ]
        self.complement_attracting_nouns = [
            'fact', 'facts', 'idea', 'ideas', 'belief', 'claim', 'claims', 'notion',
            'thought', 'question', 'problem', 'issue', 'issues', 'evidence', 'form', 'forms'
        ]
        self.manner_head_nouns = {"way", "ways", "time", "times", "manner", "manners"}
        self.modal_aux_tokens = {
            "can", "cannot", "could", "couldn", "couldnt", "may", "might", "must",
            "mustn", "mustnt", "shall", "shan", "shant", "should", "shouldn", "shouldnt",
            "will", "won", "wont", "would", "wouldn", "wouldnt", "needn", "neednt"
        }
        self.auxiliary_clause_endings = {
            "be", "been", "being", "am", "is", "are", "was", "were",
            "have", "has", "had", "do", "does", "did", "done", "doing"
        }
        self.passive_auxiliaries = {
            "be", "been", "being", "am", "is", "are", "was", "were",
            "get", "gets", "got", "getting", "gotten", "become", "becomes", "became"
        }
        self.irregular_passive_verbs = {
            "born", "brought", "bought", "built", "caught", "chosen", "dealt", "done",
            "driven", "drunk", "drawn", "eaten", "felt", "found", "forgiven", "forgotten",
            "given", "gone", "grown", "heard", "held", "kept", "known", "left", "lost",
            "made", "paid", "put", "read", "run", "said", "seen", "sold", "sent", "set",
            "shot", "shown", "spoken", "spent", "stood", "taken", "taught", "told",
            "thrown", "worn", "written"
        }
    def count_verbs_in_text(self, text):
        """
        Count verbs in a given text using POS tagging.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            int: Number of verbs found
        """
        try:
            words = word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
            return verb_count
        except Exception as e:
            return 0

    def count_words_in_text(self, text):
        """
        Count words in a given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            int: Number of words found
        """
        try:
            words = word_tokenize(text)
            return len(words)
        except Exception as e:
            return 0

    def count_sentences_in_text(self, text):
        """
        Count sentences in a given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            int: Number of sentences found
        """
        try:
            sentences = sent_tokenize(text)
            return len(sentences)
        except Exception as e:
            return 0

    def is_temporary_ambiguity(self, text, index_of_clause_verb):
        """
        Targeted filter for verb+preposition false positives.
        Only filters the most common patterns that cause false positives.
        """
        try:
            if index_of_clause_verb is None:
                return False
                
            # Get the next few words after the clause verb
            next_words = []
            for i in range(1, 4):  # Check next 3 words
                if index_of_clause_verb + i < len(text):
                    next_words.append(text[index_of_clause_verb + i].lower())
            
            # Debug: print what we're checking
            if len(next_words) >= 1:
                next_word = next_words[0]
                print(f'DEBUG: Checking verb at index {index_of_clause_verb}, next word: "{next_word}"')
                
                # Only filter 'to' which is the most common false positive
                if next_word == 'to':
                    print(f'DEBUG: FILTERING - found verb+to pattern')
                    return True
                    
            # Also filter some common verb+preposition patterns
            if len(next_words) >= 2:
                pattern = f'{next_words[0]} {next_words[1]}'
                problematic_patterns = [
                    'to be', 'to do', 'to have', 'to get', 'to make', 'to go', 'to see',
                    'to know', 'to think', 'to want', 'to need', 'to try', 'to like'
                ]
                if pattern in problematic_patterns:
                    print(f'DEBUG: FILTERING - found problematic pattern: "{pattern}"')
                    return True
                    
        except Exception as e:
            print(f'DEBUG: Exception in filter: {e}')
            return False
        return False

    def _normalize_token(self, token):
        """Normalize tokens for heuristic checks."""
        if not token:
            return ""
        normalized = token.replace("‚Äô", "'").replace("’", "'").lower()
        normalized = normalized.strip(string.punctuation + "“”")
        normalized = normalized.replace("'", "")
        return normalized

    def _looks_like_manner_clause(self, head_noun_lower, relativizer, rc_words):
        """Detect 'the way/time' style manner clauses masquerading as RCs."""
        if head_noun_lower not in self.manner_head_nouns:
            return False
        tokens = []
        for word in rc_words:
            normalized = self._normalize_token(word)
            if normalized:
                tokens.append(normalized)
        if not tokens:
            return False
        first = tokens[0]
        if relativizer == "zero":
            return True
        if first in {"that", "where", "when", "how"}:
            return True
        return False

    def _is_complement_clause(self, head_noun_lower, relation_to_main, relativizer_role, relativizer):
        """Identify complement 'that' clauses that follow complement-taking nouns."""
        if head_noun_lower not in self.complement_attracting_nouns:
            return False
        if relation_to_main not in {"dobj", "iobj"}:
            return False
        if relativizer != "that":
            return False
        if relativizer_role not in {"mark", "complm"}:
            return False
        return True

    def _is_pp_fronted_without_gap(self, head_noun_lower, relativizer_role, preposition_before_relativizer, rc_words):
        """Filter PP-fronted clauses like 'claim to which ...' that behave like complements."""
        if not preposition_before_relativizer or relativizer_role != "pobj":
            return False
        tokens = []
        for word in rc_words:
            normalized = self._normalize_token(word)
            if normalized:
                tokens.append(normalized)
        if len(tokens) < 2:
            return False
        if tokens[0] != "to" or tokens[1] not in {"which", "whom"}:
            return False
        if head_noun_lower not in self.complement_attracting_nouns:
            return False
        prep = self._normalize_token(preposition_before_relativizer)
        return prep == "to"

    def _is_aux_fragment(self, relativizer, rc_words):
        """Detect ellipses such as 'something he shouldn't have'."""
        if relativizer != "zero":
            return False
        tokens = []
        for word in rc_words:
            normalized = self._normalize_token(word)
            if normalized:
                tokens.append(normalized)
        if not tokens:
            return True
        if len(tokens) > 6:
            return False
        last = tokens[-1]
        if last not in self.auxiliary_clause_endings:
            return False
        preceding = tokens[:-1]
        return any(tok in self.modal_aux_tokens or tok in self.auxiliary_clause_endings for tok in preceding)

    def _looks_passive_clause(self, rc_words):
        """Heuristic passive detection for better SRC/ORC labelling."""
        tokens = []
        for word in rc_words:
            normalized = self._normalize_token(word)
            if normalized:
                tokens.append(normalized)
        length = len(tokens)
        if length < 2:
            return False
        for idx, token in enumerate(tokens):
            if token in self.passive_auxiliaries:
                if idx + 1 < length:
                    nxt = tokens[idx + 1]
                    if nxt.endswith("ed") or nxt in self.irregular_passive_verbs:
                        return True
            if token in self.modal_aux_tokens:
                for j in range(idx + 1, min(length, idx + 4)):
                    aux = tokens[j]
                    if aux in self.passive_auxiliaries:
                        if j + 1 < length:
                            nxt = tokens[j + 1]
                            if nxt.endswith("ed") or nxt in self.irregular_passive_verbs:
                                return True
                        break
        return False


    def parsing(self, sent):
        """
        Parse a sentence using SuPar dependency and constituency parsers.
        
        Args:
            sent (str): Input sentence to parse
            
        Returns:
            tuple: (tokenized_text, dependency_dataset, constituency_dataset)
                   Returns (None, None, None) if parsing fails
        """
        try:
            text = nltk.word_tokenize(sent)
            if not text:  # Skip empty sentences
                return None, None, None
                
            # Ensure text is not too long for the parser
            if len(text) > 512:  # SuPar's typical max length
                text = text[:512]
                
            dataset = self.parser.predict(text, verbose=False)
            dataset2 = self.parser2.predict([text], verbose=False)
            return text, dataset, dataset2
        except Exception as e:
            return None, None, None

    def acquire_subtrees(self, parse_tree):
        """
        Extract all SBAR (subordinate clause) subtrees from constituency parse tree.
        
        Args:
            parse_tree: Constituency parse tree object
            
        Returns:
            list: List of SBAR subtree strings
        """
        return [' '.join(subtree.leaves()) for subtree in parse_tree.subtrees() if subtree.label() == 'SBAR']

    def zero_relative_clause(self, subtrees, clause_verb, text, index_of_head_noun):
        """
        Identify zero relative clauses (reduced relative clauses without explicit relativizers).
        
        Args:
            subtrees (list): List of SBAR subtrees from constituency parsing
            clause_verb (str): Verb that might be part of a relative clause
            text (list): Tokenized sentence text
            index_of_head_noun (int): Index of the head noun in the sentence
            
        Returns:
            str: Zero relative clause content if found, empty string otherwise
        """
        relative_clause_content = ''
        length_of_RC = float('inf')
        for subtree in set(subtrees):
            if clause_verb in subtree and len(subtree) < length_of_RC:
                # Look for reduced relative clauses
                for i in range(index_of_head_noun, len(text)):
                    if " ".join(text[i:i+len(subtree.split())]) == subtree:
                        # Check if this looks like a reduced relative clause
                        if any(tag[1].startswith('VB') for tag in nltk.pos_tag(subtree.split())):
                            relative_clause_content = subtree
                            length_of_RC = len(subtree)
        return relative_clause_content

    def relative_clause(self, subtrees, clause_verb, relativizer, text):
        """
        Identify full relative clauses with explicit relativizers.
        
        Args:
            subtrees (list): List of SBAR subtrees from constituency parsing
            clause_verb (str): Verb that might be part of a relative clause
            relativizer (str): Relativizer word (which, that, who, etc.)
            text (list): Tokenized sentence text
            
        Returns:
            str: Full relative clause content if found, empty string otherwise
        """
        relative_clause_content = ''
        length_of_RC = float('inf')
        for subtree in set(subtrees):
            if clause_verb in subtree and relativizer in subtree and len(subtree) < length_of_RC:
                # Try to find the most complete relative clause
                for i in range(len(text)):
                    if " ".join(text[i:i+len(subtree.split())]) == subtree:
                        # Check if this is a complete relative clause
                        if any(marker in subtree for marker in ["who", "whom", "whose", "which", "that", "where", "when", "why"]):
                            relative_clause_content = subtree
                            length_of_RC = len(subtree)
        return relative_clause_content

    def process_relative_clause(self, text, dataset, index_of_word, subtrees, i):
        """Process a relative clause and extract its components."""
        try:
            rc = None
            relativizer_dict = {}

            index_of_head_noun = dataset.arcs[i][index_of_word] - 1
            if index_of_word < index_of_head_noun:
                return None

            head_noun = dataset.words[i][index_of_head_noun]
            relation_to_main = dataset.rels[i][index_of_head_noun]
            head_noun_lower = head_noun.lower()

            # Accept more relations for head nouns
            if relation_to_main not in ["nsubj", "dobj", "iobj", "csubj", "nsubjpass", "poss", "nmod", "compound"]:
                return None

            clause_verb = dataset.words[i][index_of_word]
            index_of_clause_verb = index_of_word

            # NEW: Filter temporary ambiguities (verb+preposition false positives)
            if self.is_temporary_ambiguity(text, index_of_clause_verb):
                return None

            # Find relativizer and its role
            index_of_relativizer = None
            relativizer_role = None
            has_preposition_before_relativizer = False
            preposition_before_relativizer = None
            
            for idx in range(index_of_head_noun, index_of_clause_verb):
                if dataset.words[i][idx] in self.relativizer_list:
                    index_of_relativizer = idx
                    relativizer_role = dataset.rels[i][idx]
                    
                    # Check if there's a preposition immediately before the relativizer
                    # This indicates a passive SRC (e.g., "eight of which were added")
                    if idx > 0:
                        prev_word = dataset.words[i][idx - 1]
                        prev_pos = dataset.tags[i][idx - 1] if hasattr(dataset, 'tags') and len(dataset.tags) > i else None
                        
                        # Check if previous word is a preposition (POS tag IN or just common prepositions)
                        common_preps = ['of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'about']
                        if (prev_pos == "IN") or (prev_word.lower() in common_preps):
                            has_preposition_before_relativizer = True
                            preposition_before_relativizer = prev_word
                    break

            relativizer = dataset.words[i][index_of_relativizer] if index_of_relativizer else "zero"
            
            # FILTER FALSE POSITIVES: Exclude complement clauses (only for zero relatives)
            # TEMPORARILY DISABLED - was blocking all zero relatives
            # TODO: Make this more selective
            # if relativizer == "zero" and relation_to_main == "dobj":
            #     main_verb_idx = dataset.arcs[i][index_of_head_noun] - 1
            #     if main_verb_idx >= 0:
            #         main_verb = dataset.words[i][main_verb_idx].lower()
            #         complement_verbs = ['mark', 'marks', 'show', 'shows', ...]
            #         if main_verb in complement_verbs:
            #             return None
            
            rc = self.relative_clause(subtrees, clause_verb, relativizer, text) if relativizer != "zero" else self.zero_relative_clause(subtrees, clause_verb, text, index_of_head_noun)
            if not rc:
                return None
            rc = rc.strip()
            rc_words = rc.split()
            if self._looks_like_manner_clause(head_noun_lower, relativizer, rc_words):
                return None
            if self._is_complement_clause(head_noun_lower, relation_to_main, relativizer_role, relativizer):
                return None
            if self._is_pp_fronted_without_gap(head_noun_lower, relativizer_role, preposition_before_relativizer, rc_words):
                return None
            if self._is_aux_fragment(relativizer, rc_words):
                return None

            # NEW FIX 4: REFINED multi-factor filtering for zero relatives
            if relativizer == "zero":
                confidence_score = 0  # Higher = more likely to be valid RC
                
                # === ABSOLUTE FILTERS (immediate rejection) ===
                
                # Filter 1: Minimum length (< 4 words almost always fragments)
                if len(rc_words) < 4:
                    return None
                
                # Filter 2: Parentheticals (em-dashes or 3+ commas)
                if '—' in rc or rc.count(',') >= 3:
                    return None
                
                # Filter 3: Short contraction fragments
                if len(rc_words) >= 2 and rc_words[1] in ["'", "'ve", "'re", "'s", "'m", "'ll", "'d"]:
                    if len(rc_words) <= 4:
                        return None
                
                # === SCORING SYSTEM (accumulate confidence) ===
                
                # Factor 1: Filler-gap distance (shorter = more reliable for zero relatives)
                filler_gap_distance = index_of_clause_verb - index_of_head_noun
                if filler_gap_distance <= 3:
                    confidence_score += 3  # Very short gap = strong indicator (78% of zeros)
                elif filler_gap_distance <= 5:
                    confidence_score += 2  # Moderate gap = good indicator
                elif filler_gap_distance <= 7:
                    confidence_score += 1  # Acceptable
                elif filler_gap_distance > 10:
                    confidence_score -= 3  # Very long gap = likely false positive
                
                # Factor 2: Head noun semantics
                
                if head_noun_lower in self.rc_attracting_nouns:
                    confidence_score += 2  # Strong RC indicator
                elif head_noun_lower in self.complement_attracting_nouns:
                    confidence_score -= 2  # Likely complement, not RC
                
                # Factor 3: Pronoun-initial patterns (more lenient)
                if rc_words and rc_words[0].lower() in ['i', 'you', 'we']:
                    if len(rc_words) == 4:
                        # Only 4 words + pronoun-initial = risky
                        if head_noun_lower not in self.rc_attracting_nouns:
                            confidence_score -= 1  # Penalty, but not too harsh
                    elif len(rc_words) >= 6:
                        # Longer pronoun-initial = probably okay
                        confidence_score += 1
                
                # Factor 4: RC length (longer = more context = more reliable)
                if len(rc_words) >= 8:
                    confidence_score += 2  # Good length
                elif len(rc_words) >= 6:
                    confidence_score += 1  # Decent length
                
                # Factor 5: Check for object gap (linguistic test)
                # If verb is transitive but missing object, head noun likely fills gap
                has_object = False
                for j in range(index_of_clause_verb, len(text)):
                    if dataset.arcs[i][j] - 1 == index_of_clause_verb:
                        if dataset.rels[i][j] in ['dobj', 'iobj']:
                            has_object = True
                            break
                
                if not has_object:
                    # Missing object = likely zero relative (head noun fills the gap)
                    confidence_score += 2
                
                # === DECISION: Filter if confidence too low ===
                # Lower threshold to -1 (was 0) to be less strict
                if confidence_score < -1:
                    return None  # Too many red flags, filter it out

            # Improved SRC/ORC detection
            clause_type = None
            structure_pattern = ""
            passive_like = self._looks_passive_clause(rc_words)

            if relativizer != "zero":
                # FIX 1: Check for passive SRC - nsubjpass means relativizer is subject of passive verb
                # e.g., "who was born", "who were fired", "that's called" - these are SRC, not ORC
                if relativizer_role == "nsubjpass" and passive_like:
                    clause_type = "SRC"
                    structure_pattern = "NP + Comp + Verb + NP2 (passive)"
                elif has_preposition_before_relativizer and relativizer_role in ["pobj"]:
                    # This is a passive SRC: NP + Prep + Comp + Verb
                    clause_type = "SRC"
                    structure_pattern = "NP + Prep + Comp + Verb (passive)"
                
                # Check for SRC - subject relative clause
                elif (relativizer_role in ["nsubj", "nsubjpass"] or 
                    (relativizer in ["who", "that"] and dataset.rels[i][index_of_relativizer] in ["nsubj", "nsubjpass"])):
                    clause_type = "SRC"
                    structure_pattern = "NP + Comp + Verb + NP2"
                
                # Check for ORC - object relative clause
                elif (relativizer_role in ["dobj", "iobj", "pobj", "mark"] or 
                      (relativizer in ["whom", "which", "that"] and dataset.rels[i][index_of_relativizer] in ["dobj", "iobj", "pobj"])):
                    clause_type = "ORC"
                    structure_pattern = "NP + Comp + NP2 + Verb"
                
                # Check for Other - relative clauses not affecting the verb
                elif (relativizer_role in ["poss", "nmod", "compound"] or 
                      relation_to_main in ["poss", "nmod", "compound"] or
                      (relativizer in ["whose", "which", "where", "when", "why"] and 
                       dataset.rels[i][index_of_relativizer] not in ["nsubj", "nsubjpass", "dobj", "iobj", "pobj"])):
                    clause_type = "Other"
                    structure_pattern = "NP + Comp + NP2"
                
                # Additional checks for ambiguous cases
                else:
                    # Look for intervening subject
                    intervening_subject = False
                    if index_of_relativizer is not None:
                        for j in range(index_of_relativizer + 1, index_of_clause_verb):
                            if dataset.rels[i][j] in ["nsubj", "nsubjpass"] and dataset.words[i][j] not in self.relativizer_list:
                                intervening_subject = True
                                break

                    if intervening_subject:
                        clause_type = "ORC"
                        structure_pattern = "NP + Comp + NP2 + Verb"
                    elif index_of_relativizer is not None and dataset.arcs[i][index_of_relativizer] - 1 == index_of_clause_verb:
                        clause_type = "SRC"
                        structure_pattern = "NP + Comp + Verb + NP2"
                    else:
                        # If we can't determine, check if it's modifying a non-verbal element
                        if any(dataset.rels[i][j] in ["poss", "nmod", "compound"] 
                              for j in range(index_of_head_noun, index_of_clause_verb)):
                            clause_type = "Other"
                            structure_pattern = "NP + Comp + NP2"
                        else:
                            # Use the relativizer's role as a fallback
                            if relativizer_role in ["nsubj", "nsubjpass"]:
                                clause_type = "SRC"
                            elif relativizer_role in ["dobj", "iobj", "pobj"]:
                                clause_type = "ORC"
                            else:
                                clause_type = "Other"
                                structure_pattern = "NP + Comp + NP2"
            else:
                # FIX 2: For zero relatives, check if there's an intervening subject
                # e.g., "the time the Hornets will meet" - "Hornets" is between "time" and "meet"
                # This means "time" is the object: "the time [that] the Hornets will meet" = ORC
                has_intervening_subject = False
                for j in range(index_of_head_noun + 1, index_of_clause_verb):
                    if (dataset.rels[i][j] in ["nsubj", "nsubjpass"] and 
                        dataset.arcs[i][j] - 1 == index_of_clause_verb):
                        # Found a subject that depends on the clause verb
                        has_intervening_subject = True
                        break
                
                # For zero relatives, determine type based on structure
                has_subject = any(dataset.rels[i][j] in ["nsubj", "nsubjpass"] 
                                for j in range(index_of_head_noun, index_of_clause_verb))
                has_object = any(dataset.rels[i][j] in ["dobj", "iobj", "pobj"] 
                               for j in range(index_of_head_noun, index_of_clause_verb))
                has_non_verbal = any(dataset.rels[i][j] in ["poss", "nmod", "compound"] 
                                   for j in range(index_of_head_noun, index_of_clause_verb))
                
                if has_intervening_subject:
                    # If there's a subject between the head noun and verb, the head noun is the object
                    clause_type = "ORC"
                elif has_non_verbal and not (has_subject or has_object):
                    clause_type = "Other"
                elif has_subject and not has_object:
                    clause_type = "SRC"
                elif has_object and not has_subject:
                    clause_type = "ORC"
                else:
                    # For zero relatives, check if it's modifying a non-verbal element
                    if relation_to_main in ["poss", "nmod", "compound"]:
                        clause_type = "Other"
                    else:
                        clause_type = "SRC"
                structure_pattern = "zero relative"

            # Calculate filler-gap dependency more accurately
            filler_gap = 0
            if relativizer != "zero":
                # Count words between relativizer and verb
                filler_gap = index_of_clause_verb - index_of_relativizer
            else:
                # For zero relatives, count words between head noun and verb
                filler_gap = index_of_clause_verb - index_of_head_noun

            relativizer_dict.update({
                "head_noun": head_noun,
                "head_noun_in_main_clause": relation_to_main,
                "clause_verb": clause_verb,
                "filler_gap_dependency": filler_gap,
                "restrictiveness": "restrictive",
                "relativizer": relativizer,
                "head_noun_in_rc": relativizer_role if relativizer_role else "zero",
                "relative_clause": rc,
                "relative_clause_length": len(rc.split(" ")),
                "rc_type": clause_type,
                "structure_pattern": structure_pattern
            })

            return relativizer_dict
        except Exception as e:
            return None

    def extract_relative_clauses(self):
        """Extract relative clauses from the input files."""
        relativizer_dict_list = []
        message = ""

        for file in self.input_files:
            try:
                print(f"\nProcessing file: {file}")
                total_sentences = 0
                processed_sentences = 0
                found_rcs = 0

                with open(file, 'r', encoding='utf-8') as f:
                    # Count total sentences first
                    total_sentences = sum(len(sent_tokenize(line)) for line in f if line.strip())
                    f.seek(0)  # Reset file pointer to beginning
                    
                    # Process the file line by line
                    for line_num, line in enumerate(f, 1):
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        # Process each sentence separately
                        sentences = sent_tokenize(line)
                        for sent_num, sent in enumerate(sentences, 1):
                            processed_sentences += 1
                            if processed_sentences % 100 == 0:  # Show progress every 100 sentences
                                print(f"Progress: {processed_sentences}/{total_sentences} sentences processed ({(processed_sentences/total_sentences)*100:.1f}%)")
                            
                            try:
                                # Parse the sentence using SuPar
                                text, dataset, dataset2 = self.parsing(sent)
                                if text is None or dataset is None or dataset2 is None:
                                    continue

                                for i in range(len(dataset2.trees)):
                                    subtrees = self.acquire_subtrees(dataset2.trees[i])
                                    for index_of_word, dep in enumerate(dataset.rels[i]):
                                        if dep == "rcmod":
                                            result = self.process_relative_clause(text, dataset, index_of_word, subtrees, i)
                                            if result:
                                                # Add sentence-level metrics to each relative clause
                                                result.update({
                                                    "file": file,
                                                    "sent": sent,
                                                    "sentence_word_count": self.count_words_in_text(sent),
                                                    "sentence_verb_count": self.count_verbs_in_text(sent)
                                                })
                                                relativizer_dict_list.append(result)
                                                found_rcs += 1
                                                print(f"Found relative clause: {result['relative_clause']}")

                                    # Clear memory
                                    del dataset
                                    del dataset2
                                    del subtrees

                            except Exception as e:
                                continue

                print(f"Found {found_rcs} relative clauses in {file}")

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

        # Convert results to DataFrame
        if relativizer_dict_list:
            df = pd.DataFrame(relativizer_dict_list)
            df.columns = ["head_noun", "head_noun_in_main_clause", "clause_verb",
                          "filler_gap_dependency", "restrictiveness", "relativizer", "head_noun_in_rc",
                          "relative_clause", "relative_clause_length", "rc_type", "structure_pattern", "file", "sent",
                          "sentence_word_count", "sentence_verb_count"]

            # Save results to CSV (append mode to preserve existing data)
            output_path = os.path.join(self.output_folder, "results_cursor.csv")
            
            # Check if file exists to determine whether to write header
            file_exists = os.path.exists(output_path)
            if file_exists:
                # Append to existing file without header
                df.to_csv(output_path, mode='a', header=False, index=False)
                message += f"\nResults appended to {output_path}"
            else:
                # Create new file with header
                df.to_csv(output_path, index=False)
                message += f"\nResults saved to {output_path}"
            
            message += f"\nTotal relative clauses found in this run: {len(df)}"
        else:
            df = pd.DataFrame()
            message += "\nNo relative clauses were found in the input files."

        return df, message

    def get_index_of_word(self, dataset, word):
        """Get the index of a word in the dataset."""
        try:
            if not dataset or not hasattr(dataset, 'words'):
                print("Invalid dataset object")
                return None
                
            if not word:
                print("Invalid word")
                return None
                
            # Ensure we're using the first sentence's words
            if not dataset.words or len(dataset.words) == 0:
                print("No words in dataset")
                return None
                
            words = dataset.words[0]  # Always use first sentence
            if not words:
                print("Empty words list")
                return None
                
            try:
                return words.index(word)
            except ValueError:
                print(f"Word '{word}' not found in dataset")
                return None
        except Exception as e:
            print(f"Error in get_index_of_word: {str(e)}")
            return None