# Relative Clause Extraction and Analysis

A Python-based system for extracting and analyzing relative clauses from text corpora using advanced syntactic parsing techniques.

## Overview

This project implements a comprehensive relative clause extraction system that identifies and analyzes different types of relative clauses in text. The system uses both dependency and constituency parsing to accurately identify relative clauses and their components.

### Key Features

- **Multiple RC Types**: Identifies full, reduced, and zero relative clauses with accurate SRC/ORC classification
- **Advanced Parsing**: Uses SuPar for both dependency and constituency parsing
- **Intelligent Filtering**: Automatically filters out false positives including manner clauses, complement constructions, and elliptical fragments
- **Comprehensive Analysis**: Extracts relativizers, head nouns, and RC content with detailed structural patterns
- **Corpus Processing**: Batch processing of multiple text files with progress tracking
- **Detailed Reporting**: Comprehensive analysis with statistics and examples

## Files

### Core Analysis
- `relative_clause_extractor.py` - Main relative clause extraction system
- `main.py` - Entry point and execution script
- `tidy.py` - Text cleaning and preprocessing utilities

### Data Files (Download from Google Drive)
- `models/` - Parser models (ptb.biaffine.dep.lstm.char, ptb.crf.con.lstm.char) - **Download from Google Drive**
- `input_texts/` - Sample input text files (f23, f24, f25 corpora) - **Download from Google Drive**
- `result/` - Pre-computed results for f23, f24, f25 corpora - **Download from Google Drive** (optional, for reference)
- `rc-nlp-sample-2025Nov.xlsx` - Validation data with annotated results - **Download from Google Drive**

### Generated Outputs
- When you run the extractor, new results will be saved to `result/` directory

## Files Description

### Core Analysis Files

#### `relative_clause_extractor.py` - **MAIN EXTRACTOR**
- **Purpose**: Core relative clause extraction system using SuPar parsing
- **Key Features**:
  - **Dual Parsing**: Uses both dependency and constituency parsers for robust extraction
  - **Multiple RC Types**: Identifies full, reduced, and zero relative clauses with accurate SRC/ORC classification
  - **Passive Voice Detection**: Correctly classifies passive relative clauses (e.g., "who was born", "that's called") as subject relative clauses
  - **False Positive Filtering**: Applies linguistic heuristics to filter out manner clauses, complement constructions, and elliptical fragments
  - **Comprehensive Relativizers**: Supports which, that, who, whom, whose, where, when, why, etc.
  - **Head Noun Detection**: Identifies the noun that the relative clause modifies and its role in both main and relative clauses
  - **Content Extraction**: Extracts the complete relative clause content with structural pattern analysis
- **Output**: Detailed analysis of each relative clause found including type, structure, and dependency information
- **Usage**: Used by main.py for batch processing of text files

#### `main.py` - **ENTRY POINT**
- **Purpose**: Main execution script for relative clause extraction
- **Key Features**:
  - Initializes the relative clause extractor
  - Processes all text files in input_texts directory
  - Generates comprehensive statistics and reports
  - Saves results to CSV format
- **Output**: Processed results and summary statistics
- **Usage**: Primary script for running the complete analysis pipeline

#### `tidy.py` - **DATA PROCESSING**
- **Purpose**: Data cleaning and processing utilities
- **Key Features**:
  - Cleans and preprocesses extracted data
  - Handles edge cases and formatting issues
  - Prepares data for analysis and reporting
- **Output**: Cleaned and processed datasets
- **Usage**: Post-processing utility for data refinement

### Data and Results Files

#### `result/spaCy_RCs.csv` - **MAIN RESULTS DATASET**
- **Purpose**: Contains all extracted relative clauses and their analysis
- **Columns**:
  - `file`: Source text file name
  - `sentence`: Original sentence containing the relative clause
  - `rc_type`: Type of relative clause (full, reduced, zero)
  - `relativizer`: The relativizer used (which, that, who, etc.)
  - `head_noun`: The noun that the relative clause modifies
  - `rc_content`: The complete relative clause content
  - `sentence_index`: Position of the sentence in the file
  - `rc_index`: Position of the relative clause in the sentence
- **Usage**: Primary dataset for analysis and reporting

#### `input_texts/` - **INPUT DATA**
- **Purpose**: Contains text files to be processed
- **Content**: Various text files (.txt format) for relative clause extraction
- **Usage**: Source data for the extraction process

#### `result/` - **OUTPUT FILES**
- **Purpose**: Contains processed results and output files
- **Content**: CSV files with extraction results and analysis
- **Usage**: Processed output from the extraction pipeline

#### `models/` - **PARSER MODELS**
- **Purpose**: Contains SuPar parser models for syntactic analysis
- **Content**: 
  - `ptb.biaffine.dep.lstm.char` - Dependency parser model
  - `ptb.crf.con.lstm.char` - Constituency parser model
- **Source**: Pre-trained models from SuPar research library
- **Research Paper**: "SuPar: A Unified Parser for Syntactic Analysis" (Yu et al., 2020)
- **Usage**: Required for syntactic parsing and analysis

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alarakaymak/relative-clause-extractor.git
   cd relative-clause-extractor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

4. **Download required files from Google Drive**:
   
   Due to file size limitations, the following are hosted on Google Drive:
   - **Models** (~665MB): Parser models needed to run the extractor
   - **Input corpora**: Sample input text files (f23, f24, f25) for testing
   - **Results** (optional): Pre-computed extraction results for f23, f24, f25 corpora
   - **Validation data**: Annotated Excel file with validation results
   
   **Download from Google Drive**: https://drive.google.com/drive/folders/18fTLbj3uqtEk22aJ94A0w-UqqyyLd-_e?usp=drive_link
   
   After downloading, extract and organize as follows:
   ```
   relative-clause-extractor/
   ├── models/
   │   ├── ptb.biaffine.dep.lstm.char.zip
   │   └── ptb.crf.con.lstm.char.zip
   ├── input_texts/
   │   ├── completed-websites-f23/
   │   ├── completed-websites-f24/
   │   └── completed-websites-f25/
   ├── result/
   │   ├── f23/
   │   │   └── results_cursor.csv
   │   ├── f24/
   │   │   └── results_cursor.csv
   │   └── f25/
   │       └── results_cursor.csv
   └── rc-nlp-sample-2025Nov.xlsx (validation data)
   ```
   
   **Note**: The models will be automatically unzipped when first used, or you can unzip them manually.
   
   **Option B: Automatic download script**
   ```bash
   python download_models.py
   ```
   This will attempt to download both required models (~665MB total) from official sources.
   If the URLs don't work, use Option A or C instead.
   
   **Option C: Manual download from SuPar releases**
   - Visit: https://github.com/yzhangcs/parser/releases
   - Look for release v1.0.0 or latest release
   - Download these two files:
     - `ptb.biaffine.dep.lstm.char.zip` (dependency parser, ~330MB)
     - `ptb.crf.con.lstm.char.zip` (constituency parser, ~330MB)
   - Place both `.zip` files in the `models/` directory
   - The extractor will automatically unzip them when first used, or unzip manually

## Usage

### Basic Usage

Run the main extraction script:
```bash
python main.py
```

This will:
1. Process all text files in `input_texts/`
2. Extract relative clauses using both parsers
3. Generate comprehensive results in `result/spaCy_RCs.csv`
4. Display summary statistics

### Advanced Usage

For custom analysis:
```python
from relative_clause_extractor import RelativeClause

# Initialize extractor
rc_extractor = RelativeClause(input_texts="input_texts", output_folder="result")

# Extract relative clauses
df, message = rc_extractor.extract_relative_clauses()

# Analyze results
print(df.head())
print(df['rc_type'].value_counts())
```

## Results and Analysis

### Relative Clause Types Found

The system identifies three main types of relative clauses:

1. **Full Relative Clauses**: Complete relative clauses with explicit relativizers
   - Example: "The book that I read was interesting"

2. **Reduced Relative Clauses**: Relative clauses without explicit relativizers
   - Example: "The book I read was interesting"

3. **Zero Relative Clauses**: Minimal relative clauses with implied relativizers
   - Example: "The man walking down the street"

### Relativizer Distribution

Common relativizers identified:
- **which**: Most common for non-human antecedents
- **that**: General purpose relativizer
- **who**: Human antecedents
- **where**: Location references
- **when**: Time references

### Classification System

The extractor classifies relative clauses into three main types:

1. **Subject Relative Clauses (SRC)**: The relativizer functions as the subject of the relative clause
   - Example: "The book **that** sits on the shelf" (that = subject of "sits")
   - Includes passive constructions: "The man **who** was born in 1990" (who = subject of passive "was born")

2. **Object Relative Clauses (ORC)**: The relativizer functions as the object of the relative clause
   - Example: "The book **that** I read" (that = object of "read")

3. **Other**: Relative clauses where the relativizer has other grammatical roles (possessive, adverbial, etc.)
   - Example: "The student **whose** book was lost" (whose = possessive)

The system uses dependency parsing to determine the grammatical role of the relativizer, enabling accurate classification even in complex passive constructions.

### Performance Statistics

- **Processing Speed**: Handles large text files efficiently, processing thousands of sentences per corpus
- **Accuracy**: High precision in relative clause identification with automatic filtering of common false positives
- **Coverage**: Comprehensive detection of various RC types including passive constructions and complex structures
- **Validation**: System has been validated on annotated corpora with high accuracy in both RC detection and SRC/ORC classification

## Technical Details

### Text Processing Approach

The relative clause system uses a **two-stage text processing approach**:

1. **Text Preprocessing** (`tidy.py`):
   - Cleans input text files before analysis
   - Fixes punctuation errors and spacing issues
   - Removes emojis and normalizes text
   - Prepares text for accurate syntactic parsing

2. **Syntactic Analysis** (`relative_clause_extractor.py`):
   - Uses cleaned text for dual parsing
   - Dependency parsing for syntactic relationships
   - Constituency parsing for phrase structure

This approach ensures high accuracy by providing clean, well-formatted text to the parsers.

### Parsing Approach

The system uses a dual-parsing approach:

1. **Dependency Parsing**: Identifies syntactic relationships between words, including the role of relativizers (subject, object, etc.)
2. **Constituency Parsing**: Identifies phrase structure and extracts complete relative clause boundaries

This combination ensures accurate relative clause detection across different sentence structures. The system then applies linguistic heuristics to filter out false positives, such as manner/temporal adjuncts (e.g., "the way that..."), complement clauses, and incomplete elliptical constructions that might be misidentified as relative clauses.

### Parser Models

The system uses pre-trained models from the **SuPar research library**:

- **Source**: SuPar: A Unified Parser for Syntactic Analysis (Yu et al., 2020)
- **Paper**: https://arxiv.org/abs/2004.11794
- **Models**: 
  - `ptb.biaffine.dep.lstm.char`: Dependency parsing model
  - `ptb.crf.con.lstm.char`: Constituency parsing model
- **Training Data**: Penn Treebank (PTB) corpus
- **Performance**: State-of-the-art parsing accuracy
- **License**: Research/academic use

### Relativizer Detection

The system recognizes a comprehensive list of relativizers:
- **Basic**: which, that, who, whom, whose
- **Extended**: where, when, why, what, whatever
- **Compound**: whoever, whomever, whichever

### False Positive Filtering

The system employs targeted linguistic heuristics to filter out common false positives:

- **Manner/Temporal Clauses**: Filters constructions like "the way that..." and "the time that..." which are adjuncts rather than relative clauses
- **Complement Clauses**: Identifies and excludes complement-taking noun constructions (e.g., "the fact that...", "the idea that...") that follow verbs like "show", "prove", etc.
- **PP-Fronted Constructions**: Filters prepositional phrase constructions without true extraction (e.g., "claim to which trends...")
- **Elliptical Fragments**: Removes incomplete constructions ending in auxiliaries (e.g., "something he shouldn't have")

These filters are applied early in the extraction pipeline, ensuring high precision while maintaining recall for legitimate relative clauses.

### Error Handling

- **Long Sentences**: Truncates sentences exceeding parser limits
- **Empty Input**: Handles empty or malformed input gracefully
- **Parser Failures**: Continues processing even if individual sentences fail

## Future Enhancements

1. **Cross-linguistic Support**: Extend to other languages
2. **Semantic Analysis**: Add semantic role labeling
3. **Performance Optimization**: Improve processing speed for large corpora
4. **Web Interface**: Create a web-based analysis tool

## Dependencies

- **SuPar**: Advanced parsing library
- **NLTK**: Natural language processing
- **pandas**: Data manipulation and analysis
- **torch**: Deep learning framework (for SuPar)

---

## Contact

For questions or contributions, please contact alara.kaymak@vanderbilt.edu. 