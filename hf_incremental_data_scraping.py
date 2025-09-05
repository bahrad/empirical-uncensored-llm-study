import time
import json
import csv
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from huggingface_hub import HfApi, list_models, list_datasets, ModelCard
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import re
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('huggingface_scraper_incremental.log'),
        logging.StreamHandler()
    ]
)

class IncrementalHuggingFaceScraper:
    def __init__(self, rate_limit_delay=1.0, previous_timestamp=None):
        """
        Initialize the incremental scraper with rate limiting
        
        Args:
            rate_limit_delay: Delay between API calls in seconds
            previous_timestamp: Timestamp of previous run to load data from (format: YYYYMMDD_HHMMSS)
        """
        self.api = HfApi()
        self.rate_limit_delay = rate_limit_delay
        self.session = self._create_session()
        self.previous_timestamp = previous_timestamp
        
        # Track what we've already seen
        self.existing_models = set()
        self.existing_datasets = set()
        self.previous_models_data = []
        self.previous_datasets_data = []
        
        # Keywords that might indicate safety modifications
        self.safety_keywords = [
            'uncensored', 'abliterat', 'unfilter', 'jailbreak', 'jailbrok',
            'no-safe', 'no-filter', 'nofilter', 'nosafe', 'unrestrict', 'unlock', 'freed',
            'decensor', 'unsafe', 'unalign', 'dealign', 'de-align', 'roleplay', 'role-play'
        ]
        
        # Keywords for datasets used in abliteration
        self.dataset_keywords = self.safety_keywords + [
            'abliteration', 'uncensor', 'harmful', 'toxic',
            'refusal', 'alignment', 'safety-removal', 'do-anything'
        ]
        
        # Load previous data if timestamp provided
        if self.previous_timestamp:
            self.load_previous_data()
    
    def _create_session(self):
        """Create a session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def load_previous_data(self) -> bool:
        """Load data from previous run based on timestamp"""
        if not self.previous_timestamp:
            return False
            
        # Find files with the given timestamp
        models_pattern = f"safety_analysis_models_{self.previous_timestamp}.json"
        datasets_pattern = f"safety_analysis_datasets_{self.previous_timestamp}.json"
        
        models_loaded = False
        datasets_loaded = False
        
        # Load models
        if os.path.exists(models_pattern):
            try:
                with open(models_pattern, 'r') as f:
                    self.previous_models_data = json.load(f)
                    self.existing_models = {model['model_id'] for model in self.previous_models_data}
                    logging.info(f"Loaded {len(self.previous_models_data)} existing models from {models_pattern}")
                    models_loaded = True
            except Exception as e:
                logging.error(f"Error loading previous models data: {e}")
        else:
            logging.warning(f"Previous models file not found: {models_pattern}")
        
        # Load datasets
        if os.path.exists(datasets_pattern):
            try:
                with open(datasets_pattern, 'r') as f:
                    self.previous_datasets_data = json.load(f)
                    self.existing_datasets = {dataset['dataset_id'] for dataset in self.previous_datasets_data}
                    logging.info(f"Loaded {len(self.previous_datasets_data)} existing datasets from {datasets_pattern}")
                    datasets_loaded = True
            except Exception as e:
                logging.error(f"Error loading previous datasets data: {e}")
        else:
            logging.warning(f"Previous datasets file not found: {datasets_pattern}")
        
        return models_loaded or datasets_loaded
    
    def find_latest_timestamp(self, prefix='safety_analysis') -> Optional[str]:
        """Find the latest timestamp from existing files"""
        model_files = glob.glob(f"{prefix}_models_*.json")
        dataset_files = glob.glob(f"{prefix}_datasets_*.json")
        
        all_files = model_files + dataset_files
        if not all_files:
            return None
        
        # Extract timestamps from filenames
        timestamps = []
        for filename in all_files:
            match = re.search(r'(\d{8}_\d{6})\.json$', filename)
            if match:
                timestamps.append(match.group(1))
        
        if timestamps:
            latest = sorted(timestamps)[-1]
            logging.info(f"Found latest timestamp: {latest}")
            return latest
        return None
        
    def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """Extract comprehensive metadata for a model"""
        # Skip if we've already processed this model
        if model_id in self.existing_models:
            logging.debug(f"Skipping already processed model: {model_id}")
            return None
            
        try:
            time.sleep(self.rate_limit_delay)
            info = self.api.model_info(model_id, files_metadata=True)
            
            # Extract base model if it's a fine-tune
            base_model = None
            model_card_content = None
            
            # Try extracting from card_data (structured metadata)
            if hasattr(info, 'card_data') and info.card_data:
                base_model = info.card_data.get('base_model')
            
            # Only load model card if base_model wasn't found
            if base_model is None:
                try:
                    card = ModelCard.load(model_id)
                    model_card_content = card.content
            
                    # Fallback: regex to find base model in card text
                    match = re.search(r'fine[- ]?tune(?:d)? from ([\w\-/\.]+)', model_card_content, re.IGNORECASE)
                    if match:
                        base_model = match.group(1)
        
                except Exception as e:
                    logging.warning(f"Could not load model card for {model_id}: {e}")
            else:
                # Optionally still load the card if you want the content for storage/analysis
                try:
                    card = ModelCard.load(model_id)
                    model_card_content = card.content
                except Exception as e:
                    logging.warning(f"Could not load model card content for {model_id}: {e}")
                
            # Check for safety-related keywords in model ID and tags
            is_potentially_modified = any(
                keyword in model_id.lower() for keyword in self.safety_keywords
            ) or any(
                keyword in tag.lower() for tag in (info.tags or []) 
                for keyword in self.safety_keywords
            )
            
            metadata = {
                'model_id': info.modelId,
                'author': info.author,
                'created_at': info.created_at.isoformat() if info.created_at else None,
                'last_modified': info.last_modified.isoformat() if info.last_modified else None,
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags,
                'pipeline_tag': getattr(info, 'pipeline_tag', None),
                'library_name': getattr(info, 'library_name', None),
                'license': getattr(info, 'license', None),
                'base_model': base_model,
                'is_potentially_modified': is_potentially_modified,
                'model_card_content': model_card_content,
                'private': info.private,
                'gated': getattr(info, 'gated', False),
                'disabled': getattr(info, 'disabled', False),
                'files': [f.rfilename for f in info.siblings] if hasattr(info, 'siblings') else [],
                'scraped_at': datetime.now().isoformat()  # Add scrape timestamp
            }
            
            # Check if model has been flagged or has discussions about safety
            if hasattr(info, 'discussions_disabled'):
                metadata['discussions_disabled'] = info.discussions_disabled
            
            return metadata
                
        except Exception as e:
            logging.error(f"Error fetching {model_id}: {e}")
            return None
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Extract metadata for datasets that might be used for abliteration"""
        # Skip if we've already processed this dataset
        if dataset_id in self.existing_datasets:
            logging.debug(f"Skipping already processed dataset: {dataset_id}")
            return None
            
        try:
            time.sleep(self.rate_limit_delay)
            info = self.api.dataset_info(dataset_id)
            
            # Check if dataset might be used for safety removal
            is_safety_related = any(
                keyword in dataset_id.lower() for keyword in self.dataset_keywords
            ) or any(
                keyword in tag.lower() for tag in (info.tags or []) 
                for keyword in self.dataset_keywords
            )
            
            metadata = {
                'dataset_id': info.id,
                'author': info.author,
                'created_at': info.created_at.isoformat() if info.created_at else None,
                'last_modified': info.last_modified.isoformat() if info.last_modified else None,
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags,
                'task_categories': getattr(info, 'task_categories', None),
                'size_categories': getattr(info, 'size_categories', None),
                'is_safety_related': is_safety_related,
                'private': info.private,
                'gated': getattr(info, 'gated', False),
                'scraped_at': datetime.now().isoformat()  # Add scrape timestamp
            }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error fetching dataset {dataset_id}: {e}")
            return None
    
    def find_modified_models(self, limit=1000) -> List[Dict]:
        """Find models that might have safety modifications"""
        new_models = []
        processed_count = 0
        skipped_count = 0
        
        # Search for each safety keyword
        for keyword in self.safety_keywords:
            logging.info(f"Searching for models with keyword: {keyword}")
            
            try:
                models = list_models(
                    search=f"{keyword}",
                    sort="downloads",
                    direction=-1,
                    limit=limit
                )
                
                for model in tqdm(models, desc=f"Processing {keyword} models"):
                    if model.modelId in self.existing_models:
                        skipped_count += 1
                        continue
                        
                    metadata = self.get_model_metadata(model.modelId)
                    if metadata:
                        metadata['search_keyword'] = keyword
                        new_models.append(metadata)
                        processed_count += 1
                        
            except Exception as e:
                logging.error(f"Error searching for {keyword}: {e}")
        
        logging.info(f"Found {processed_count} new models, skipped {skipped_count} existing ones")
        return new_models
    
    def find_safety_datasets(self, limit=500) -> List[Dict]:
        """Find datasets that might be used for safety removal"""
        new_datasets = []
        processed_count = 0
        skipped_count = 0
        
        for keyword in self.dataset_keywords:
            logging.info(f"Searching for datasets with keyword: {keyword}")
            
            try:
                datasets = list_datasets(
                    search=f"{keyword}",
                    sort="downloads",
                    direction=-1,
                    limit=limit
                )
                
                for dataset in tqdm(datasets, desc=f"Processing {keyword} datasets"):
                    if dataset.id in self.existing_datasets:
                        skipped_count += 1
                        continue
                        
                    metadata = self.get_dataset_metadata(dataset.id)
                    if metadata:
                        metadata['search_keyword'] = keyword
                        new_datasets.append(metadata)
                        processed_count += 1
                        
            except Exception as e:
                logging.error(f"Error searching for dataset {keyword}: {e}")
        
        logging.info(f"Found {processed_count} new datasets, skipped {skipped_count} existing ones")
        return new_datasets
    
    def merge_with_previous_data(self, new_models: List[Dict], new_datasets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Merge new data with previously loaded data"""
        # Combine models
        all_models = self.previous_models_data + new_models
        
        # Remove duplicates based on model_id (shouldn't happen, but just in case)
        seen_models = set()
        unique_models = []
        for model in all_models:
            if model['model_id'] not in seen_models:
                seen_models.add(model['model_id'])
                unique_models.append(model)
        
        # Combine datasets
        all_datasets = self.previous_datasets_data + new_datasets
        
        # Remove duplicates based on dataset_id
        seen_datasets = set()
        unique_datasets = []
        for dataset in all_datasets:
            if dataset['dataset_id'] not in seen_datasets:
                seen_datasets.add(dataset['dataset_id'])
                unique_datasets.append(dataset)
        
        return unique_models, unique_datasets
    
    def analyze_model_relationships(self, models: List[Dict]) -> pd.DataFrame:
        """Analyze relationships between base models and their modifications"""
        relationships = []
        
        for model in models:
            if model.get('base_model'):
                relationships.append({
                    'modified_model': model['model_id'],
                    'base_model': model['base_model'],
                    'author': model['author'],
                    'downloads': model['downloads'],
                    'created_at': model['created_at'],
                    'search_keyword': model.get('search_keyword', ''),
                    'scraped_at': model.get('scraped_at', '')
                })
                
        return pd.DataFrame(relationships)
    
    def save_results(self, models: List[Dict], datasets: List[Dict], prefix='safety_analysis'):
        """Save results to files with current timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        models_file = f"{prefix}_models_{timestamp}.json"
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)
        logging.info(f"Saved {len(models)} total models to {models_file}")
        
        # Save datasets
        datasets_file = f"{prefix}_datasets_{timestamp}.json"
        with open(datasets_file, 'w') as f:
            json.dump(datasets, f, indent=2)
        logging.info(f"Saved {len(datasets)} total datasets to {datasets_file}")
        
        # Create summary CSV
        summary_data = []
        for model in models:
            summary_data.append({
                'model_id': model['model_id'],
                'author': model['author'],
                'base_model': model.get('base_model', ''),
                'downloads': model['downloads'],
                'likes': model['likes'],
                'created_at': model['created_at'],
                'is_potentially_modified': model['is_potentially_modified'],
                'search_keyword': model.get('search_keyword', ''),
                'tags': ', '.join(model.get('tags', [])),
                'scraped_at': model.get('scraped_at', '')
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"{prefix}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Saved summary to {summary_file}")
        
        # Create incremental report
        report_file = f"{prefix}_incremental_report_{timestamp}.txt"
        self.create_incremental_report(report_file, models, datasets)
        
        return models_file, datasets_file, summary_file
    
    def create_incremental_report(self, filename: str, all_models: List[Dict], all_datasets: List[Dict]):
        """Create a report showing what was added in this run"""
        with open(filename, 'w') as f:
            f.write("=== INCREMENTAL SCRAPING REPORT ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if self.previous_timestamp:
                f.write(f"Previous data from: {self.previous_timestamp}\n")
                f.write(f"Previous models count: {len(self.previous_models_data)}\n")
                f.write(f"Previous datasets count: {len(self.previous_datasets_data)}\n")
            else:
                f.write("No previous data loaded (fresh run)\n")
            
            new_models_count = len(all_models) - len(self.previous_models_data)
            new_datasets_count = len(all_datasets) - len(self.previous_datasets_data)
            
            f.write(f"\n=== UPDATE SUMMARY ===\n")
            f.write(f"New models found: {new_models_count}\n")
            f.write(f"New datasets found: {new_datasets_count}\n")
            f.write(f"Total models now: {len(all_models)}\n")
            f.write(f"Total datasets now: {len(all_datasets)}\n")
            
            # List new models
            if new_models_count > 0:
                f.write("\n=== NEW MODELS ===\n")
                new_models = [m for m in all_models if m['model_id'] not in self.existing_models]
                for model in new_models[:50]:  # Show first 50
                    f.write(f"- {model['model_id']} (downloads: {model['downloads']}, keyword: {model.get('search_keyword', 'N/A')})\n")
                if new_models_count > 50:
                    f.write(f"... and {new_models_count - 50} more\n")
            
            # List new datasets
            if new_datasets_count > 0:
                f.write("\n=== NEW DATASETS ===\n")
                new_datasets = [d for d in all_datasets if d['dataset_id'] not in self.existing_datasets]
                for dataset in new_datasets[:50]:  # Show first 50
                    f.write(f"- {dataset['dataset_id']} (downloads: {dataset['downloads']}, keyword: {dataset.get('search_keyword', 'N/A')})\n")
                if new_datasets_count > 50:
                    f.write(f"... and {new_datasets_count - 50} more\n")
        
        logging.info(f"Created incremental report: {filename}")

def main():
    """Main execution function with incremental update support"""
    parser = argparse.ArgumentParser(description='Incremental HuggingFace safety-related model scraper')
    parser.add_argument('--timestamp', type=str, help='Previous timestamp to load data from (YYYYMMDD_HHMMSS)')
    parser.add_argument('--auto-latest', action='store_true', help='Automatically find and use latest timestamp')
    parser.add_argument('--model-limit', type=int, default=100, help='Limit for models per keyword search')
    parser.add_argument('--dataset-limit', type=int, default=50, help='Limit for datasets per keyword search')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Rate limit delay in seconds')
    
    args = parser.parse_args()
    
    # Determine which timestamp to use
    previous_timestamp = None
    if args.timestamp:
        previous_timestamp = args.timestamp
        logging.info(f"Using specified timestamp: {previous_timestamp}")
    elif args.auto_latest:
        scraper_temp = IncrementalHuggingFaceScraper()
        previous_timestamp = scraper_temp.find_latest_timestamp()
        if previous_timestamp:
            logging.info(f"Using automatically detected latest timestamp: {previous_timestamp}")
        else:
            logging.info("No previous data found, starting fresh")
    
    # Initialize scraper with previous data if available
    scraper = IncrementalHuggingFaceScraper(
        rate_limit_delay=args.rate_limit,
        previous_timestamp=previous_timestamp
    )
    
    # Find new modified models
    logging.info("Starting to find new/updated modified models...")
    new_models = scraper.find_modified_models(limit=args.model_limit)
    
    # Remove duplicates within new models
    seen = set()
    unique_new_models = []
    for model in new_models:
        if model['model_id'] not in seen:
            seen.add(model['model_id'])
            unique_new_models.append(model)
    
    logging.info(f"Found {len(unique_new_models)} unique new modified models")
    
    # Find new safety-related datasets
    logging.info("Starting to find new/updated safety-related datasets...")
    new_datasets = scraper.find_safety_datasets(limit=args.dataset_limit)
    
    # Merge with previous data
    all_models, all_datasets = scraper.merge_with_previous_data(unique_new_models, new_datasets)
    
    # Analyze relationships
    relationships_df = scraper.analyze_model_relationships(all_models)
    if not relationships_df.empty:
        relationships_file = f"model_relationships_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        relationships_df.to_csv(relationships_file, index=False)
        logging.info(f"Saved relationships to {relationships_file}")
    
    # Save all results (old + new)
    scraper.save_results(all_models, all_datasets)
    
    # Print summary statistics
    print("\n=== INCREMENTAL UPDATE SUMMARY ===")
    if previous_timestamp:
        print(f"Loaded previous data from: {previous_timestamp}")
        print(f"Previous models: {len(scraper.previous_models_data)}")
        print(f"Previous datasets: {len(scraper.previous_datasets_data)}")
    print(f"New models found: {len(unique_new_models)}")
    print(f"New datasets found: {len(new_datasets)}")
    print(f"Total unique modified models: {len(all_models)}")
    print(f"Total safety-related datasets: {len(all_datasets)}")
    print(f"Models with identified base models: {len(relationships_df)}")
    
    # Top authors of modified models
    author_counts = pd.Series([m['author'] for m in all_models]).value_counts()
    print("\nTop 10 authors of modified models:")
    print(author_counts.head(10))
    
    # Most common base models
    if not relationships_df.empty:
        base_model_counts = relationships_df['base_model'].value_counts()
        print("\nTop 10 most commonly modified base models:")
        print(base_model_counts.head(10))
    
    # Show some statistics about new vs old
    if unique_new_models:
        print("\nNew models by keyword:")
        keyword_counts = pd.Series([m.get('search_keyword', 'Unknown') for m in unique_new_models]).value_counts()
        print(keyword_counts.head())

if __name__ == "__main__":
    main()