#!/usr/bin/env python3
"""
Results Migration Utility - Simplified Structure

This script migrates existing results to the new simplified directory structure:
- results/symbols/{SYMBOL}/ for individual symbol analysis
- results/portfolios/ for multi-symbol portfolio analysis
- results/general/ for general testing and comparisons

Usage:
    python tools/migrate_results.py --migrate-all
    python tools/migrate_results.py --check-structure
    python tools/migrate_results.py --cleanup-old
"""

import os
import shutil
import argparse
from pathlib import Path
import json
from datetime import datetime
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the simplified directory structure
SIMPLIFIED_STRUCTURE = {
    'symbols': {
        'description': 'Individual symbol analysis',
        'subdirs': ['1h', '4h', '1d', 'optimization']
    },
    'portfolios': {
        'description': 'Multi-symbol portfolio analysis',
        'subdirs': ['crypto_majors', 'defi_tokens', 'optimization']
    },
    'general': {
        'description': 'General testing and comparisons',
        'subdirs': ['strategy_comparisons', 'timeframe_analysis', 'testing']
    }
}

def normalize_symbol(symbol):
    """Convert symbol to directory-safe format"""
    return symbol.replace('/', '_').replace('-', '_')

def extract_symbol_from_path(path_str):
    """Extract symbol from various path patterns"""
    patterns = [
        r'optimization_([A-Z]+_[A-Z]+)_\d+[hd]',  # optimization_BTC_USDT_4h
        r'([A-Z]+_[A-Z]+)_\d+[hd]',               # BTC_USDT_4h
        r'([A-Z]+_[A-Z]+)',                       # BTC_USDT
    ]
    
    for pattern in patterns:
        match = re.search(pattern, path_str)
        if match:
            return match.group(1)
    return None

def extract_timeframe_from_path(path_str):
    """Extract timeframe from path"""
    match = re.search(r'(\d+[hd])', path_str)
    return match.group(1) if match else None

def create_simplified_structure():
    """Create the simplified directory structure"""
    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    
    for category, info in SIMPLIFIED_STRUCTURE.items():
        category_dir = base_dir / category
        category_dir.mkdir(exist_ok=True)
        logger.info(f"Created {category}/ - {info['description']}")
        
        # Create standard subdirectories
        for subdir in info['subdirs']:
            (category_dir / subdir).mkdir(exist_ok=True)
            logger.info(f"  Created {category}/{subdir}/")

def migrate_symbol_results():
    """Migrate symbol-specific results to symbols/ directory"""
    logger.info("Migrating symbol-specific results...")
    
    # Find all existing result directories that contain symbol data
    existing_dirs = []
    
    # Check various locations for symbol results
    search_paths = [
        "results/examples",
        "results/tools/strategy_tester",
        "results/production",
        "results/research"
    ]
    
    for search_path in search_paths:
        if Path(search_path).exists():
            for item in Path(search_path).iterdir():
                if item.is_dir():
                    symbol = extract_symbol_from_path(item.name)
                    if symbol:
                        existing_dirs.append((item, symbol))
    
    # Migrate each symbol directory
    symbols_migrated = set()
    for old_path, symbol in existing_dirs:
        symbol_dir = Path("results/symbols") / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Determine target subdirectory based on content
        timeframe = extract_timeframe_from_path(old_path.name)
        if "optimization" in old_path.name.lower():
            target_dir = symbol_dir / "optimization"
        elif timeframe:
            target_dir = symbol_dir / timeframe
        else:
            target_dir = symbol_dir / "general"
        
        target_dir.mkdir(exist_ok=True)
        
        # Move files
        try:
            for file_item in old_path.iterdir():
                target_file = target_dir / file_item.name
                if file_item.is_file():
                    shutil.copy2(file_item, target_file)
                elif file_item.is_dir():
                    shutil.copytree(file_item, target_file, dirs_exist_ok=True)
            
            logger.info(f"Migrated {old_path} -> {target_dir}")
            symbols_migrated.add(symbol)
            
        except Exception as e:
            logger.error(f"Error migrating {old_path}: {e}")
    
    logger.info(f"Migrated {len(symbols_migrated)} symbols: {', '.join(sorted(symbols_migrated))}")

def migrate_portfolio_results():
    """Migrate portfolio-related results"""
    logger.info("Migrating portfolio results...")
    
    # Look for multi-symbol results
    portfolio_patterns = [
        "multi_symbol",
        "portfolio",
        "crypto_majors",
        "defi_tokens"
    ]
    
    search_paths = [
        "results/examples",
        "results/tools",
        "results/research"
    ]
    
    portfolios_migrated = 0
    for search_path in search_paths:
        if Path(search_path).exists():
            for item in Path(search_path).iterdir():
                if item.is_dir():
                    for pattern in portfolio_patterns:
                        if pattern in item.name.lower():
                            # Determine portfolio name
                            if "crypto" in item.name.lower():
                                portfolio_name = "crypto_majors"
                            elif "defi" in item.name.lower():
                                portfolio_name = "defi_tokens"
                            else:
                                portfolio_name = "custom_portfolio_1"
                            
                            target_dir = Path("results/portfolios") / portfolio_name
                            target_dir.mkdir(exist_ok=True)
                            
                            # Move files
                            try:
                                for file_item in item.iterdir():
                                    target_file = target_dir / file_item.name
                                    if file_item.is_file():
                                        shutil.copy2(file_item, target_file)
                                    elif file_item.is_dir():
                                        shutil.copytree(file_item, target_file, dirs_exist_ok=True)
                                
                                logger.info(f"Migrated portfolio {item} -> {target_dir}")
                                portfolios_migrated += 1
                                break
                                
                            except Exception as e:
                                logger.error(f"Error migrating portfolio {item}: {e}")
    
    logger.info(f"Migrated {portfolios_migrated} portfolio results")

def migrate_general_results():
    """Migrate general testing results"""
    logger.info("Migrating general testing results...")
    
    # Look for general testing files
    general_files = []
    
    # Main testing results file
    if Path("results/tools/strategy_tester/testing_results.csv").exists():
        general_files.append(("results/tools/strategy_tester/testing_results.csv", "testing/testing_results.csv"))
    
    # Strategy comparison files
    search_paths = ["results/tools", "results/examples", "results/research"]
    for search_path in search_paths:
        if Path(search_path).exists():
            for item in Path(search_path).rglob("*comparison*"):
                if item.is_file():
                    general_files.append((str(item), f"strategy_comparisons/{item.name}"))
            
            for item in Path(search_path).rglob("*timeframe*"):
                if item.is_file():
                    general_files.append((str(item), f"timeframe_analysis/{item.name}"))
    
    # Migrate general files
    for source, target_rel in general_files:
        target = Path("results/general") / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(source, target)
            logger.info(f"Migrated general file {source} -> {target}")
        except Exception as e:
            logger.error(f"Error migrating {source}: {e}")

def cleanup_old_directories():
    """Remove old directory structure after migration"""
    logger.info("Cleaning up old directories...")
    
    old_dirs = [
        "results/examples",
        "results/tools",
        "results/production", 
        "results/research"
    ]
    
    for old_dir in old_dirs:
        if Path(old_dir).exists():
            try:
                # Check if directory is empty or only contains migrated content
                remaining_files = list(Path(old_dir).rglob("*"))
                if len(remaining_files) <= 5:  # Allow some leftover files
                    shutil.rmtree(old_dir)
                    logger.info(f"Removed old directory: {old_dir}")
                else:
                    logger.warning(f"Directory {old_dir} still contains {len(remaining_files)} files - not removing")
            except Exception as e:
                logger.error(f"Error removing {old_dir}: {e}")

def check_structure():
    """Check and display the current directory structure"""
    logger.info("Checking current directory structure...")
    
    base_dir = Path("results")
    if not base_dir.exists():
        logger.warning("Results directory does not exist")
        return
    
    def count_files_recursive(path):
        """Count files recursively in a directory"""
        if not path.exists():
            return 0
        return len([f for f in path.rglob("*") if f.is_file()])
    
    print("\n" + "="*60)
    print("CURRENT RESULTS DIRECTORY STRUCTURE")
    print("="*60)
    
    # Check symbols
    symbols_dir = base_dir / "symbols"
    if symbols_dir.exists():
        symbols = [d.name for d in symbols_dir.iterdir() if d.is_dir()]
        print(f"\n📊 SYMBOLS ({len(symbols)} symbols):")
        for symbol in sorted(symbols):
            symbol_path = symbols_dir / symbol
            file_count = count_files_recursive(symbol_path)
            timeframes = [d.name for d in symbol_path.iterdir() if d.is_dir() and d.name in ['1h', '4h', '1d']]
            has_opt = (symbol_path / "optimization").exists()
            print(f"  {symbol}: {len(timeframes)} timeframes, optimization: {'✓' if has_opt else '✗'}, {file_count} files")
    
    # Check portfolios
    portfolios_dir = base_dir / "portfolios"
    if portfolios_dir.exists():
        portfolios = [d.name for d in portfolios_dir.iterdir() if d.is_dir()]
        print(f"\n📈 PORTFOLIOS ({len(portfolios)} portfolios):")
        for portfolio in sorted(portfolios):
            portfolio_path = portfolios_dir / portfolio
            file_count = count_files_recursive(portfolio_path)
            print(f"  {portfolio}: {file_count} files")
    
    # Check general
    general_dir = base_dir / "general"
    if general_dir.exists():
        general_subdirs = [d.name for d in general_dir.iterdir() if d.is_dir()]
        print(f"\n🔧 GENERAL ({len(general_subdirs)} categories):")
        for subdir in sorted(general_subdirs):
            subdir_path = general_dir / subdir
            file_count = count_files_recursive(subdir_path)
            print(f"  {subdir}: {file_count} files")
    
    # Total summary
    total_files = count_files_recursive(base_dir)
    print(f"\n📁 TOTAL: {total_files} files in results/")
    print("="*60)

def generate_structure_report():
    """Generate a detailed structure report"""
    logger.info("Generating structure report...")
    
    report = {
        "migration_date": datetime.now().isoformat(),
        "structure_type": "simplified_symbol_portfolio",
        "categories": {}
    }
    
    base_dir = Path("results")
    
    for category in ["symbols", "portfolios", "general"]:
        category_dir = base_dir / category
        if category_dir.exists():
            category_info = {
                "description": SIMPLIFIED_STRUCTURE[category]["description"],
                "subdirectories": [],
                "total_files": 0
            }
            
            for item in category_dir.iterdir():
                if item.is_dir():
                    file_count = len([f for f in item.rglob("*") if f.is_file()])
                    category_info["subdirectories"].append({
                        "name": item.name,
                        "file_count": file_count
                    })
                    category_info["total_files"] += file_count
            
            report["categories"][category] = category_info
    
    # Save report
    report_file = Path("results/structure_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Structure report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Migrate results to simplified structure")
    parser.add_argument('--migrate-all', action='store_true', help='Migrate all results to new structure')
    parser.add_argument('--check-structure', action='store_true', help='Check current directory structure')
    parser.add_argument('--cleanup-old', action='store_true', help='Remove old directory structure')
    parser.add_argument('--generate-report', action='store_true', help='Generate structure report')
    
    args = parser.parse_args()
    
    if args.migrate_all:
        logger.info("Starting migration to simplified structure...")
        create_simplified_structure()
        migrate_symbol_results()
        migrate_portfolio_results()
        migrate_general_results()
        logger.info("Migration completed!")
        
    elif args.check_structure:
        check_structure()
        
    elif args.cleanup_old:
        cleanup_old_directories()
        
    elif args.generate_report:
        generate_structure_report()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 