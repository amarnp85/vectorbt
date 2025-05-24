#!/usr/bin/env python3
"""Data Storage Implementation

Uses VectorBT Pro's built-in pickle persistence for optimal VBT data preservation.
All VBT metadata, functionality, and features are maintained.
"""

import os
import logging
from typing import List, Dict, Union, Optional, Any
from datetime import datetime
import pandas as pd
import vectorbtpro as vbt

logger = logging.getLogger(__name__)

# Default storage directory
DEFAULT_STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'vbt_data')

class DataStorage:
    """
    VBT data storage using VectorBT Pro's native pickle persistence.
    
    Features:
    - Complete VBT metadata preservation
    - Native VBT functionality maintained
    - Efficient compression with blosc
    - Simple and reliable storage/retrieval
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize VBT data storage."""
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        
        # Create main storage directory (no subdirectories)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        logger.info(f"Data Storage initialized: {self.storage_dir}")
    
    def _get_pickle_path(self, exchange_id: str, timeframe: str, market_type: str = 'spot') -> str:
        """Get path to pickle file for exchange-timeframe-market."""
        base_filename = f"{exchange_id.lower()}_{market_type.lower()}_{timeframe}.pickle"
        base_path = os.path.join(self.storage_dir, base_filename)
        
        # Check if compressed version exists first
        compressed_path = base_path + '.blosc'
        if os.path.exists(compressed_path):
            return compressed_path
        
        return base_path
    
    def save_data(self, data: vbt.Data, exchange_id: str, timeframe: str, market_type: str = 'spot') -> bool:
        """
        Save VBT data object using pickle storage.
        
        Args:
            data: VBT Data object to save
            exchange_id: Exchange identifier
            timeframe: Timeframe identifier
            market_type: Market type ('spot' or 'swap')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            base_filename = f"{exchange_id.lower()}_{market_type.lower()}_{timeframe}.pickle"
            pickle_path = os.path.join(self.storage_dir, base_filename)
            
            # Use VBT's native save method with compression
            data.save(pickle_path, compression='blosc')
            
            # Check what file was actually created
            if os.path.exists(pickle_path):
                logger.debug(f"Created uncompressed file: {pickle_path}")
            if os.path.exists(pickle_path + '.blosc'):
                logger.debug(f"Created compressed file: {pickle_path}.blosc")
            
            logger.info(f"Saved VBT data to pickle: {pickle_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving VBT data to pickle: {e}")
            return False
    
    def load_data(self, exchange_id: str, timeframe: str, symbols: Optional[List[str]] = None, market_type: str = 'spot') -> Optional[vbt.Data]:
        """
        Load VBT data object from pickle storage.
        
        Args:
            exchange_id: Exchange identifier
            timeframe: Timeframe identifier
            symbols: Optional symbol filter (applied after loading)
            market_type: Market type ('spot' or 'swap')
            
        Returns:
            VBT Data object or None if not found
        """
        try:
            pickle_path = self._get_pickle_path(exchange_id, timeframe, market_type)
            
            if not os.path.exists(pickle_path):
                logger.debug(f"Pickle file not found: {pickle_path}")
                return None
            
            # Use VBT's native load method
            data = vbt.Data.load(pickle_path)
            
            # Apply symbol filter if specified
            if symbols and hasattr(data, 'symbols'):
                # Filter to requested symbols that exist in the data
                available_symbols = [s for s in symbols if s in data.symbols]
                if available_symbols and len(available_symbols) < len(data.symbols):
                    # Create filtered data by selecting only the requested symbols
                    data = data.select(available_symbols)
            
            logger.debug(f"Loaded VBT data from pickle: {pickle_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading VBT data from pickle: {e}")
            return None
    
    def data_exists(self, exchange_id: str, timeframe: str, market_type: str = 'spot') -> bool:
        """Check if data exists for exchange-timeframe-market."""
        pickle_path = self._get_pickle_path(exchange_id, timeframe, market_type)
        return os.path.exists(pickle_path)
    
    def get_stored_symbols(self, exchange_id: str, timeframe: str, market_type: str = 'spot') -> List[str]:
        """Get list of symbols stored for exchange-timeframe-market."""
        try:
            data = self.load_data(exchange_id, timeframe, market_type=market_type)
            if data is not None and hasattr(data, 'symbols'):
                return list(data.symbols)
            return []
        except Exception as e:
            logger.error(f"Error getting stored symbols: {e}")
            return []
    
    def get_date_range(self, exchange_id: str, timeframe: str, market_type: str = 'spot') -> Optional[tuple]:
        """Get date range for stored data."""
        try:
            data = self.load_data(exchange_id, timeframe, market_type=market_type)
            if data is not None and hasattr(data, 'wrapper'):
                start_date = data.wrapper.index[0]
                end_date = data.wrapper.index[-1]
                return (start_date, end_date)
            return None
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            return None
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get comprehensive storage summary."""
        summary = {
            'storage_dir': self.storage_dir,
            'pickle_files': 0,
            'total_size_mb': 0.0,
            'files': {}
        }
        
        if not os.path.exists(self.storage_dir):
            return summary
        
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.pickle') or filename.endswith('.pickle.blosc'):
                    filepath = os.path.join(self.storage_dir, filename)
                    
                    # Parse filename: exchange_market_timeframe.pickle(.blosc)
                    if filename.endswith('.pickle.blosc'):
                        parts = filename[:-13].split('_')  # Remove .pickle.blosc
                    else:
                        parts = filename[:-7].split('_')  # Remove .pickle
                    if len(parts) >= 3:
                        exchange = parts[0]
                        market_type = parts[1]
                        timeframe = '_'.join(parts[2:])
                        
                        # Get file details
                        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                        symbols = self.get_stored_symbols(exchange, timeframe, market_type)
                        date_range = self.get_date_range(exchange, timeframe, market_type)
                        
                        summary['files'][filename] = {
                            'exchange': exchange,
                            'market_type': market_type,
                            'timeframe': timeframe,
                            'type': 'VBT Data (pickle)',
                            'size_mb': round(file_size, 2),
                            'symbol_count': len(symbols),
                            'symbols': symbols[:10] if symbols else [],  # First 10 symbols
                            'date_range': (str(date_range[0]), str(date_range[1])) if date_range else None
                        }
                        
                        summary['pickle_files'] += 1
                        summary['total_size_mb'] += file_size
                        
        except Exception as e:
            logger.error(f"Error getting storage summary: {e}")
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        return summary

# Global storage instance
data_storage = DataStorage()

# Legacy function names for compatibility
def save_data_pickle(data: vbt.Data, exchange_id: str, timeframe: str, market_type: str = 'spot') -> bool:
    """Legacy function name - redirects to main save_data."""
    return data_storage.save_data(data, exchange_id, timeframe, market_type)

def load_data_pickle(exchange_id: str, timeframe: str, symbols: Optional[List[str]] = None, market_type: str = 'spot') -> Optional[vbt.Data]:
    """Legacy function name - redirects to main load_data."""
    return data_storage.load_data(exchange_id, timeframe, symbols, market_type)

if __name__ == "__main__":
    print("\n=== VBT Data Storage ===")
    print("Using VectorBT Pro's native pickle persistence")
    print("All VBT metadata and functionality preserved")
    print("=" * 40) 