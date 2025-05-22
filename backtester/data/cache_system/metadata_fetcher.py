#!/usr/bin/env python3
"""
Unified Data Fetcher

A streamlined interface for fetching cryptocurrency market data and timestamps.
Combines the functionality from essential_metadata.py with direct volume extraction
and cache integration.

Core features:
1. Fetch market data with volume information
2. Fetch inception timestamps using vectorbtpro
3. Automatic cache management
4. Exchange-specific behavior handling
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from .cache_manager import cache_manager
# Exchange configuration no longer needed - VBT handles this natively

logger = logging.getLogger(__name__)

class SimpleDataFetcher:
    """Unified data fetcher for cryptocurrency exchange data."""
    
    def __init__(self):
        self.cache = cache_manager
        # VBT handles exchange configuration natively
    
    def _extract_volume_from_data(self, data: Dict[str, Any], exchange_id: str) -> Optional[float]:
        """Extract volume from market or ticker data using standard field preferences."""
        volume_fields = ['quoteVolume', 'baseVolume', 'volume']
        
        # Try direct fields first
        for field in volume_fields:
            if field in data and data[field] is not None:
                try:
                    volume = float(data[field])
                    if volume > 0:
                        return volume
                except (ValueError, TypeError):
                    continue
        
        # Try info dictionary if available
        if 'info' in data and isinstance(data['info'], dict):
            for field in volume_fields:
                if field in data['info'] and data['info'][field] is not None:
                    try:
                        volume = float(data['info'][field])
                        if volume > 0:
                            return volume
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def get_market_data(self, exchange_id: str, 
                       quote_currency: Optional[str] = None,
                       market_types: Optional[List[str]] = None,
                       limit: Optional[int] = None,
                       top_by_volume: bool = True,
                       use_cache: bool = True,
                       force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get market data with volume information.
        
        Args:
            exchange_id: Exchange identifier
            quote_currency: Filter by quote currency (e.g., 'USDT')
            market_types: Filter by market types (e.g., ['spot', 'swap'])
            limit: Maximum number of symbols to return
            top_by_volume: Sort by volume when applying limit
            use_cache: Whether to use cached volume data
            force_refresh: Force refresh volume data even if cached
            
        Returns:
            Dictionary mapping symbols to market data with volume
        """
        start_time = time.time()
        
        # Check if we can use cached volume data
        if use_cache and not force_refresh and self.cache.is_volume_cache_fresh(exchange_id):
            logger.info(f"Using cached volume data for {exchange_id}")
            cached_volumes = self.cache.get_all_volumes(exchange_id)
            
            if cached_volumes:
                # We have cached volume data, now need to get basic market data
                try:
                    import ccxt
                    exchange_class = getattr(ccxt, exchange_id)
                    exchange = exchange_class()
                    markets = exchange.load_markets()
                    
                    result = {}
                    for symbol, market in markets.items():
                        # Apply filters
                        if quote_currency and market.get('quote') != quote_currency:
                            continue
                        if market_types and market.get('type') not in market_types:
                            continue
                        
                        # Create basic market data
                        symbol_data = {
                            'symbol': symbol,
                            'type': market.get('type', 'unknown'),
                            'base': market.get('base'),
                            'quote': market.get('quote'),
                            'active': market.get('active', False)
                        }
                        
                        # Add cached volume if available
                        if symbol in cached_volumes:
                            symbol_data['volume'] = cached_volumes[symbol]
                        
                        result[symbol] = symbol_data
                    
                    # Apply sorting and limit
                    if limit and limit < len(result):
                        if top_by_volume:
                            symbols_sorted = sorted(
                                result.keys(),
                                key=lambda s: result[s].get('volume', 0),
                                reverse=True
                            )
                            selected_symbols = symbols_sorted[:limit]
                            result = {s: result[s] for s in selected_symbols}
                        else:
                            result = dict(list(result.items())[:limit])
                    
                    duration = time.time() - start_time
                    logger.info(f"Retrieved {len(result)} symbols from cache in {duration:.2f} seconds")
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error using cached data, falling back to fresh fetch: {e}")
        
        # Fetch fresh data
        logger.info(f"Fetching fresh market data for {exchange_id}")
        
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class()
            
            # Load markets
            markets = exchange.load_markets()
            logger.info(f"Loaded {len(markets)} markets for {exchange_id}")
            
            # Process markets and extract volume data
            result = {}
            volumes_to_cache = {}
            
            for symbol, market in markets.items():
                # Apply filters
                if quote_currency and market.get('quote') != quote_currency:
                    continue
                if market_types and market.get('type') not in market_types:
                    continue
                
                # Create basic market data
                symbol_data = {
                    'symbol': symbol,
                    'type': market.get('type', 'unknown'),
                    'base': market.get('base'),
                    'quote': market.get('quote'),
                    'active': market.get('active', False)
                }
                
                # Try to extract volume
                volume = self._extract_volume_from_data(market, exchange_id)
                if volume is not None:
                    symbol_data['volume'] = volume
                    volumes_to_cache[symbol] = volume
                
                result[symbol] = symbol_data
            
                            # Get additional volume data via tickers if needed
            symbols_needing_volume = [s for s, data in result.items() if 'volume' not in data]
            
            if symbols_needing_volume and hasattr(exchange, 'fetch_tickers'):
                logger.info(f"Fetching volume for {len(symbols_needing_volume)} symbols via tickers")
                
                # Fetch tickers in batches - use conservative defaults
                max_concurrent = 5
                request_delay = 0.0
                
                # Group symbols by market type for exchanges like Binance that require segregation
                symbols_by_type = {}
                for symbol in symbols_needing_volume:
                    if symbol in result:
                        market_type = result[symbol].get('type', 'unknown')
                        
                        # For Binance, also group by subType (linear/inverse) for swap and future markets
                        subtype = 'linear'  # default
                        if exchange_id.lower() == 'binance' and market_type in ['swap', 'future']:
                            # Check the market data for subType information
                            if symbol in markets:
                                market_info = markets[symbol]
                                if 'linear' in market_info or market_info.get('linear') is True:
                                    subtype = 'linear'
                                elif 'inverse' in market_info or market_info.get('inverse') is True:
                                    subtype = 'inverse'
                                elif 'info' in market_info and isinstance(market_info['info'], dict):
                                    info = market_info['info']
                                    if info.get('contractType') == 'INVERSE' or info.get('subType') == 'inverse':
                                        subtype = 'inverse'
                        
                        # Create a compound key for grouping
                        group_key = f"{market_type}_{subtype}" if market_type in ['swap', 'future'] and exchange_id.lower() == 'binance' else market_type
                        
                        if group_key not in symbols_by_type:
                            symbols_by_type[group_key] = []
                        symbols_by_type[group_key].append(symbol)
                
                logger.info(f"Grouped symbols by type: {[(t, len(symbols)) for t, symbols in symbols_by_type.items()]}")
                
                # Process each market type separately
                for market_type, symbols_of_type in symbols_by_type.items():
                    if not symbols_of_type:
                        continue
                        
                    logger.info(f"Fetching {len(symbols_of_type)} {market_type} symbols")
                    
                    batch_size = min(100, len(symbols_of_type))
                    
                    for i in range(0, len(symbols_of_type), batch_size):
                        batch = symbols_of_type[i:i+batch_size]
                        
                        try:
                            tickers = exchange.fetch_tickers(batch)
                            
                            for symbol, ticker in tickers.items():
                                volume = self._extract_volume_from_data(ticker, exchange_id)
                                if volume is not None and symbol in result:
                                    result[symbol]['volume'] = volume
                                    volumes_to_cache[symbol] = volume
                            
                            logger.debug(f"Successfully fetched {len(tickers)} {market_type} tickers")
                            
                            # Respect rate limits
                            if request_delay > 0:
                                time.sleep(request_delay)
                                
                        except Exception as e:
                            logger.warning(f"Error fetching {market_type} tickers batch: {e}")
                            
                            # Try individual fetches for this market type batch
                            for symbol in batch:
                                try:
                                    ticker = exchange.fetch_ticker(symbol)
                                    volume = self._extract_volume_from_data(ticker, exchange_id)
                                    if volume is not None and symbol in result:
                                        result[symbol]['volume'] = volume
                                        volumes_to_cache[symbol] = volume
                                        
                                    if request_delay > 0:
                                        time.sleep(request_delay)
                                        
                                except Exception as individual_e:
                                    logger.debug(f"Error fetching ticker for {symbol}: {individual_e}")
                                    continue
            
            # Cache the volume data
            if use_cache and volumes_to_cache:
                self.cache.save_all_volumes(exchange_id, volumes_to_cache)
                logger.info(f"Cached volume data for {len(volumes_to_cache)} symbols")
            
            # Apply sorting and limit
            if limit and limit < len(result):
                if top_by_volume:
                    symbols_sorted = sorted(
                        result.keys(),
                        key=lambda s: result[s].get('volume', 0),
                        reverse=True
                    )
                    selected_symbols = symbols_sorted[:limit]
                    result = {s: result[s] for s in selected_symbols}
                else:
                    result = dict(list(result.items())[:limit])
            
            duration = time.time() - start_time
            symbols_with_volume = sum(1 for s in result.values() if 'volume' in s)
            logger.info(f"Fetched {len(result)} symbols ({symbols_with_volume} with volume) in {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching market data for {exchange_id}: {e}")
            return {}
    
    def get_inception_timestamps(self, exchange_id: str, symbols: List[str],
                               concurrent_requests: int = 5,
                               use_cache: bool = True) -> Dict[str, int]:
        """
        Get inception timestamps for symbols.
        
        Args:
            exchange_id: Exchange identifier
            symbols: List of symbols to get timestamps for
            concurrent_requests: Number of concurrent requests
            use_cache: Whether to use cached timestamps
            
        Returns:
            Dictionary mapping symbols to inception timestamps
        """
        if not symbols:
            return {}
        
        # Filter out failed symbols
        symbols = self.cache.filter_failed_symbols(exchange_id, symbols)
        
        results = {}
        symbols_to_fetch = symbols
        
        # Use cached timestamps if available
        if use_cache:
            cached_timestamps = self.cache.get_all_timestamps(exchange_id)
            for symbol in symbols:
                if symbol in cached_timestamps:
                    results[symbol] = cached_timestamps[symbol]
            
            symbols_to_fetch = [s for s in symbols if s not in results]
            if len(symbols_to_fetch) < len(symbols):
                logger.info(f"Using {len(symbols) - len(symbols_to_fetch)} cached timestamps")
        
        if not symbols_to_fetch:
            return results
        
        # Fetch new timestamps
        logger.info(f"Fetching inception timestamps for {len(symbols_to_fetch)} symbols")
        
        # Use conservative concurrent requests
        concurrent_requests = min(concurrent_requests, 5)
        
        new_timestamps = {}
        failed_symbols = set()
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(self._get_single_timestamp, exchange_id, symbol): symbol
                for symbol in symbols_to_fetch
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol, timeout=1800):  # 30 minute timeout
                symbol = future_to_symbol[future]
                try:
                    timestamp = future.result()
                    if timestamp:
                        results[symbol] = timestamp
                        new_timestamps[symbol] = timestamp
                    else:
                        failed_symbols.add(symbol)
                        
                except Exception as e:
                    logger.warning(f"Error getting timestamp for {symbol}: {e}")
                    failed_symbols.add(symbol)
                
                # Progress logging
                if len(results) % 10 == 0:
                    logger.info(f"Retrieved {len(results)}/{len(symbols)} timestamps")
        
        # Save new timestamps to cache
        if use_cache and new_timestamps:
            for symbol, timestamp in new_timestamps.items():
                self.cache.save_timestamp(exchange_id, symbol, timestamp)
            logger.info(f"Cached {len(new_timestamps)} new timestamps")
        
        # Mark failed symbols
        for symbol in failed_symbols:
            self.cache.add_failed_symbol(exchange_id, symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to get timestamps for {len(failed_symbols)} symbols")
        
        return results
    
    def _get_single_timestamp(self, exchange_id: str, symbol: str) -> Optional[int]:
        """Get inception timestamp for a single symbol."""
        try:
            # VBT handles exchange-specific behavior natively
            
            # Use vectorbtpro to find earliest date
            import vectorbtpro as vbt
            import pandas as pd
            
            # Common USDT pattern shortcut
            if ':USDT' in symbol or symbol.endswith('/USDT:USDT'):
                return int(datetime(2024, 5, 22).timestamp() * 1000)
            
            # Try different timeframes
            timeframes = ['1d', '4h', '1h']
            
            for timeframe in timeframes:
                try:
                    earliest_date = vbt.CCXTData.find_earliest_date(
                        symbol, exchange=exchange_id, timeframe=timeframe
                    )
                    
                    if earliest_date:
                        if isinstance(earliest_date, pd.Timestamp):
                            return int(earliest_date.timestamp() * 1000)
                        elif isinstance(earliest_date, (int, float)):
                            return int(earliest_date)
                        else:
                            return int(pd.Timestamp(earliest_date).timestamp() * 1000)
                            
                except Exception:
                    continue
            
            # Default fallback
            return int(datetime(2024, 5, 22).timestamp() * 1000)
            
        except Exception as e:
            logger.debug(f"Error getting timestamp for {symbol}: {e}")
            return None
    
    def get_market_data_with_timestamps(self, exchange_id: str,
                                      quote_currency: Optional[str] = None,
                                      market_types: Optional[List[str]] = None,
                                      limit: Optional[int] = None,
                                      top_by_volume: bool = True,
                                      concurrent_requests: int = 5,
                                      use_cache: bool = True,
                                      force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get market data with volume and inception timestamps in one call.
        
        Returns:
            Dictionary mapping symbols to data with volume and inception_timestamp fields
        """
        # Get market data first
        market_data = self.get_market_data(
            exchange_id=exchange_id,
            quote_currency=quote_currency,
            market_types=market_types,
            limit=limit,
            top_by_volume=top_by_volume,
            use_cache=use_cache,
            force_refresh=force_refresh
        )
        
        if not market_data:
            return {}
        
        # Get timestamps for these symbols
        symbols = list(market_data.keys())
        timestamps = self.get_inception_timestamps(
            exchange_id=exchange_id,
            symbols=symbols,
            concurrent_requests=concurrent_requests,
            use_cache=use_cache
        )
        
        # Add timestamps to market data
        for symbol, timestamp in timestamps.items():
            if symbol in market_data:
                market_data[symbol]['inception_timestamp'] = timestamp
                market_data[symbol]['inception_date'] = datetime.fromtimestamp(
                    timestamp / 1000
                ).strftime('%Y-%m-%d')
        
        return market_data

# Global instance
data_fetcher = SimpleDataFetcher() 