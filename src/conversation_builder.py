"""
Conversation construction from raw tweets.

Builds multi-turn conversations from tweets linked by in_response_to_tweet_id.
"""

import logging
from typing import Optional, dict, list, tuple

import pandas as pd

from src.schemas import ConversationTurn, Message

logger = logging.getLogger(__name__)


def build_message_index(df: pd.DataFrame) -> dict[str, dict]:
    """
    Build an index mapping tweet_id to message data.
    
    Args:
        df: DataFrame with columns: tweet_id, author_id, inbound, created_at, text
        
    Returns:
        Dict mapping tweet_id (as string) to message info
    """
    message_index = {}
    
    for _, row in df.iterrows():
        tweet_id = str(row["tweet_id"])
        message_index[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": row["author_id"],
            "inbound": bool(row["inbound"]),  # True = customer, False = brand
            "created_at": row["created_at"],
            "text": row["text"],
            "in_response_to": str(row["in_response_to_tweet_id"]) if pd.notna(row["in_response_to_tweet_id"]) else None,
        }
    
    return message_index


def get_conversation_thread(
    start_tweet_id: str,
    message_index: dict[str, dict],
    include_previous: bool = True,
    include_subsequent: bool = True,
) -> list[tuple[str, dict]]:
    """
    Get all messages in a conversation thread linked to a tweet.
    
    Walks backwards through in_response_to chain to find root.
    Walks forwards through replies to find all subsequent messages.
    
    Args:
        start_tweet_id: Starting tweet ID
        message_index: Message index from build_message_index
        include_previous: Walk backwards to find originating message
        include_subsequent: Walk forwards to find replies
        
    Returns:
        List of (tweet_id, message_info) tuples in chronological order
    """
    if start_tweet_id not in message_index:
        return []
    
    thread = {}  # Use dict to maintain order and avoid duplicates
    
    # Walk backwards to find the root message
    current = start_tweet_id
    while current and current in message_index:
        thread[current] = message_index[current]
        if include_previous:
            current = message_index[current].get("in_response_to")
        else:
            break
    
    # Walk forwards to find replies (requires second pass through all messages)
    # For efficiency, this is done separately when needed
    
    return sorted(thread.items(), key=lambda x: x[1]["created_at"])


def find_customer_brand_pairs(df: pd.DataFrame, brand_name: str = "AmazonHelp") -> list[dict]:
    """
    Find customer message + brand response pairs in the dataset.
    
    For the selected brand, find:
    1. Brand messages (inbound=False, author_id=brand_name) that reply to customer tweets
    2. The customer's original messages (inbound=True)
    3. Build conversation context for each
    
    Args:
        df: Full dataset DataFrame
        brand_name: Brand to focus on (e.g., "AmazonHelp")
        
    Returns:
        List of conversation pair dictionaries
    """
    logger.info(f"Finding customer-brand pairs for {brand_name}...")
    
    # Get all brand messages (outbound) that reply to someone
    brand_msgs = df[(df["author_id"] == brand_name) & (df["in_response_to_tweet_id"].notna())].copy()
    logger.info(f"  Brand messages with responses: {len(brand_msgs)}")
    
    # Convert tweet_id to int for matching
    brand_msgs = brand_msgs.copy()
    brand_msgs["in_response_to_tweet_id"] = brand_msgs["in_response_to_tweet_id"].astype(int)
    
    # Get all customer messages (inbound)
    customer_tweets = df[df["inbound"] == True].copy()
    customer_tweets_set = set(customer_tweets["tweet_id"].astype(int).tolist())
    logger.info(f"  Total customer tweets: {len(customer_tweets_set)}")
    
    # Build quick lookup for customer message details
    customer_index = {}
    for _, row in customer_tweets.iterrows():
        customer_index[int(row["tweet_id"])] = {
            "author_id": row["author_id"],
            "text": row["text"],
            "created_at": row["created_at"],
            "in_response_to": row["in_response_to_tweet_id"],
        }
    
    pairs = []
    processed = 0
    
    for _, brand_msg in brand_msgs.iterrows():
        processed += 1  # noqa: SIM113
        if processed % 50000 == 0:
            logger.info(f"  Processed {processed:,} brand messages...")
        
        replied_to_id = int(brand_msg["in_response_to_tweet_id"])
        
        # Is this a customer message?
        if replied_to_id in customer_index:
            customer_info = customer_index[replied_to_id]
            
            pair = {
                "customer_tweet_id": replied_to_id,
                "customer_author": customer_info["author_id"],
                "customer_text": customer_info["text"],
                "customer_created_at": customer_info["created_at"],
                "brand_tweet_id": int(brand_msg["tweet_id"]),
                "brand_author": brand_name,
                "brand_text": brand_msg["text"],
                "brand_created_at": brand_msg["created_at"],
                "has_followup": False,
            }
            
            pairs.append(pair)
    
    logger.info(f"Found {len(pairs)} customer-brand pairs")
    return pairs



def filter_usable_pairs(pairs: list[dict], min_text_len: int = 10) -> list[dict]:
    """
    Filter pairs for usability.
    
    Remove:
    - Pairs with empty/null text
    - Text shorter than min_text_len
    - Duplicates
    
    Args:
        pairs: List of conversation pairs
        min_text_len: Minimum text length to keep
        
    Returns:
        Filtered list of pairs
    """
    usable = []
    seen = set()
    
    for pair in pairs:
        # Check for missing text
        if pd.isna(pair["customer_text"]) or pd.isna(pair["brand_text"]):
            continue
        
        # Check text length
        if len(str(pair["customer_text"])) < min_text_len:
            continue
        
        if len(str(pair["brand_text"])) < min_text_len:
            continue
        
        # Check for duplicates
        pair_hash = hash((pair["customer_text"], pair["brand_text"]))
        if pair_hash in seen:
            continue
        
        seen.add(pair_hash)
        usable.append(pair)
    
    logger.info(f"After filtering: {len(usable)} usable pairs (removed {len(pairs) - len(usable)} duplicates/invalid)")
    return usable


def prepare_conversation_data(
    df: pd.DataFrame,
    brand_name: str = "AmazonHelp",
    sample_size: Optional[int] = None,  # noqa: F821
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Prepare conversation data for training and evaluation.
    
    Args:
        df: Full dataset
        brand_name: Brand to focus on
        sample_size: If set, sample this many conversations
        seed: Random seed
        
    Returns:
        Tuple of (conversations_df, statistics_dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: Conversation Construction")
    logger.info("=" * 80)
    
    # Find pairs
    pairs = find_customer_brand_pairs(df, brand_name)
    
    # Filter
    usable = filter_usable_pairs(pairs)
    
    # Convert to DataFrame
    conversations_df = pd.DataFrame(usable)
    
    if len(conversations_df) == 0:
        logger.warning("No usable conversations found!")
        return conversations_df, {"error": "No usable conversations found"}
    
    # Sample if needed
    if sample_size and len(conversations_df) > sample_size:
        conversations_df = conversations_df.sample(n=sample_size, random_state=seed)
        logger.info(f"Sampled {sample_size} conversations")
    
    # Statistics
    stats = {
        "total_pairs": len(pairs),
        "usable_pairs": len(usable),
        "final_count": len(conversations_df),
        "with_followup": conversations_df["has_followup"].sum() if "has_followup" in conversations_df.columns else 0,
        "avg_customer_text_len": conversations_df["customer_text"].str.len().mean() if "customer_text" in conversations_df.columns else 0,
        "avg_brand_text_len": conversations_df["brand_text"].str.len().mean() if "brand_text" in conversations_df.columns else 0,
    }
    
    logger.info("\nConversation Statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.1f}")
        else:
            logger.info(f"  {key}: {val}")
    
    return conversations_df, stats
