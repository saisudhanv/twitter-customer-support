"""Debug conversation pair finding."""
import pandas as pd

df = pd.read_csv(r'C:\Users\saisu\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10\twcs\twcs.csv')

# Get AmazonHelp messages
amazon = df[df['author_id'] == 'AmazonHelp'].copy()
print(f"Total AmazonHelp messages: {len(amazon)}")

# Check which ones have in_response_to_tweet_id
amazon_with_reply = amazon[amazon['in_response_to_tweet_id'].notna()]
print(f"AmazonHelp messages with in_response_to_tweet_id: {len(amazon_with_reply)}")

# Check the inbound status of what they're replying to
if len(amazon_with_reply) > 0:
    sample_ids = amazon_with_reply['in_response_to_tweet_id'].head(5).astype(int).tolist()
    print(f"\nSample tweet IDs AmazonHelp is replying to: {sample_ids}")
    
    # Check these tweets
    replied_to = df[df['tweet_id'].isin(sample_ids)]
    print(f"\nDetailed view of what AmazonHelp is replying to:")
    for _, row in replied_to.iterrows():
        print(f"  Tweet ID {row['tweet_id']}: inbound={row['inbound']}, author={row['author_id']}")
    
    # Check if they're all inbound
    all_replied_to = df[df['tweet_id'].isin(amazon_with_reply['in_response_to_tweet_id'].astype(int))]
    print(f"\nAll tweets AmazonHelp replies to:")
    print(f"  Total: {len(all_replied_to)}")
    print(f"  Inbound (customer): {all_replied_to['inbound'].sum()}")
    print(f"  Outbound (brand): {(~all_replied_to['inbound']).sum()}")
