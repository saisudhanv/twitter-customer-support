"""Quick analysis of conversation structure."""
import pandas as pd

df = pd.read_csv(r'C:\Users\saisu\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10\twcs\twcs.csv')

# Check AmazonHelp data
amazon = df[df['author_id'] == 'AmazonHelp']
print('AmazonHelp analysis:')
print(f'  Total records: {len(amazon)}')
print(f'  Inbound messages (customer): {amazon["inbound"].sum()}')
print(f'  Outbound messages (brand): {(~amazon["inbound"]).sum()}')
print(f'  Has in_response_to_tweet_id: {amazon["in_response_to_tweet_id"].notna().sum()}')
print(f'  Has response_tweet_id: {amazon["response_tweet_id"].notna().sum()}')

print('\nSample AmazonHelp records:')
print(amazon[['tweet_id', 'author_id', 'inbound', 'text', 'in_response_to_tweet_id', 'response_tweet_id']].head(3).to_string())

# Find conversations involving AmazonHelp
print('\n\nConversation structure analysis:')
print('Records with in_response_to_tweet_id:')
amazon_with_reply = amazon[amazon['in_response_to_tweet_id'].notna()]
if len(amazon_with_reply) > 0:
    print(f'  AmazonHelp tweets that reply to others: {len(amazon_with_reply)}')
    # Find what they're replying to
    replied_to = df[df['tweet_id'].isin(amazon_with_reply['in_response_to_tweet_id'].dropna())]
    print(f'  Tweets AmazonHelp is replying to: {len(replied_to)}')
    print(f'  Authors they reply to: {replied_to["author_id"].nunique()}')
    print(f'  Sample authors: {replied_to["author_id"].unique()[:5]}')
else:
    print('  AmazonHelp does not reply to others')

print('\n\nConversations where customers reply to AmazonHelp:')
amazon_replies = df[df['in_response_to_tweet_id'].isin(amazon['tweet_id'])]
print(f'  Customers who reply to AmazonHelp: {len(amazon_replies)}')
print(f'  Unique authors replying: {amazon_replies["author_id"].nunique()}')
