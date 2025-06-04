#!/usr/bin/env python3
"""
Synthetic Social Media Data Generator

This script generates synthetic social media data including user profiles,
relationships, posts, activities, engagement metrics, and community analysis.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from faker import Faker
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import json

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    return True

def generate_user_profiles(num_users=1000):
    """Generate synthetic user profiles"""
    print("Generating user profiles...")
    
    # List of possible interests
    interests_list = [
        "technology", "sports", "music", "movies", "books", "travel", 
        "food", "fashion", "photography", "gaming", "fitness", "art", 
        "politics", "science", "education", "environment", "health", 
        "business", "finance", "pets"
    ]
    
    users = []
    for user_id in range(1, num_users + 1):
        # Generate random number of interests (1-5)
        num_interests = random.randint(1, 5)
        user_interests = random.sample(interests_list, num_interests)
        
        user = {
            'user_id': user_id,
            'age': random.randint(18, 70),
            'location': fake.city(),
            'interests': ','.join(user_interests)
        }
        users.append(user)
    
    return pd.DataFrame(users)

def generate_relationships(user_profiles, avg_connections=20):
    """Generate follower-followed relationships"""
    print("Generating user relationships...")
    
    num_users = len(user_profiles)
    relationships = []
    
    # For each user, generate random followers
    for user_id in user_profiles['user_id']:
        # Number of people this user follows (random around avg_connections)
        num_follows = max(1, int(np.random.normal(avg_connections, avg_connections/3)))
        num_follows = min(num_follows, num_users - 1)  # Can't follow more than available users
        
        # Select random users to follow (excluding self)
        potential_follows = [u for u in user_profiles['user_id'] if u != user_id]
        followed_ids = random.sample(potential_follows, num_follows)
        
        for followed_id in followed_ids:
            relationships.append({
                'follower_id': user_id,
                'followed_id': followed_id
            })
    
    return pd.DataFrame(relationships)

def generate_posts(user_profiles, start_date=None, end_date=None, avg_posts_per_user=10):
    """Generate synthetic posts"""
    print("Generating posts...")
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    content_types = ["text", "image", "video"]
    content_type_weights = [0.6, 0.3, 0.1]  # Probability weights
    
    # List of possible hashtags
    hashtags_list = [
        "trending", "viral", "love", "instagood", "photooftheday", "fashion", "beautiful", 
        "happy", "cute", "tbt", "like4like", "followme", "picoftheday", "follow", 
        "me", "selfie", "summer", "art", "instadaily", "friends", "repost", "nature", 
        "girl", "fun", "style", "smile", "food", "instalike", "likeforlike", "family"
    ]
    
    posts = []
    post_id = 1
    
    for user_id in user_profiles['user_id']:
        # Number of posts for this user (random around avg_posts_per_user)
        num_posts = max(1, int(np.random.normal(avg_posts_per_user, avg_posts_per_user/3)))
        
        for _ in range(num_posts):
            # Random timestamp between start_date and end_date
            timestamp = start_date + (end_date - start_date) * random.random()
            
            # Random content type based on weights
            content_type = random.choices(content_types, weights=content_type_weights)[0]
            
            # Random number of hashtags (0-5)
            num_hashtags = random.randint(0, 5)
            post_hashtags = random.sample(hashtags_list, num_hashtags) if num_hashtags > 0 else []
            
            # Media attachment for image/video posts
            media_attachment = None
            if content_type == "image":
                media_attachment = f"image_{fake.uuid4()}.jpg"
            elif content_type == "video":
                media_attachment = f"video_{fake.uuid4()}.mp4"
            
            # Quality score (1-5)
            quality_score = random.randint(1, 5)
            
            post = {
                'post_id': post_id,
                'user_id': user_id,
                'timestamp': timestamp,
                'content_type': content_type,
                'hashtags': ','.join(post_hashtags) if post_hashtags else None,
                'media_attachment': media_attachment,
                'quality_score': quality_score
            }
            
            posts.append(post)
            post_id += 1
    
    return pd.DataFrame(posts)

def generate_user_activities(user_profiles, posts_df, avg_activities_per_post=5):
    """Generate user activities (likes, comments, shares)"""
    print("Generating user activities...")
    
    activity_types = ["like", "comment", "share"]
    activity_weights = [0.7, 0.2, 0.1]  # Probability weights
    
    activities = []
    
    # First, add all posts as activities
    for _, post in posts_df.iterrows():
        activity = {
            'user_id': post['user_id'],
            'activity_type': 'post',
            'timestamp': post['timestamp'],
            'content_type': post['content_type'],
            'hashtags': post['hashtags'],
            'media_attachment': post['media_attachment'],
            'post_id': post['post_id']
        }
        activities.append(activity)
    
    # Then generate other activities (likes, comments, shares)
    for _, post in posts_df.iterrows():
        # Number of activities for this post
        num_activities = max(0, int(np.random.normal(avg_activities_per_post, avg_activities_per_post/3)))
        
        # Select random users who might interact with this post (excluding post author)
        potential_users = [u for u in user_profiles['user_id'] if u != post['user_id']]
        
        # If we have more activities than potential users, cap it
        num_activities = min(num_activities, len(potential_users))
        
        if num_activities > 0:
            interacting_users = random.sample(potential_users, num_activities)
            
            for user_id in interacting_users:
                # Random activity type based on weights
                activity_type = random.choices(activity_types, weights=activity_weights)[0]
                
                # Activity timestamp is after post timestamp
                time_delta = random.random() * timedelta(days=2).total_seconds()
                activity_timestamp = post['timestamp'] + timedelta(seconds=time_delta)
                
                activity = {
                    'user_id': user_id,
                    'activity_type': activity_type,
                    'timestamp': activity_timestamp,
                    'content_type': post['content_type'] if activity_type == 'share' else None,
                    'hashtags': post['hashtags'] if activity_type == 'share' else None,
                    'media_attachment': post['media_attachment'] if activity_type == 'share' else None,
                    'post_id': post['post_id']
                }
                
                activities.append(activity)
    
    return pd.DataFrame(activities)

def calculate_engagement_metrics(activities_df):
    """Calculate engagement metrics for each user"""
    print("Calculating engagement metrics...")
    
    # Group by user_id and activity_type and count
    user_activity_counts = activities_df.groupby(['user_id', 'activity_type']).size().unstack(fill_value=0)
    
    # Make sure all activity types are represented
    for activity_type in ['post', 'like', 'comment', 'share']:
        if activity_type not in user_activity_counts.columns:
            user_activity_counts[activity_type] = 0
    
    # Rename columns for clarity
    engagement_metrics = user_activity_counts.rename(columns={
        'post': 'total_posts',
        'like': 'total_likes',
        'comment': 'total_comments',
        'share': 'total_shares'
    })
    
    # Calculate engagement score (weighted sum of activities)
    engagement_metrics['engagement_score'] = (
        engagement_metrics['total_posts'] * 5 +
        engagement_metrics['total_likes'] * 1 +
        engagement_metrics['total_comments'] * 3 +
        engagement_metrics['total_shares'] * 4
    )
    
    # Reset index to make user_id a column
    engagement_metrics = engagement_metrics.reset_index()
    
    return engagement_metrics

def analyze_network(relationships_df, engagement_metrics_df, user_profiles_df):
    """Analyze the social network to identify communities and influencers"""
    print("Analyzing social network...")
    
    # Create a directed graph from relationships
    G = nx.from_pandas_edgelist(
        relationships_df, 
        source='follower_id', 
        target='followed_id', 
        create_using=nx.DiGraph()
    )
    
    # Calculate PageRank (measure of influence)
    pagerank = nx.pagerank(G)
    
    # Calculate betweenness centrality (limited to save computation time)
    if len(G) > 1000:
        print("Large network detected, using approximate betweenness centrality...")
        betweenness = nx.betweenness_centrality(G, k=100)  # Approximate with k samples
    else:
        betweenness = nx.betweenness_centrality(G)
    
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    
    # Detect communities using Louvain method
    communities = community_louvain.best_partition(G_undirected)
    
    # Create dataframe for community assignments
    community_df = pd.DataFrame({
        'user_id': list(communities.keys()),
        'community_id': list(communities.values())
    })
    
    # Count followers for each user
    follower_counts = relationships_df.groupby('followed_id').size().reset_index(name='num_followers')
    
    # Merge follower counts with engagement metrics
    influence_df = follower_counts.merge(engagement_metrics_df, left_on='followed_id', right_on='user_id', how='outer')
    
    # Fill NaN values
    influence_df['num_followers'] = influence_df['num_followers'].fillna(0)
    
    # Add PageRank and betweenness centrality
    influence_df['pagerank'] = influence_df['followed_id'].map(pagerank).fillna(0)
    influence_df['betweenness_centrality'] = influence_df['followed_id'].map(betweenness).fillna(0)
    
    # Calculate influence score (weighted combination of metrics)
    influence_df['influence_score'] = (
        influence_df['num_followers'] * 0.4 +
        influence_df['pagerank'] * 10000 +  # Scale up PageRank which is typically small
        influence_df['betweenness_centrality'] * 1000 +  # Scale up betweenness which is typically small
        influence_df['engagement_score'] * 0.01  # Scale down engagement score which can be large
    )
    
    # Merge with user profiles to get demographic info
    influence_df = influence_df.merge(user_profiles_df, left_on='followed_id', right_on='user_id', how='left', suffixes=('', '_y'))
    
    # Merge with community assignments
    influence_df = influence_df.merge(community_df, on='user_id', how='left')
    
    return influence_df, community_df

def visualize_network(relationships_df, influence_df, output_dir):
    """Create visualization of the social network"""
    print("Creating network visualization...")
    
    # Create a directed graph from relationships
    G = nx.from_pandas_edgelist(
        relationships_df, 
        source='follower_id', 
        target='followed_id', 
        create_using=nx.DiGraph()
    )
    
    # If the network is too large, sample it for visualization
    if len(G) > 100:
        # Get top influencers
        top_influencers = influence_df.nlargest(20, 'influence_score')['user_id'].tolist()
        
        # Create a subgraph with top influencers and their direct connections
        nodes_to_keep = set(top_influencers)
        for node in top_influencers:
            if node in G:
                nodes_to_keep.update(G.predecessors(node))
                nodes_to_keep.update(G.successors(node))
        
        G = G.subgraph(nodes_to_keep)
    
    # Set up the plot
    plt.figure(figsize=(12, 12))
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(G, k=0.3)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=True, arrowsize=10)
    
    # Save the figure
    plt.title("Social Network Visualization (Sample)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_visualization.png'), dpi=300)
    plt.close()

def generate_synthetic_data(output_dir=None):
    """Main function to generate all synthetic social media data"""
    print("Starting Synthetic Social Media Data Generator...")
    
    # Use default output directory if none is provided
    if output_dir is None:
        output_dir = "./synthetic_data"
        print(f"Using default output directory: {output_dir}")
    
    # Create directory if it doesn't exist
    if not create_directory(output_dir):
        print("Failed to create or access the specified directory. Exiting.")
        return
    
    # Parameters
    num_users = 1000
    avg_connections = 20
    avg_posts_per_user = 10
    avg_activities_per_post = 5
    
    # Generate data
    user_profiles_df = generate_user_profiles(num_users)
    relationships_df = generate_relationships(user_profiles_df, avg_connections)
    
    # Generate posts from the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    posts_df = generate_posts(user_profiles_df, start_date, end_date, avg_posts_per_user)
    
    # Generate activities
    activities_df = generate_user_activities(user_profiles_df, posts_df, avg_activities_per_post)
    
    # Calculate engagement metrics
    engagement_metrics_df = calculate_engagement_metrics(activities_df)
    
    # Analyze network
    influence_df, community_df = analyze_network(relationships_df, engagement_metrics_df, user_profiles_df)
    
    # Save data to CSV files
    user_profiles_df.to_csv(os.path.join(output_dir, 'user_profiles.csv'), index=False)
    relationships_df.to_csv(os.path.join(output_dir, 'user_relationships.csv'), index=False)
    posts_df.to_csv(os.path.join(output_dir, 'posts.csv'), index=False)
    activities_df.to_csv(os.path.join(output_dir, 'user_activities.csv'), index=False)
    engagement_metrics_df.to_csv(os.path.join(output_dir, 'engagement_metrics.csv'), index=False)
    influence_df.to_csv(os.path.join(output_dir, 'influence_metrics.csv'), index=False)
    community_df.to_csv(os.path.join(output_dir, 'communities.csv'), index=False)
    
    # Save top influencers to text file
    top_influencers = influence_df.nlargest(10, 'influence_score')['user_id'].tolist()
    with open(os.path.join(output_dir, 'top_influencers.txt'), 'w') as f:
        for influencer in top_influencers:
            f.write(f"{influencer}\n")
    
    # Create network visualization
    visualize_network(relationships_df, influence_df, output_dir)
    
    print(f"Data generation complete! Files saved to {output_dir}")
    print(f"Generated {len(user_profiles_df)} user profiles")
    print(f"Generated {len(relationships_df)} relationships")
    print(f"Generated {len(posts_df)} posts")
    print(f"Generated {len(activities_df)} activities")
    print("Network analysis completed")

if __name__ == "__main__":
    generate_synthetic_data()
