# Synthetic Social Media Data Generator

A Python tool that generates realistic synthetic social media data for analysis, testing, and demonstration purposes.

## Features

- Generates user profiles with demographic information
- Creates follower-followed relationships between users
- Produces synthetic posts with timestamps and content types
- Simulates user activities (posts, likes, comments, shares)
- Calculates engagement metrics for each user
- Performs network analysis to identify communities and influencers
- Visualizes the social network structure

## Generated Data Files

The tool generates the following CSV files:

- **user_profiles.csv**: User demographic information
- **user_relationships.csv**: Follower-followed relationships
- **posts.csv**: Information about each post
- **user_activities.csv**: Logs of all user activities
- **engagement_metrics.csv**: Aggregated engagement scores
- **influence_metrics.csv**: Influence metrics for each user
- **communities.csv**: Community assignments for users
- **top_influencers.txt**: List of top 10 influencers
- **network_visualization.png**: Visual representation of the social network

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-social-media-data-generator.git
cd synthetic-social-media-data-generator

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## Usage

1. Navigate to the project directory:
   ```bash
   cd synthetic-social-media-data-generator
   ```

2. Run the script:
   ```bash
   python generate_synthetic_data.py
   ```

   The script will automatically create and use a `./synthetic_data` directory to store the generated files. No input is required.

3. Alternatively, you can specify a custom output directory in your code:
   ```python
   # Import the function
   from generate_synthetic_data import generate_synthetic_data
   
   # Call with custom output directory
   generate_synthetic_data(output_dir="./my_custom_directory")
   ```

## Data Schema

### user_profiles.csv
- `user_id`: Unique identifier for each user
- `age`: Age of the user
- `location`: Geographical location of the user
- `interests`: List of user interests

### user_relationships.csv
- `follower_id`: User ID of the follower
- `followed_id`: User ID of the user being followed

### posts.csv
- `post_id`: Unique identifier for each post
- `user_id`: User ID of the poster
- `timestamp`: Date and time when the post was made
- `content_type`: Type of content (text, image, video)
- `hashtags`: List of hashtags associated with the post
- `media_attachment`: Media file attached to the post (if any)
- `quality_score`: Quality rating of the post (1-5)

### user_activities.csv
- `user_id`: User ID performing the activity
- `activity_type`: Type of activity (post, like, comment, share)
- `timestamp`: Date and time of the activity
- `content_type`: Type of content involved in the activity (if applicable)
- `hashtags`: List of hashtags involved in the activity (if applicable)
- `media_attachment`: Media file involved in the activity (if applicable)
- `post_id`: Reference to the associated post (if applicable)

### engagement_metrics.csv
- `user_id`: User ID
- `total_posts`: Total number of posts made
- `total_likes`: Total number of likes given
- `total_comments`: Total number of comments made
- `total_shares`: Total number of shares made
- `engagement_score`: Calculated engagement score

### influence_metrics.csv
- `followed_id`: User ID being followed
- `num_followers`: Number of followers the user has
- `user_id`: User ID
- `total_posts`: Total number of posts made
- `total_likes`: Total number of likes given
- `total_comments`: Total number of comments made
- `total_shares`: Total number of shares made
- `engagement_score`: Engagement score
- `influence_score`: Calculated influence score
- `pagerank`: PageRank centrality score
- `betweenness_centrality`: Betweenness centrality score
- `location`: User location
- `age`: User age
- `interests`: User interests
- `community_id`: Community assignment

### communities.csv
- `user_id`: User ID
- `community_id`: Identifier for the community the user belongs to

## Requirements

- Python 3.6+
- pandas
- numpy
- faker
- networkx
- python-louvain (community)
- matplotlib

## License

MIT
