# import libs
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

from torch_geometric.data import download_url, extract_zip
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.loader import LinkNeighborLoader

url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

movies_path = './ml-latest-small/movies.csv'
ratings_path = './ml-latest-small/ratings.csv'

# load the entire movie data frame into memory
movies_df = pd.read_csv(movies_path, index_col='movieId')

# load the entire ratings data frame into memory
ratings_df = pd.read_csv(ratings_path)

movies_df.head(4)

ratings_df.head(4)

# split genres and convert into indicator variables
genres = movies_df['genres'].str.get_dummies('|')
print(genres[["Action", "Adventure", "Drama", "Horror"]].head())

# use genres as movie input features
movie_feat = torch.from_numpy(genres.values).to(torch.float)

genres

# create a maaping from unique user indices to range[0, num_user_nodes]
unique_user_id = ratings_df['userId'].unique()
unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id))
})


print("Mapping od user IDs to consecutive values:")
print(unique_user_id.head())
print()

# create a maaping from unique movie indices to range[0, num_movie_nodes]
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id))
})

print("Mapping od movie IDs to consecutive values:")
print(unique_movie_id.head())
print()

# preform merge to obtain the edges from users and movies

ratings_user_id = pd.merge(
    ratings_df['userId'],
    unique_user_id,
    left_on='userId',
    right_on='userId',
    how='left'
    )
ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values) # converting to tensor
ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id, left_on='movieId', right_on='movieId', how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)

#print(type(ratings_user_id))
#print(type(ratings_movie_id))


# construct edge index in COO format
edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)

print()
print("final edge indices pointing from users t movies:")
print(edge_index_user_to_movie)

ratings_df

len(unique_user_id)

edge_index_user_to_movie.shape

# creating a hetero dataset
data = HeteroData()

# save node indices
data['user'].node_id = torch.arange(len(unique_user_id)) # creating nodes for users and giving them ids from 0 to length of unique user ID
data['movie'].node_id = torch.arange(len(movies_df))

# save node features
data['user'].x = movie_feat

#save edge indices
data['user', 'rates', 'movie'].edge_index = edge_index_user_to_movie

#reversing edges from movies to users and adding them to the list, so GNN will be able to pass messages in both directions
data = T.ToUndirected()(data)

data

# training the model

# splitting the data into train/ val/ test
transform = T.RandomLinkSplit(  # a lib specifically used to split graph dataset
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=0.2,
    add_negative_train_samples=False,
    edge_types=('user', 'rates', 'movie'),
    rev_edge_types=('movie', 'rev_rates', 'user'),
)
train_data, val_data, test_data = transform(data)

# creating a mini-batch loader to generate sub-graphs that will be input of our GNN
edge_label_index = train_data["user", "rates", "movie"].edge_label_index
edge_label = train_data["user", "rates", "movie"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

data.metadata()

class GNN(torch.nn.Module):
  def __init__(self, hidden_channels):
    super().__init__()
    self.conv1 = SAGEConv(hidden_channels, hidden_channels)
    self.conv2 = SAGEConv(hidden_channels, hidden_channels)

  def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
    x = F.relu(self.conv1(x, edge_index))
    x = self.conv2(x, edge_index)
    return x

# this classifier applies dot-product between sourve and destination nde embeddings to derive edge-level prediction
class Classifier(torch.nn.Module):
  def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
    edge_feat_user = x_user[edge_label_index[0]]  # convert node embeddings to edge-level representation
    edge_feat_movie = x_movie[edge_label_index[1]]

    return(edge_feat_user * edge_feat_movie).sum(dim=-1)  # applying the dot prduct to get a prediction per supervision edge


class Model(torch.nn.Module):
  def __init__(self, hidden_channels):
    super().__init__()
    self.movie_lin = torch.nn.Linear(20, hidden_channels)  # 20 -> size of feature vector of the movies was 20 too
    self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)  # learning the embedding matrices for users and mvies(cuz dataset doesn't have that many features)
    self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)  # in fact, we're creating an embedding layer that will be used instead of feature vector

    self.gnn = GNN(hidden_channels)   # instantiate homogenous GNN

    self.gnn = to_hetero(self.gnn, metadata=data.metadata())  # converting  gnn model into a heterogenous variant, we need to have a heterogenous model for a heterogenous dataset
    # meta data contains nodes and relationships info

    self.classifier = Classifier()

  def forward(self, data: HeteroData) -> Tensor:
    x_dict = {  # defining a dictionary
        "user": self.user_emb(data["user"].node_id),
        "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id), # combo of linear layer and embedding layer
    }
    x_dict = self.gnn(x_dict, data.edge_index_dict) # holding feature matrices of all node types & edge_index_dict = holding all edges indices of all edge types
    pred = self.classifier(
        x_dict["user"],
        x_dict["movie"],
        data["user", "rates", "movie"].edge_label_index
    )
    return pred

model = Model(hidden_channels=64)

# train

import tqdm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # switching to gpu
print(f"device:  '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6): # 5 epochs
  total_loss = total_examples = 0
  for sampled_data in tqdm.tqdm(train_loader):
    optimizer.zero_grad()
    sampled_data.to(device)
    pred = model(sampled_data)
    ground_truth = sampeled_data["user", "rates", "movie"].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
  print(f"epoch: {epoch:03d}, loss: {total_loss / total_examples:.4f}")

# validation

edge_label_index = val_data["user", "rates", "movie"].edge_label_index
edge_label = val_data["user". "rates", "movie"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=3*128,
    shuffle=False
)
sampeled_data = next(iter(val_loader))

# evaluation

from sklearn.metrics import roc_auc_score
preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
  with torch.no_grad():
    sampled_data.to(device)
    preds.append(model(sampled_data))
    ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)

pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truths, pred)
print()
print(f"validation AUC: {auc:.4f}")











