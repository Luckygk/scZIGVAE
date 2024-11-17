import time
from Preprocessing import preprocess_raw_data1, read_data, process_data
from model import ZIVGAEncoder, ZIVGADecoder
import Utils
import torch
import torch.nn.functional as F
import torch_geometric.transforms as Trans
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import argparse
import pandas as pd


def pretrain(model, optimizer, train_data,  true_label, device):
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
    for epoch in range(args['max_epoch']):
        model.train()
        z = model.encode(x, edge_index)
        reconstruction_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        L_vgaa = args['re_loss'] * reconstruction_loss + (1 / train_data.num_nodes) * model.kl_loss()
        x_bar,mean,disp,pi=model(z)
        decoder_loss = F.mse_loss(x_bar, x)
        loss = args['vgaa_loss'] * L_vgaa + decoder_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z = model.encode(x, edge_index)
            kmeans = KMeans(n_clusters=args['num_clusters'], n_init=20).fit(z.detach().numpy())
            torch.save(
                model.state_dict(),
                f"./Pretrain/{args['name']}.pkl"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain")
    parser.add_argument("--dataset", type=str, default="Pollen")
    parser.add_argument("--name", type=str, default="Pollen_counts")
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_clusters", default=11, type=int)
    parser.add_argument("--num_heads", default=[3,3,3,3], type=int)
    parser.add_argument('--k', type=int, default=10, help='K of neighbors PKNN')
    parser.add_argument('--decoder_dim1', type=int, default=128,
                        help='First hidden dimension for the neural network decoder')
    parser.add_argument('--dropout', type=float, default=[0.2, 0.2], help='Dropout for each layer')
    parser.add_argument('--hidden_dims', type=int, default=[128, 128], help='Output dimension for each hidden layer.')
    parser.add_argument('--latent_dim', type=int, default=50, help='output dimension for node embeddings')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--re_loss', type=float, default=1)
    parser.add_argument('--vgaa_loss', type=float, default=0.1)
    parser.add_argument('--zinb_loss', type=float, default=0.1)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(args)
    df = pd.read_csv('Dataset/{}/{}_labels.csv'.format(args['dataset'], args['name']))
    true_lab = df['label'].values
    adata = read_data('Dataset/{}/{}.csv'.format(args['dataset'], args['name']), file_type='csv')
    adata_hvg , X_hvg = process_data(adata)
    np.save('Process/{}.npy'.format(args['datasetName']), X_hvg)
    distances, neighbors, cutoff, edgelist = Utils.get_edgelist_PKNN(datasetName=args['name'], X_hvg=X_hvg, k=args['k'],
                                                                     type='PKNN')
    edges = Utils.load_separate_graph_edgelist('Process/{}_edgelist.txt'.format(args['name']))
    data_obj = Utils.create_graph(edges, X_hvg)
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    test_split = args['test_split']
    try:
        transform = Trans.RandomLinkSplit(num_test=test_split,
                                          is_undirected=True, add_negative_train_samples=True,
                                          split_labels=True)
        train_data, test_data = transform(data_obj)
    except IndexError as ie:
        print()
        print('Need transpose.')

    num_features = data_obj.num_features
    heads = args['num_heads']
    num_heads = {}
    num_heads['first'] = heads[0]
    num_heads['second'] = heads[1]
    num_heads['mean'] = heads[2]
    num_heads['std'] = heads[3]
    hidden_dims = args['hidden_dims']
    latent_dim = args['latent_dim']
    dropout = args['dropout']
    num_clusters = args['num_clusters']

    encoder = ZIVGAEncoder(
        in_channels=num_features,
        num_heads=num_heads,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        concat={'first': True, 'second': False},
    )
    model = ZIVGADecoder(encoder=encoder, decoder_nn_dim1=args['decoder_dim1'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    pretrain(model=model,
             optimizer=optimizer,
             train_data=train_data,
             true_label=true_lab,
             device=device)
