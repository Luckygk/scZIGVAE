import argparse
import time
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Utils import *
import pandas as pd
from model import ZIVGAEncoder, ZIVGADecoder
import torch_geometric.transforms as Trans
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class DEC(nn.Module):
    def __init__(self, model, latent_dims, num_clusters, alpha=1):
        super(DEC, self).__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.model = model
        self.model.load_state_dict(
            torch.load('./Pretrain/{}/{}.pkl'.format(args['datasetName'], args['datasetName']), map_location='cpu'))
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, latent_dims))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def forward(self, x, edge_index):
        z = self.model.encode(x, edge_index)
        q = self.get_Q(z)
        return z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train(dec, optimizer,train_data,  device, true_label):
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
    with torch.no_grad():
        z = dec.model.encode(x, edge_index)
    kmeans = KMeans(n_clusters=args['num_clusters'], n_init=20)
    y_pred = kmeans.fit_predict(z.data.detach().numpy())
    y_pred_last = np.copy(y_pred)
    dec.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    for epoch in range(args['max_epoch']):
        if epoch % args['update_interval'] == 0:
            z, Q = dec(x, edge_index)
            q = Q.detach().data.cpu().numpy().argmax(1)
            y_pred = np.copy(q)
            ari = adjusted_rand_score(true_label, q)
            if res_ari <= ari:
                res_ari = ari
                np.save(
                    f"/{args['name']}/cluster.npy", q)
                np.save(
                    f"/{args['name']}/embedding_data.npy",z.detach().numpy())
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and delta_label < 1e-3:
                print('delta_label=={}'.format(delta_label))
                print('Reach tolerance threshold,Stopping training.')
                break
        z, q = dec(x, edge_index)
        p = target_distribution(q.detach())
        clu_loss = F.kl_div(q.log(), p, reduction='batchmean')
        vgaa_loss = 0.1 * dec.model.recon_loss(z, train_data.pos_edge_label_index) + (
                1 / train_data.num_nodes) * dec.model.kl_loss()
        x_bar, mean, disp, pi = dec.model(z)
        zinb_loss_value = dec.model.zinb_loss(x, mean=mean, disp=disp, pi=pi)
        loss = args['clu_loss'] * clu_loss + args['vgaa_loss'] * vgaa_loss + args['zinb_loss'] * zinb_loss_value
        decoder_loss = F.mse_loss(x_bar, x)
        loss += decoder_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('final ARI={:.4f}'.format(res_ari))

if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='train', description='scZIGVAE train')
    parse.add_argument("--dataset", type=str, default="name")
    parse.add_argument("--name", type=str, default="name")
    parse.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers')
    parse.add_argument('--hidden_dims', type=int, default=[128, 128],
                       help='Output dimension for each hidden layer.')
    parse.add_argument('--latent_dim', type=int, default=50, help='output dimension for node embeddings')
    parse.add_argument('--dropout', type=float, default=[0.2, 0.2], help='Dropout for each layer')
    parse.add_argument('--num_heads', type=int, default=[3, 3, 3, 3],
                       help='Number of attention heads for each layer')
    parse.add_argument('--decoder_dim1', type=int, default=128,
                       help='First hidden dimension for the neural network decoder')
    parse.add_argument('--lr', type=int, default=1e-3, help='Learning rate of Adma')
    parse.add_argument('--max_epoch', type=int, default=100, help='Number of training epoch ')
    parse.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parse.add_argument('--update_interval', default=1, type=int)
    parse.add_argument('--num_clusters', type=int, default=7, help='Number of clusters')
    parse.add_argument('--clu_loss', type=float, default=1)
    parse.add_argument('--vgaa_loss', type=float, default=0.1)
    parse.add_argument('--zinb_loss', type=float, default=0.1)

    args = parse.parse_args()
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args = vars(args)

    df = pd.read_csv('Dataset/{}/{}_labels.csv'.format(args['dataset'], args['name']))
    true_lab = df['label'].values
    types = np.unique(true_lab)
    true_lab = np.array(true_lab)
    ids = np.arange(0, len(types))
    dict1 = {}
    dict1 = dict(zip(ids, types))
    for id, type in dict1.items():
        for i in range(len(true_lab)):
            if true_lab[i] == type:
                true_lab[i] = id
    X_hvg = np.load('Process/{}.npy'.format(args['name']))
    edges = load_separate_graph_edgelist('Process/{}_edgelist.txt'.format(args['name']))
    data_obj = create_graph(edges, X_hvg)
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    test_split = args['test_split']
    try:
        transform = Trans.RandomLinkSplit(num_test=test_split,
                                          is_undirected=True, add_negative_train_samples=True,
                                          split_labels=True)
        train_data,  test_data = transform(data_obj)
    except IndexError as ie:
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
    model = ZIVGADecoder(encoder=encoder,
                        decoder_dim1=args['decoder_dim1'])
    dec = DEC(num_clusters=num_clusters,
              model=model,
              latent_dims=latent_dim)
    optimizer = torch.optim.Adam(dec.parameters(), lr=args['lr'])
    train(dec=dec,
          optimizer=optimizer,
          train_data=train_data,
          device=device,
          true_label=true_lab)
    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print('cost of train: run time is %.2f ' % run_time, 'minutes')
