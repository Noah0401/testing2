
def GNNNodeEva(data, idx_test,  gnn, answering):
    r"""Node classification accuracy of GNN"""
    gnn.eval()
    out = gnn(data.x, data.edge_index, batch=None)
    out = answering(out)
    pred = out.argmax(dim=1) 
    correct = pred[idx_test] == data.y[idx_test]  
    acc = int(correct.sum()) / len(idx_test)  
    return acc

def GNNGraphEva(loader, gnn, answering, device):
    r"""graph classification accuracy of GNN"""
    gnn.eval()
    if answering:
        answering.eval()
    correct = 0
    for batch in loader: 
        batch = batch.to(device) 
        out = gnn(batch.x, batch.edge_index, batch.batch)
        if answering:
            out = answering(out)  
        pred = out.argmax(dim=1)  
        correct += int((pred == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  