

def GPFEva(loader, gnn, prompt, answering, device):
    r"""Graph classification accuracy of GPF.

        Args:
            loader (DataLoader): The selected loader.
            gnn (model): The chosen GNN type.
            prompt (model): Used to perform some processing on the input data in preparation for passing it to the GNN model for processing.
            answering (model): The predicted answer for classification.
            device (device): The chosen device."""

    prompt.eval()
    if answering:
        answering.eval()
    correct = 0
    for batch in loader: 
        batch = batch.to(device) 
        batch.x = prompt.add(batch.x)
        out = gnn(batch.x, batch.edge_index, batch.batch)
        if answering:
            out = answering(out)  
        pred = out.argmax(dim=1)  
        correct += int((pred == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  