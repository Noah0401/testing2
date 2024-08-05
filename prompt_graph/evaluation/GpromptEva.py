import torch.nn.functional as F
import torch

def GpromptEva(loader, gnn, prompt, center_embedding, device):
    r"""Graph classification accuracy of Gprompt.

    Args:
        loader (DataLoader): The selected loader.
        gnn (model): The chosen GNN type.
        prompt (model): Used to perform some processing on the input data in preparation for passing it to the GNN model for processing.
        device (device): The chosen device.
        center_embedding (Tensor): The embedding of the center of the graph.

        """

    prompt.eval()
    correct = 0
    for batch in loader: 
        batch = batch.to(device) 
        out = gnn(batch.x, batch.edge_index, batch.batch, prompt, 'Gprompt')
        similarity_matrix = F.cosine_similarity(out.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
        pred = similarity_matrix.argmax(dim=1)
        correct += int((pred == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  