import torch



def MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, Preprompt, DownPrompt, sp_adj):
    r"""Node classification accuracy of multi-Gprompt.

    Args:
        test_embs (Tensor): The embedding of the testing set.
        test_lbls (Tensor): The label of the testing set.
        idx_test (Tensor): The index of the testing set.
        prompt_feature (Tensor): The feature of the prompt.
        Preprompt (Tensor): Prompt for pre-train.
        DownPrompt (Tensor): Prompt for downstream task.
        sp_adj (Tensor): Sparse tensor.

    """
    embeds1, _ = Preprompt.embed(prompt_feature, sp_adj, True, None, False)
    test_embs1 = embeds1[0, idx_test]
    print('idx_test', idx_test)
    logits = DownPrompt(test_embs, test_embs1, test_lbls)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    return acc.cpu().numpy()